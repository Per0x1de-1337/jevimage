"""The `jevimage` command: try an encoder, train a head, see whether training helped.

A thin shell over the Python API - nothing is computed here that `jevimage.Jev` does
not already compute, so a number that looks wrong is wrong upstream, not in the
formatting. Plain ASCII on purpose: this output gets pasted into issues and logs.

    jevimage encoders
    jevimage ask photo.jpg --choice indoor="an indoor scene" outdoor="the outdoors"
    jevimage train street dataset/
    jevimage heads
    jevimage serve
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import re
import sys
import time
from pathlib import Path

from .client import JevError
from .store import env

# The errors a user causes: a typo'd encoder name, a head that was never trained, a path
# that is not there. A traceback for those tells them nothing they can act on.
EXPECTED = (ValueError, KeyError, OSError, JevError)

# Deliberately not 8000. That is the default of every other Python web server on the
# machine, which is how `jevimage serve` ended up racing a dashboard for the same
# socket. 8731 is unassigned; `serve` walks up from it when something has taken it
# anyway, and the probe in `_local_server` sweeps the same range so the two agree.
DEFAULT_PORT = 8731
PORT_TRIES = 10

# Where a key is good. Used when a key is set and no URL was given: a key names the
# service it belongs to, so holding one is the same as choosing it. $JEVIMAGE_URL still wins,
# which is how a self-hosted server, a staging one, or anything else is reached - so
# this needs no variable of its own.
DEFAULT_URL = "https://api.jevimage.org"


# -- formatting ----------------------------------------------------------------

def _table(rows, headers) -> None:
    """First column left, the rest right - so digits line up under each other."""
    grid = [[str(c) for c in r] for r in [headers] + list(rows)]
    w = [max(len(r[i]) for r in grid) for i in range(len(headers))]
    for r in grid:
        print("  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i])
                        for i, c in enumerate(r)).rstrip())


def _pct(x) -> str:
    """None means "not measured" - k-fold reports no accuracy when a class was too thin
    to hold anything out. Printing 0.0% there would read as "it got everything wrong"."""
    return "-" if x is None else f"{x * 100:5.1f}%"


def _bar(p, width=22) -> str:
    return "#" * round(p * width)


def _show(qid, a) -> None:
    if a["type"] == "noul":
        # confidence and the cannot-answer warning only exist on the grid reading, and the
        # engine sends one or the other
        extra = (f", confidence {a['confidence']:.3f}" if a.get("confidence") is not None
                 else "")
        print(f"{qid}  [noul]  {a['noul']:.3f}   ({a['basis']}{extra})")
        if a.get("warning"):
            print(f"    ! {a['warning']}")
        print()
        return

    if a["type"] == "score":
        head = f"{a['score']:.2f} / {len(a['legend']) - 1}"
    else:
        head = f"{a['choice']}"
    head += f"   confidence {a['confidence']:.3f}"
    if a.get("readout") == "grid":
        head += "   (read on the patch grid)"
    if a["type"] == "head":
        # `accuracy` is what jevimage's own server calls it; a hosted one may publish
        # the same number as `cv_accuracy`. Reading only one turned a perfectly good
        # answer into a KeyError whose whole text was the key name. Both are optional:
        # a head too thin to hold anything out has no accuracy to report at all.
        acc = a.get("accuracy", a.get("cv_accuracy"))
        n = a.get("trained_on")
        head += (f"   (head {a['head']!r}"
                 + (f", {n} examples" if n is not None else "")
                 + (f", held out {_pct(acc).strip()}" if acc is not None else "")
                 + ")")
    print(f"{qid}  [{a['type']}]  {head}")

    # a score's keys are level numbers, which mean nothing without the rubric text
    legend = a.get("legend", {})
    width = max(len(legend.get(k, k)) for k in a["probabilities"])
    for k, p in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"    {legend.get(k, k):<{width}}  {p:6.3f}  {_bar(p)}")
    print()


# -- questions -----------------------------------------------------------------

def _uid(taken, base) -> str:
    if base not in taken:
        return base
    i = 2
    while f"{base}{i}" in taken:
        i += 1
    return f"{base}{i}"


def _questions(args) -> dict:
    """The --choice/--score/--noul/--head flags -> the {id: question} ask() takes.

    Each flag repeats, because asking one image several questions is the whole point:
    they share the single encode.
    """
    qs: dict[str, dict] = {}
    for options in args.choice or []:
        criteria = {}
        for opt in options:
            name, sep, description = opt.partition("=")
            if not sep or not name:
                raise ValueError(
                    f"--choice takes name=description pairs; got {opt!r}. Try: "
                    '--choice indoor="an indoor scene" outdoor="the outdoors"')
            if name in criteria:
                # the dict would eat the earlier pair, leaving a distribution over fewer
                # options than were typed - the same refusal core.plan() makes for the
                # list form, which never sees this one because the CLI builds the dict
                raise ValueError(f"--choice repeats the name {name!r}; each option "
                                 "needs its own name")
            criteria[name] = description
        qs[_uid(qs, "choice")] = {"type": "choice", "criteria": criteria}
    for levels in args.score or []:
        qs[_uid(qs, "score")] = {"type": "score",
                                 "criteria": [s.strip() for s in levels.split(",")]}
    for texts in args.noul or []:
        # two captions build the pair form, which works on every encoder; one builds the
        # sigmoid readout, which needs an encoder that publishes scale AND bias
        if len(texts) > 2:
            raise ValueError(f"--noul takes one caption, or two for a true/false pair; "
                             f"got {len(texts)}: {texts!r}")
        q = ({"type": "noul", "criteria": {"true": texts[0], "false": texts[1]}}
             if len(texts) == 2 else {"type": "noul", "instructions": texts[0]})
        qs[_uid(qs, "noul")] = q
    for name in args.head or []:
        qs[_uid(qs, name)] = {"type": "head", "head": name}
    if not qs:
        raise ValueError('ask needs at least one question: --choice name="a description" '
                         '..., --score "low,high", --noul "..." or --head NAME')
    return qs


# -- backend -------------------------------------------------------------------

def _opt(args, name, envname):
    """A flag's value, else the environment, else None.

    `is None`, not truthiness: `--timeout 0` is a value the user typed, and the flags are
    documented to beat their variables. A flag that was passed but is empty - the shape
    `--url "$SERVER"` takes when $SERVER is unset - is a mistake, not a request for the
    default, and on `rm` that mistake silently retargets a delete at the local directory.
    """
    v = getattr(args, name, None)
    if v is None:
        return env(envname) or None
    if isinstance(v, str) and not v.strip():
        raise ValueError(f"--{name.replace('_', '-')} was given but is empty")
    return v


def _remote(args):
    """The server to run against, or None for local. One resolution for every command
    that branches on it, so `--url` means the same thing to `rm` as to `ask`."""
    return _opt(args, "url", "JEVIMAGE_URL")


def _local_server(timeout=0.25):
    """A `jevimage serve` already running here, or None.

    Loading the encoder costs about twelve seconds and the question itself costs twenty
    milliseconds, so a command that loads its own copy spends 99.8% of its life on
    startup - and on a small GPU it also collides with the server it ignored. If one is
    already listening locally, use it.

    The whole range is swept, not just `DEFAULT_PORT`, because `serve` walks up when its
    port is taken and a server we cannot find is a server we start a second copy of. A
    closed port on loopback refuses immediately rather than waiting out the timeout, so
    the sweep costs about a millisecond when nothing is running.
    """
    import http.client

    for port in range(DEFAULT_PORT, DEFAULT_PORT + PORT_TRIES):
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            c.request("GET", "/v1/health")
            r = c.getresponse()
            ok = r.status == 200 and b'"primitives"' in r.read(400)
            c.close()
        except Exception:
            continue
        if ok:
            return f"http://127.0.0.1:{port}"
    return None


def _config_path():
    """`~/.jev/config.json`, or under `$JEVIMAGE_HOME` if that is set."""
    return Path(env("JEVIMAGE_HOME") or Path.home() / ".jev") / "config.json"


def _config() -> dict:
    """What `jevimage login` saved, or {}.

    Exists because the environment is the wrong place to keep this. A shell that has
    not exported `$JEVIMAGE_API_KEY` is indistinguishable from someone with no account, so
    the same command in two terminals reached two different servers and listed two
    different sets of heads. The browser does not ask you to re-authenticate per tab.
    """
    try:
        with open(_config_path()) as f:
            c = json.load(f)
        return c if isinstance(c, dict) else {}
    except (OSError, ValueError):
        return {}


def _api_key(args=None) -> str | None:
    """The key to send: the environment first, then whatever `login` saved."""
    return env("JEVIMAGE_API_KEY") or _config().get("key") or None


def cmd_login(args) -> int:
    """Save a key and a server so every shell agrees, instead of one terminal knowing
    who you are and the next one not."""
    import stat

    key = args.key or env("JEVIMAGE_API_KEY")
    if not key and sys.stdin.isatty():
        import getpass
        key = getpass.getpass("API key (not echoed): ").strip()
    elif not key:
        key = sys.stdin.readline().strip()
    if not key:
        raise ValueError("no key given. Pass --key, pipe it in, or set $JEVIMAGE_API_KEY. "
                         "Keys come from https://console.jevimage.org")

    url = (args.url or env("JEVIMAGE_URL") or DEFAULT_URL).rstrip("/")
    jev = jevimage_connect(url, key)
    who = jev.health()                       # fail now, not on their next command

    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # written 0600 before anything goes in it: a secret must never exist, even briefly,
    # at whatever the umask happens to be
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as f:
        json.dump({"url": url, "key": key}, f)
    print(f"signed in to {url}")
    print(f"  model {who.get('encoder')}, {who.get('heads', 0)} head(s) you can use")
    print(f"  saved to {path} (0600). `jevimage logout` removes it.")
    return 0


def cmd_logout(args) -> int:
    path = _config_path()
    try:
        os.unlink(path)
    except FileNotFoundError:
        print("not signed in")
        return 0
    print(f"removed {path}")
    return 0


def jevimage_connect(url, key):
    import jevimage
    return jevimage.connect(url, api_key=key)


def _server_for(args):
    """The server this command should use, or None to do the work in this process.

    In order:

      --local              this machine: a `jevimage serve` running here if there is
                           one, otherwise this process
      --url / $JEVIMAGE_URL     that server
      --encoder / $JEVIMAGE_ENCODER  this process: you named an encoder, so you meant one
      $JEVIMAGE_API_KEY set     the hosted API - a key is only good against it
      a `jevimage serve` running here
      this process

    A key is the signal for the hosted API rather than a hardcoded default, because
    without one that default is a 401 and this package is self-hostable. Someone with a
    key is a customer of the service and expects `jevimage heads` to list what they
    trained on it; someone without one is running their own server or their own
    weights, and pointing them at an endpoint they cannot authenticate to would break a
    working install to make a sale.
    """
    if getattr(args, "local", False):
        # "local" is about the machine, not the process. Someone who has a server
        # running here and asks for local means that server, not twelve seconds of
        # reloading the same weights beside it. `--encoder NAME` is the way to insist
        # on this process, and it reads as what it does.
        return _local_server()
    url = _remote(args)
    if url:
        return url
    if _opt(args, "encoder", "JEVIMAGE_ENCODER"):
        return None
    if env("JEVIMAGE_API_KEY"):
        return DEFAULT_URL
    saved = _config()
    if saved.get("key"):
        return saved.get("url") or DEFAULT_URL
    return _local_server()


def _backend(args):
    """A remote instance when --url (or $JEVIMAGE_URL) is set, otherwise a local one.

    The two expose the same methods, so every command below is written once and works
    either way - which is the whole point of `connect` mirroring `load`.
    """
    import jevimage

    url = _server_for(args)
    if url and url != _remote(args):
        # say which rule picked it, because the two look identical in the output and
        # lead to different heads. Not printed when a URL was named: you know already.
        if url == (_config().get("url") or DEFAULT_URL) and _config().get("key") \
                and not env("JEVIMAGE_API_KEY"):
            why, how = "signed in", "--local for this machine"
        elif env("JEVIMAGE_API_KEY"):
            why, how = "$JEVIMAGE_API_KEY is set", "--local for this machine"
        else:
            why = "a jevimage serve is running here"
            how = "--encoder NAME to load the weights in this process"
        print(f"using {url}   ({why}; {how})", file=sys.stderr)
    if url:
        # training a few thousand images through a server routinely outlives the 60s
        # client default, and a job that cannot be waited for cannot be run
        raw = _opt(args, "timeout", "JEVIMAGE_TIMEOUT")
        raw = 60.0 if raw is None else raw
        where = "--timeout" if getattr(args, "timeout", None) is not None else "$JEVIMAGE_TIMEOUT"
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{where}={raw!r} is not a number of seconds") from None
        if timeout <= 0:
            raise ValueError(f"{where} must be a positive number of seconds; got {raw}")
        return jevimage.connect(url, api_key=_api_key(args), timeout=timeout)
    name = _opt(args, "encoder", "JEVIMAGE_ENCODER")
    if not name:
        # Nothing selected a server and nothing named an encoder, so this is about to
        # download weights and run them here. Someone who passed --encoder meant that;
        # someone who passed nothing usually believed they had pointed at the API, and
        # the commonest reason they had not is that `JEVIMAGE_URL=...` on its own line sets a
        # SHELL variable, which a child process never sees. Silence there costs a 400MB
        # download and a confusing answer from the wrong place.
        print("no server selected, so the weights will run in this process "
              "(first use downloads them).\n"
              "   for the hosted API:  export JEVIMAGE_URL=https://api.jevimage.org\n"
              "                        export JEVIMAGE_API_KEY=jev_...\n"
              "   `export` matters: a bare VAR=value is not passed to this command.",
              file=sys.stderr)
    if name:
        # Checked against the registry first, which reads without torch. Otherwise a
        # typo on an API-only install answered "install torch": you would install it,
        # run the same command, and only then be told the name was wrong.
        from . import encoders

        encoders.check(name)
    return jevimage.load(name)


# Every variable anything in the package reads. A name that is not here is read by
# nobody, which is the quietest way a setting can fail.
ENV_VARS = ("JEVIMAGE_URL", "JEVIMAGE_API_KEY", "JEVIMAGE_API_KEYS",
            "JEVIMAGE_TIMEOUT", "JEVIMAGE_ENCODER", "JEVIMAGE_DEVICE",
            "JEVIMAGE_THREADS", "JEVIMAGE_BATCH", "JEVIMAGE_HOME",
            "JEVIMAGE_HEADS_DIR", "JEVIMAGE_GRID")


def _check_env() -> None:
    """Say something about a JEVIMAGE_ variable nothing reads.

    `export JEVIMAGE_BASE=...` is not an error anywhere: it is never looked at, and the
    command carries on with the default as though nothing had been asked for. One line
    costs nothing and saves the half hour spent wondering why the setting did not take.

    Both prefixes are checked, because both are read. The old `JEV_` names still work,
    so `JEVIMAGE_URL` must NOT be reported as unread - that would send someone to fix a
    setting that is working. Only a name whose suffix nothing knows is reported, and the
    suggestion is offered under the prefix the reader actually typed.

    One line, though. Listing all nine every time turned a note about one variable into
    a wall, and the reader has to find their own name in it. A suggestion is offered
    only when it is nearly certain - `JEVIMAGE_TIMOUT` for `JEVIMAGE_TIMEOUT` - because
    difflib will happily answer `JEVIMAGE_API_KEY` for `JEVIMAGE_BASE`, and a confident
    wrong guess costs more than no guess.
    """
    import difflib

    known = {v[len("JEVIMAGE_"):] for v in ENV_VARS}
    for name in sorted(os.environ):
        for prefix in ("JEVIMAGE_", "JEV_"):
            if not name.startswith(prefix):
                continue
            if name[len(prefix):] in known:
                break                      # read under one prefix or the other
            near = difflib.get_close_matches(prefix + name[len(prefix):],
                                             [prefix + k for k in sorted(known)], 1, 0.85)
            hint = (f" Did you mean ${near[0]}?" if near
                    else " `jevimage doctor` shows the ones that are.")
            print(f"jevimage: ${name} is set, but nothing reads it.{hint}",
                  file=sys.stderr)
            break


def _where(jev) -> str:
    return (f"server {jev.url}" if hasattr(jev, "url")
            else f"encoder {jev.encoder.name}")


# -- commands ------------------------------------------------------------------

def _age_ms() -> float | None:
    """How long this process has been alive, including the interpreter's own start.

    `perf_counter()` from the top of this module misses everything python did before
    reaching it, which is most of what a short command spends. Linux keeps the real
    start time in /proc; anywhere else this is None and the line is left out rather
    than guessed at.
    """
    try:
        with open("/proc/self/stat") as f:
            started = int(f.read().rpartition(")")[2].split()[19])
        with open("/proc/uptime") as f:
            up = float(f.read().split()[0])
        return (up - started / os.sysconf("SC_CLK_TCK")) * 1000
    except (OSError, ValueError, IndexError):
        return None


def _describe(src) -> str:
    """"2560x1440 RGBA png", from the header alone.

    `Image.open` reads the header and stops, so this costs microseconds. Decoding it
    here to ask its size would double the very number the line is reporting.
    """
    try:
        from PIL import Image
        with Image.open(src) as im:
            return f"{im.width}x{im.height} {im.mode} {(im.format or '').lower()}"
    except Exception:
        return "image"


def _timings(rows):
    """The `--time` breakdown, on stderr so `--json` stays machine-readable.

    Printed after the answer, not before it: the answer is what was asked for. stdout is
    flushed first, or a pipe holds the answer back and the two arrive out of order.
    """
    sys.stdout.flush()
    print("", file=sys.stderr)
    for label, ms, note in rows:
        print(f"  {label:<10}{ms:7.0f} ms   {note}", file=sys.stderr)
    age = _age_ms()
    if age is None:
        return          # no /proc: the stages above are still true, the total is not
    rest = age - sum(ms for _, ms, _ in rows)
    print(f"  {'startup':<10}{rest:7.0f} ms   python, argparse, and importing pillow "
          f"and http.client", file=sys.stderr)
    print(f"  {'':<10}{'':7}      ----", file=sys.stderr)
    print(f"  {'total':<10}{age:7.0f} ms", file=sys.stderr)


def cmd_ask(args) -> int:
    questions = _questions(args)            # validated before any weights load

    t0 = time.perf_counter()
    jev = _backend(args)
    load = (time.perf_counter() - t0) * 1000

    rows, out = ([] if args.time else None), {}
    if rows is not None and load > 50:
        # only when it is the model. Reaching a server that is already up is a connect()
        # and a health probe, and a line saying 1ms is noise.
        rows.append(("load", load, f"{_where(jev)}, loaded into this process"))

    # Every image goes over one connection, which is the whole point of taking more than
    # one. Against a server a round trip away, TCP and TLS cost about half a second per
    # process and the question itself costs a tenth of that - so ten files in ten
    # commands pay the handshake ten times, and ten files in one command pay it once.
    for path in args.image:
        t1 = time.perf_counter()
        image = path
        if rows is not None and getattr(jev, "max_edge", None):
            from .client import _data_url
            image = _data_url(path, jev.max_edge)
            rows.append(("image", (time.perf_counter() - t1) * 1000,
                         f"{_describe(path)} -> {len(image) * 3 // 4096} KB"))
        t2 = time.perf_counter()
        out[path] = jev.ask(image, questions)
        if rows is not None:
            rows.append(("answer", (time.perf_counter() - t2) * 1000, _where(jev)))

    if args.json:
        print(json.dumps(out if len(out) > 1 else next(iter(out.values())), indent=2))
    else:
        for i, (path, answers) in enumerate(out.items()):
            print(f"{'' if i == 0 else chr(10)}{path}   {_where(jev)}\n")
            for qid, a in answers.items():
                _show(qid, a)
    if rows is not None:
        _timings(rows)
    return 0


def cmd_train(args) -> int:
    jev = _backend(args)
    head = jev.train(args.name, args.folder, epochs=args.epochs)
    cmp = head.comparison

    if args.json:
        out = {k: v for k, v in head.to_dict().items()}
        out["comparison"] = cmp
        print(json.dumps(out, indent=2))
        return 0

    acc = (cmp or {}).get("overall", {}).get("trained")
    print(f"trained {head.name!r}: {head.n} examples, {len(head.classes)} classes"
          + (f", {_pct(acc).strip()} accurate" if acc is not None else ""))
    print(f"  classes: {', '.join(head.classes)}")
    print(f"  encoder: {head.encoder}")
    print(f"\nUse it:\n  jevimage ask IMAGE --head {head.name}")

    # The zero-shot comparison is the evidence for whether training was worth doing at
    # all, and on classes an encoder already knows the honest answer is often "no". It
    # is behind a flag rather than gone: printed by default it buried the one line most
    # people came for, which is that the head exists and how to call it.
    if not args.compare:
        return 0
    if not cmp:
        print("\n(no zero-shot comparison for this head - it was saved without the "
              "embeddings the comparison is computed from)")
        return 0
    print()
    rows = [[r["label"], r["n"], _pct(r["zero_shot"]), _pct(r["trained"]),
             _pct(None if r["trained"] is None else r["trained"] - r["zero_shot"])]
            for r in cmp["per_class"]]
    o = cmp["overall"]
    rows.append(["ALL", head.n, _pct(o["zero_shot"]), _pct(o["trained"]),
                 _pct(o["delta"])])
    _table(rows, ["class", "n", "zero-shot", "trained", "delta"])
    print(f"\n{cmp['basis']}.")
    if o["delta"] is not None and o["delta"] <= 0:
        print("Training did not beat the prompts here - try more examples per class, "
              "or a bigger encoder.")
    return 0


def _whose(jev) -> str:
    """"  (account X)" when the server names one, else "".

    A key scopes what you can see, so two keys list two different sets of heads. Without
    this, a head trained in the browser and missing from `jevimage heads` looks like a
    sync failure rather than a second account.
    """
    if not hasattr(jev, "url"):
        return ""
    try:
        acct = (jev.health() or {}).get("account") or {}
    except Exception:
        return ""
    who = acct.get("login") or acct.get("id")
    return f"   (account {who})" if who else ""


def cmd_heads(args) -> int:
    """Reads the head directory directly: listing what you trained should not download
    an encoder's weights."""
    broken = {}
    # bound for the local path too: the footer below calls _whose(jev) unconditionally,
    # and _whose already returns "" for anything without a .url, so the only thing
    # missing was a value. Without this, `jevimage heads` on a machine with local heads
    # and no server died with UnboundLocalError instead of printing the table it had
    # already built.
    jev = None
    if _server_for(args):
        jev = _backend(args)
        heads, where = jev.list_heads(), jev.url
    else:
        # imported here, not at the top: reading a head file needs torch, and listing a
        # SERVER's heads must not - that path exists precisely for installs without it
        from .store import HeadStore
        from .training import public

        store = HeadStore()
        heads, where, broken = ([public(n, h) for n, h in sorted(store.items())],
                                store.dir, store.broken)
    if args.json:
        # broken heads go in the payload too: a script that only reads stdout was the
        # one caller that could not see the warning the table prints
        out = {"heads": heads, "directory": str(where)}
        if broken:
            out["broken"] = {n: str(e) for n, e in broken.items()}
        print(json.dumps(out, indent=2))
        return 0
    for name, why in sorted(broken.items()):
        # a file sitting right there that did not load must not simply vanish from the
        # listing: a later "no head named x" about it would be a lie
        print(f"{name}.pt is present but unreadable: {why}", file=sys.stderr)
    if not heads:
        print(f"no heads in {where}. Train one: jevimage train NAME FOLDER")
        return 0
    # .get, not [] - a hosted server publishes the same heads with a smaller record
    # (no encoder name, no created timestamp), and a listing that raises KeyError on a
    # column tells you nothing about the heads that are plainly there
    _table([[h.get("name", "?"), h.get("encoder") or "-", len(h.get("classes") or []),
             h.get("n", "-"), _pct(h.get("cv_accuracy")),
             time.strftime("%Y-%m-%d", time.localtime(h["created"]))
             if h.get("created") else "-"]
            for h in heads],
           ["head", "encoder", "classes", "n", "held out", "created"])
    print(f"\n{where}{_whose(jev)}")
    return 0


def cmd_rm(args) -> int:
    if _remote(args):
        jev = _backend(args)
        if not jev.delete_head(args.name):
            raise KeyError(f"no head named {args.name!r} on {jev.url}")
    else:
        # Not a HeadStore: constructing one syncs the directory, which imports torch and
        # loads every head, and deleting a file never reads it. Going straight to the
        # path is what makes this the one local command an API-only install can run.
        from .store import default_dir, valid_name

        d = default_dir()
        try:
            (d / f"{valid_name(args.name)}.pt").unlink()
        except FileNotFoundError:
            raise KeyError(f"no head named {args.name!r} in {d}") from None
    print(f"deleted {args.name!r}")
    return 0


def cmd_encoders(args) -> int:
    from . import encoders

    override = env("JEVIMAGE_ENCODER")
    for name in encoders.available():
        mark = "*" if name == (override or encoders.DEFAULT) else " "
        print(f" {mark} {name}")
    note = f"$JEVIMAGE_ENCODER={override}" if override else f"default ({encoders.DEFAULT})"
    print(f"\n* {note}. Join names with '+' for an ensemble, e.g. "
          f"{encoders.DEFAULT}+dfn5b-h-14-384.")
    return 0


def gpu_hogs(limit: int = 4) -> list[str]:
    """What is holding the GPU, as `pid MiB name (state)` lines. Empty if we cannot tell.

    A card that is full is the commonest reason a server ends up on the CPU, and "0.1 GB
    free of 3.7 GB" does not tell you what to close. torch knows how much is free but
    not who has it, so this asks nvidia-smi, which is installed wherever CUDA is. A
    stopped process is called out by name: a suspended server holds its allocation for
    as long as it exists, and looks like nothing at all in `top`.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    rows = []
    for line in out.splitlines():
        pid, _, mib = line.partition(",")
        pid, mib = pid.strip(), mib.strip()
        if not pid.isdigit():
            continue
        name, state = "?", ""
        try:
            with open(f"/proc/{pid}/cmdline") as f:
                # basenames only: the interpreter's absolute path is 40 characters of
                # nothing, and what you need to recognise the process is the script
                argv = [a for a in f.read().split("\0") if a]
                name = " ".join(a.rsplit("/", 1)[-1] for a in argv) or name
            with open(f"/proc/{pid}/stat") as f:
                # field 3, after the comm in parentheses, which may itself contain spaces
                code = f.read().rpartition(")")[2].split()[0]
            state = {"T": "  <- STOPPED, still holding it",
                     "Z": "  <- zombie"}.get(code, "")
        except OSError:
            pass
        rows.append((int(mib or 0), f"pid {pid:<8}{mib:>6} MiB  {name[:52]}{state}"))
    rows.sort(reverse=True)
    return [r for _, r in rows[:limit]]


def _verdict(usable: bool, local_ok: bool, server_ok: bool):
    """The closing lines of `doctor`, and its exit code.

    Separate from the printing because the rule is not obvious: a machine with no torch
    is not broken, it is an API-only install, and it only fails when there is also no
    server for it to call.
    """
    if not usable:
        return ["fix the lines marked NO first - nothing works without them"], 1
    if local_ok:
        return ["ready: this machine can answer questions on its own"
                + (" and a server is reachable" if server_ok else "")], 0
    if server_ok:
        return ["ready: API-only - this machine calls the server above and needs no "
                "model stack"], 0
    return ["not ready: no local model stack and no server to call",
            "pick one: pip install 'jevimage[local]', or set $JEVIMAGE_URL to a server you "
            "can reach"], 1


def cmd_doctor(args) -> int:
    """What this machine can and cannot do, and the one line that fixes each gap.

    Every failure here is an install missing a piece, and the error arrives from deep
    inside someone else's library - torch raises about cublasCreate, transformers about
    a hub connection. One place that says "this is present, this is not, run this" beats
    improving each of those messages individually.

    There are two supported installs and they need different things: calling a server
    needs no torch at all, running models locally needs the whole stack. So a missing
    piece is only a failure if it leaves nothing that can answer a question - an
    API-only machine with no torch is correct, not broken.
    """
    import importlib.util as iu
    import platform
    import shutil

    import jevimage

    def here(label, mark, detail, fix=""):
        print(f"  {mark:<4}{label:22} {detail}")
        if fix:
            print(f"      {'':22} -> {fix}")

    print(f"jevimage {jevimage.__version__} on python {platform.python_version()} "
          f"({platform.system()} {platform.machine()})")
    print()

    old_python = sys.version_info < (3, 10)
    here("python >= 3.10", "NO" if old_python else "yes", platform.python_version(),
         "jevimage needs 3.10 or newer" if old_python else "")

    has_pil = iu.find_spec("PIL") is not None
    here("pillow", "yes" if has_pil else "NO", "present" if has_pil else "missing",
         "" if has_pil else "pip install --force-reinstall jevimage")

    # Does a server answer? Asked first because it decides whether the local stack is
    # required or merely optional.
    running = _local_server()
    url = _remote(args) or running
    health = None
    if url:
        try:
            health = jevimage.connect(
                url, api_key=_api_key()).health()
            acct = (health.get("account") or {})
            who = acct.get("login") or acct.get("id")
            here("server", "yes", f"{url} - {health.get('encoder')} on "
                                  f"{health.get('device')}, {health.get('heads', 0)} "
                                  f"head(s), auth {health.get('auth')}"
                                  + (f", account {who}" if who else ""))
        except Exception as e:
            here("server", "NO", f"{url} did not answer ({str(e)[:60]})",
                 "start one with `jevimage serve`, or unset $JEVIMAGE_URL")
    else:
        here("server", "--", "none running here, and $JEVIMAGE_URL is not set")

    has_torch = iu.find_spec("torch") is not None
    has_tf = iu.find_spec("transformers") is not None
    local_ok = has_torch and has_tf
    here("run models locally", "yes" if local_ok else "--",
         "torch + transformers present" if local_ok else
         f"torch {'yes' if has_torch else 'no'}, "
         f"transformers {'yes' if has_tf else 'no'}",
         "" if local_ok else "pip install 'jevimage[local]'" +
         ("" if health else "   - or point $JEVIMAGE_URL at a server instead"))

    if local_ok:
        import torch
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                tight = free <= 1.2 * 1024 ** 3
                here("gpu", "yes", f"{torch.cuda.get_device_name(0)}, "
                                   f"{free / 1024 ** 3:.1f} GB free of "
                                   f"{total / 1024 ** 3:.1f} GB",
                     "" if not tight else
                     "too tight for a model; it will load on the CPU instead, which is "
                     "about ten times slower")
                if tight:
                    for row in gpu_hogs() or ["(nvidia-smi could not say what has it)"]:
                        print(f"      {'':22}    {row}")
            except Exception as e:
                here("gpu", "--", f"present but unusable "
                                  f"({str(e).splitlines()[0][:50]})",
                     "JEVIMAGE_DEVICE=cpu skips it")
        else:
            here("gpu", "--", "none detected; the CPU is used - slower, works anywhere")

        # The CPU path is what most machines end up on, and torch's default thread count
        # is wrong on a hybrid chip by a factor of nearly two. Worth saying out loud.
        from . import encoders
        fast = encoders.fast_cores()
        pick = env("JEVIMAGE_THREADS") or (fast and str(fast))
        if pick and int(pick) > 0 and int(pick) != torch.get_num_threads():
            here("cpu threads", "yes",
                 f"{pick} of {os.cpu_count()}, the performance cores - torch would "
                 f"use {torch.get_num_threads()} and wait on the slow ones "
                 f"($JEVIMAGE_THREADS overrides, 0 to leave it alone)")
        else:
            here("cpu threads", "--",
                 f"{torch.get_num_threads()} of {os.cpu_count()}, all the same speed")

        try:
            from huggingface_hub import constants
            cache = constants.HF_HUB_CACHE
        except Exception:
            cache = os.path.expanduser("~/.cache/huggingface/hub")
        cached = [d for d in os.listdir(cache)] if os.path.isdir(cache) else []
        cached = [d for d in cached if d.startswith("models--")]
        here("weights cached", "yes" if cached else "--",
             f"{len(cached)} model(s) in {cache}" if cached else f"nothing in {cache}",
             "" if cached else "the first local run downloads about 1.5 GB")

        free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 1024 ** 3
        here("disk", "yes" if free_gb > 3 else "NO",
             f"{free_gb:.1f} GB free in your home directory",
             "" if free_gb > 3 else "weights need about 2 GB")

    has_srv = (iu.find_spec("fastapi") is not None
               and iu.find_spec("uvicorn") is not None)
    here("host a server", "yes" if has_srv else "--",
         "fastapi + uvicorn present" if has_srv else "not installed",
         "" if has_srv else "pip install 'jevimage[serve]' - only to run "
                            "`jevimage serve` yourself")

    # what this machine is configured with, which is the question `--help` used to
    # answer with a wall of names most people will never set. Only what is actually
    # set, and never the value: one of these is a key.
    on = [(n, os.environ[n]) for n in ENV_VARS if os.environ.get(n)]
    saved = _config()
    if on or saved.get("key"):
        print()
        if saved.get("key"):
            here(f"signed in", "yes", f"{saved.get('url') or DEFAULT_URL}"
                                      f"   ({_config_path()}, jevimage logout to forget)")
        for n, v in on:
            shown = "set" if "KEY" in n else v
            here(f"${n}", "yes", f"{shown[:60]}")
    else:
        print()
        here("configuration", "--", "nothing set; `jevimage login`, or $JEVIMAGE_URL for "
                                    "your own server")

    print()
    lines, code = _verdict(has_pil and not old_python, local_ok, health is not None)
    for line in lines:
        print(f"  {line}")
    return code


def _bind(host: str, port: int, tries: int = PORT_TRIES):
    """The first free port from `port`, as an already-bound socket.

    Bound here rather than tested here: a "is 8731 free?" check followed by a separate
    listen is a race that loses to whatever grabbed the port in between, and loses it as
    the `[Errno 98] address already in use` this exists to avoid. uvicorn is handed the
    socket itself, so nothing can take it after we have it.
    """
    import socket

    for p in range(port, port + tries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, p))
            # Listening, not just bound: SO_REUSEADDR lets two *bound* sockets share a
            # port on Linux, and it is only the listen that makes the port ours and the
            # next attempt fail. uvicorn raises the backlog to its own when it takes
            # the socket over.
            sock.listen(128)
        except OSError as e:
            sock.close()
            if e.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise       # a bad host or a permission wall is not fixed by +1
            continue
        # getsockname, not p: `--port 0` asks the kernel to choose, and reporting the 0
        # back would print a URL nobody can connect to.
        return sock, sock.getsockname()[1]
    raise OSError(f"ports {port}-{port + tries - 1} on {host} are all in use; "
                  f"pass --port to pick one yourself")


def cmd_serve(args) -> int:
    # Arguments are checked BEFORE uvicorn is imported. With the import first, a bad
    # --api-key on an install without uvicorn answered "install uvicorn" - so you
    # installed it, ran the same command, and only then learned the key had a space in
    # it. What is wrong with the command does not depend on an optional dependency.

    # --api-key repeats as well as taking several values, because `--api-key old --api-key
    # new` is the natural spelling of the rotation the flag exists for, and dropping the
    # first one is exactly the downtime it is meant to avoid.
    keys = [k for group in (args.api_key or []) for k in group]
    if args.api_key is not None:
        for k in keys:
            # $JEVIMAGE_API_KEYS is split on commas AND whitespace, so a key containing either
            # authenticates as several shorter fragments while the real key gets a 401.
            if not k.strip():
                raise ValueError("--api-key was given an empty key, which would start an "
                                 "OPEN server; pass a real key, or omit the flag to serve "
                                 "openly on purpose")
            if re.search(r"[\s,]", k):
                raise ValueError(f"--api-key {k!r}: a key cannot contain a space or a "
                                 "comma, because $JEVIMAGE_API_KEYS is split on both")
    for flag, value in (("--encoder", args.encoder), ("--heads-dir", args.heads_dir)):
        if value is not None and not value.strip():
            raise ValueError(f"{flag} was given but is empty")
    if args.heads_dir and Path(args.heads_dir).exists() \
            and not Path(args.heads_dir).is_dir():
        # otherwise the server starts, reports healthy, and 500s on the first train
        raise ValueError(f"--heads-dir {args.heads_dir!r} is a file, not a directory")

    try:
        import uvicorn
    except ImportError:
        raise ImportError("jevimage serve needs uvicorn: pip install 'jevimage[serve]'")

    # Passed through the environment rather than to the app object, so the same knobs
    # work when a container or process manager starts uvicorn itself.
    for var, value in (("JEVIMAGE_ENCODER", args.encoder), ("JEVIMAGE_HEADS_DIR", args.heads_dir),
                       ("JEVIMAGE_API_KEYS", ",".join(keys))):
        if value:
            os.environ[var] = value
    from .server import app     # touching it builds the app, so the encoder loads here

    want = DEFAULT_PORT if args.port is None else args.port
    sock, port = _bind(args.host, want)
    if want and port != want:
        # `want` of 0 asked the kernel to choose, so the port it chose is not a
        # surprise; uvicorn's own banner names it either way.
        print(f"port {want} is in use; serving on {port} instead", file=sys.stderr)
    uvicorn.Server(uvicorn.Config(app, host=args.host, port=port)).run(sockets=[sock])
    return 0


# -- argv ----------------------------------------------------------------------

def _port(s: str) -> int:
    """The kernel takes the low 16 bits of anything bigger, so `--port 99999` listened on
    34463 while uvicorn's banner announced 99999."""
    p = int(s)
    if not 0 <= p <= 65535:
        raise argparse.ArgumentTypeError(f"port {p} is outside 0-65535")
    return p


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jevimage", description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # the environment table lived here and made the help twice as long as the thing
        # it documents. `--help` answers "what can this do"; `doctor` answers "how is
        # this set up", and that is where configuration belongs.
        epilog="Configuration: jevimage doctor        Docs: https://docs.jevimage.org")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="serve the HTTP API (needs uvicorn)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=_port, default=None,
                   help=f"default {DEFAULT_PORT}; if it is taken, the next free port up "
                        f"to {DEFAULT_PORT + PORT_TRIES - 1} is used")
    s.add_argument("--encoder")
    s.add_argument("--heads-dir")
    s.add_argument("--api-key", nargs="+", action="append", metavar="KEY",
                   help="require one of these keys; without it the API is open")
    s.set_defaults(fn=cmd_serve)

    a = sub.add_parser("ask", help="ask one image typed questions")
    a.add_argument("image", nargs="+", metavar="IMAGE",
                   help="one or more; several share one connection, which "
                        "is where the time goes against a remote server")
    a.add_argument("--choice", nargs="+", action="append", metavar="NAME=DESCRIPTION",
                   help='pick one, e.g. --choice indoor="an indoor scene" outdoor="outside"')
    a.add_argument("--score", action="append", metavar="LOW,...,HIGH",
                   help='ordered rubric, e.g. --score "empty,busy,packed"')
    a.add_argument("--noul", nargs="+", action="append", metavar="TEXT",
                   help='yes/no probability: two captions for a true/false pair (works '
                        'on every encoder), or one for the sigmoid readout (needs a '
                        'calibrated encoder). Two captions that are the same words in a '
                        'different order are read off the patch grid instead of the '
                        'pooled vector, which is the only way to separate them')
    a.add_argument("--head", action="append", metavar="NAME", help="run a trained head")
    a.add_argument("--encoder")
    a.add_argument("--url", help="run against a `jevimage serve` instance instead of local "
                                 "weights (also $JEVIMAGE_URL, with $JEVIMAGE_API_KEY)")
    a.add_argument("--timeout", type=float, metavar="SECONDS",
                   help="how long to wait on --url (default 60, also $JEVIMAGE_TIMEOUT)")
    a.add_argument("--local", action="store_true",
                   help="use this machine instead of a remote API: a jevimage serve "
                        "running here, or the weights in this process")
    a.add_argument("--time", action="store_true",
                   help="print where the time went, on stderr")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_ask)

    t = sub.add_parser("train", help="fit a head on a folder of labelled images")
    t.add_argument("name")
    t.add_argument("folder", help="one sub-folder per label")
    t.add_argument("--encoder")
    t.add_argument("--url", help="train on a `jevimage serve` instance instead of locally")
    t.add_argument("--timeout", type=float, metavar="SECONDS",
                   help="how long to wait on --url (default 60, also $JEVIMAGE_TIMEOUT); raise it for a big folder")
    t.add_argument("--local", action="store_true",
                   help="use this machine instead of a remote API: a jevimage serve "
                        "running here, or the weights in this process")
    t.add_argument("--epochs", type=int, default=500)
    t.add_argument("--compare", action="store_true",
                   help="also show trained vs zero-shot prompts, per class")
    t.add_argument("--json", action="store_true")
    t.set_defaults(fn=cmd_train)

    h = sub.add_parser("heads", help="list trained heads")
    h.add_argument("--url", help="list a server's heads instead of the local ones")
    h.add_argument("--local", action="store_true",
                   help="this machine's heads, not a remote API's")
    h.add_argument("--json", action="store_true")
    h.set_defaults(fn=cmd_heads)

    r = sub.add_parser("rm", help="delete a trained head")
    r.add_argument("--url", help="delete on a server instead of locally")
    r.add_argument("name")
    r.set_defaults(fn=cmd_rm)

    sub.add_parser("encoders", help="list the encoder registry").set_defaults(fn=cmd_encoders)

    li = sub.add_parser("login", help="save a key so every shell uses the same server")
    li.add_argument("--key", help="the key; omit to be prompted, or piped on stdin")
    li.add_argument("--url", help=f"which server it is for (default {DEFAULT_URL})")
    li.set_defaults(fn=cmd_login)

    sub.add_parser("logout", help="forget the saved key").set_defaults(fn=cmd_logout)

    d = sub.add_parser("doctor", help="check this machine and show what is missing")
    d.add_argument("--url", help="also check this server")
    d.set_defaults(fn=cmd_doctor)
    return p


def _welcome() -> int:
    """What `jevimage` on its own should say.

    argparse's answer was `error: the following arguments are required: command` and
    exit 2, which reads as a failure to someone who has just installed the thing and
    typed its name to see what it does. Nothing went wrong; they asked a question and
    this is the answer.
    """
    import jevimage

    print(f"jevimage {jevimage.__version__} - typed questions about images.\n")
    print("  jevimage ask photo.jpg --choice indoor=\"an indoor scene\" "
          "outdoor=\"the outdoors\"")
    print("  jevimage train mine dataset/      one sub-folder per label")
    print("  jevimage heads                    the trained heads you can ask for")
    print("  jevimage doctor                   what this machine can do, and what is "
          "missing\n")

    saved = _config()
    url = (env("JEVIMAGE_URL")
           or (DEFAULT_URL if env("JEVIMAGE_API_KEY") else None)
           or (saved.get("url") or DEFAULT_URL if saved.get("key") else None))
    here = None if url else _local_server()
    if url:
        if env("JEVIMAGE_API_KEY"):
            key = " (with $JEVIMAGE_API_KEY)"
        elif saved.get("key"):
            key = " (signed in - jevimage logout to forget)"
        else:
            key = " - but no key is set, which a hosted server will want"
        print(f"Pointed at {url}{key}")
        print("  --local uses a jevimage serve on this machine instead.")
    elif here:
        print(f"Using the jevimage serve already running on {here}")
    else:
        print("Not pointed anywhere yet. Either sign in once, which every shell "
              "then uses:")
        print("  jevimage login                            # a key from "
              "console.jevimage.org")
        print("or run your own server, which needs no key:")
        print("  pip install 'jevimage[serve]' && jevimage serve")
        print("\n`jevimage doctor` checks all of this and names what is missing.")
    print("\nAll commands: jevimage --help      Docs: https://docs.jevimage.org")
    return 0


def main(argv=None) -> int:
    if not (sys.argv[1:] if argv is None else argv):
        _check_env()
        return _welcome()
    args = _parser().parse_args(argv)
    _check_env()
    try:
        return args.fn(args)
    except ImportError as e:
        # The commonest way to land here is an API-only install running a command that
        # would execute the encoder. "No module named 'torch'" does not tell anyone
        # which of the two fixes they want, so say both.
        if "torch" in str(e) or "transformers" in str(e):
            msg = ["jevimage: this command runs the encoder locally, which needs torch:",
                   "       pip install 'jevimage[local]'"]
            if hasattr(args, "url"):        # only the commands that can go remote
                msg += ["     or point it at a server that already has one:",
                        f"       jevimage {args.command} --url http://host:8731"
                        "   (also $JEVIMAGE_URL)"]
            print("\n".join(msg), file=sys.stderr)
            return 1
        print(f"jevimage: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        # Loading an encoder onto a full GPU raises deep inside torch, and the 60-line
        # traceback that reaches the user names cublasCreate rather than the thing they
        # can act on: something else already holds the card. Both fixes are one line.
        msg = str(e)
        if "CUDA" in msg or "CUBLAS" in msg or "out of memory" in msg.lower():
            print(f"jevimage: the GPU could not fit another copy of the model.\n"
                  f"     Something else is already using it - often a `jevimage serve`\n"
                  f"     you started earlier. Either talk to that server instead:\n"
                  f"       jevimage {getattr(args, 'command', 'ask')} --url http://127.0.0.1:8731"
                  f"   (also $JEVIMAGE_URL)\n"
                  f"     or run this one on the CPU:\n"
                  f"       JEVIMAGE_DEVICE=cpu jevimage {getattr(args, 'command', 'ask')} ...\n"
                  f"     (underlying error: {msg.splitlines()[0][:120]})", file=sys.stderr)
            return 1
        raise

    except EXPECTED as e:
        # KeyError stringifies to the repr of its argument, which prints a sentence
        # wrapped in quotes; every other error stringifies to the sentence itself.
        print(f"jevimage: {e.args[0] if isinstance(e, KeyError) and e.args else e}",
              file=sys.stderr)
        # The server's 401 tells an HTTP caller to send an Authorization header, which
        # is the wrong instruction here: from the shell the header is not yours to set.
        # Say which variable is missing instead.
        if getattr(e, "status", None) == 401 and not _api_key():
            print("     No key set in this shell and none saved. Either sign in once,\n"
                  "     which every shell then uses:\n"
                  "       jevimage login\n"
                  "     or export it here (a bare JEVIMAGE_API_KEY=... sets a shell\n"
                  "     variable, not an environment one - it needs `export`):\n"
                  "       export JEVIMAGE_API_KEY=<key>", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def demo():
    """Runnable check - `python -c 'from jevimage.cli import demo; demo()'`, since
    __main__ here is the CLI itself."""
    import contextlib
    import io

    args = _parser().parse_args(
        ["ask", "x.png", "--choice", "a=an a", "b=a b", "--choice", "c=a c", "d=a d",
         "--score", "empty, busy ,packed", "--head", "street", "--noul", "it is dark",
         "--noul", "a hat on a dog", "a dog on a hat"])
    qs = _questions(args)
    assert list(qs) == ["choice", "choice2", "score", "noul", "noul2", "street"], list(qs)
    # the CLI writes a plain noul whatever the captions say; the readout is chosen in
    # `plan`, and the grid one is opt-in because it loses on one image
    assert qs["noul2"] == {"type": "noul", "criteria": {"true": "a hat on a dog",
                                                       "false": "a dog on a hat"}}, qs["noul2"]
    from .core import plan
    os.environ.pop("JEVIMAGE_GRID", None)
    assert plan(qs["noul2"])[0] == "noul_pair", "a rearrangement must read pooled by default"
    os.environ["JEVIMAGE_GRID"] = "1"
    assert plan(qs["noul2"])[0] == "noul_grid", "JEVIMAGE_GRID=1 must enable the readout"
    os.environ.pop("JEVIMAGE_GRID", None)
    assert plan(qs["noul"])[0] == "noul_sigmoid"
    assert qs["choice"]["criteria"] == {"a": "an a", "b": "a b"}, qs["choice"]
    assert qs["choice2"]["criteria"] == {"c": "a c", "d": "a d"}, qs["choice2"]
    assert qs["score"]["criteria"] == ["empty", "busy", "packed"], qs["score"]
    assert qs["street"] == {"type": "head", "head": "street"}, qs["street"]

    for bad, want in ((["--choice", "indoor"], "name=description"),
                      # a repeated name would overwrite the earlier caption, answering
                      # over fewer options than were typed
                      (["--choice", "a=an apple", "a=an aardvark", "b=a bus"], "repeats"),
                      ([], "at least one")):
        try:
            _questions(_parser().parse_args(["ask", "x.png"] + bad))
            raise AssertionError(f"accepted {bad}")
        except ValueError as e:
            assert want in str(e), (want, str(e))

    # the flags beat their variables, including at values that are falsy, and a flag that
    # was passed but is empty is a mistake rather than a request for the default
    os.environ["JEVIMAGE_TIMEOUT"], os.environ["JEVIMAGE_URL"] = "2", "http://unused"
    try:
        for argv, want in (
                (["ask", "x.png", "--url", "http://h", "--timeout", "0", "--choice",
                  "a=x", "b=y"], "--timeout must be a positive"),
                (["ask", "x.png", "--url", "", "--choice", "a=x", "b=y"],
                 "--url was given but is empty"),
                (["rm", "colours", "--url", ""], "--url was given but is empty"),
                (["serve", "--api-key", ""], "OPEN server"),
                (["serve", "--api-key", "correct horse"], "cannot contain a space"),
                (["serve", "--api-key", "a,b"], "cannot contain a space"),
                (["serve", "--heads-dir", ""], "--heads-dir was given but is empty")):
            try:
                args = _parser().parse_args(argv)
                (_backend if argv[0] != "serve" else cmd_serve)(args)
                raise AssertionError(f"accepted {argv}")
            except ValueError as e:
                assert want in str(e), (argv, want, str(e))
        os.environ["JEVIMAGE_TIMEOUT"] = "abc"
        try:
            _backend(_parser().parse_args(["heads", "--url", "http://h"]))
            raise AssertionError("accepted JEVIMAGE_TIMEOUT=abc")
        except ValueError as e:
            assert "$JEVIMAGE_TIMEOUT" in str(e), str(e)
    finally:
        del os.environ["JEVIMAGE_TIMEOUT"], os.environ["JEVIMAGE_URL"]

    # repeating --api-key must add to the keys, not replace them: that is the rotation
    # the flag exists for. And a port the kernel would truncate is refused, not renamed.
    assert _parser().parse_args(["serve", "--api-key", "k1", "--api-key", "k2"]).api_key \
        == [["k1"], ["k2"]]
    for bad in ("99999", "-1", "65536"):
        try:
            with contextlib.redirect_stderr(io.StringIO()):      # argparse's own usage
                _parser().parse_args(["serve", "--port", bad])
            raise AssertionError(f"accepted port {bad}")
        except SystemExit:
            pass

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _table([["red", 6, "0.500"]], ["label", "n", "acc"])
        _show("scene", {"type": "choice", "choice": "outdoor", "confidence": 0.86,
                        "probabilities": {"indoor": 0.07, "outdoor": 0.93}})
    lines = out.getvalue().splitlines()
    assert lines[:2] == ["label  n    acc", "red    6  0.500"], lines[:2]
    assert lines[3].split() == ["outdoor", "0.930", "#" * 20], lines[3]

    # doctor: a missing torch is an API-only install, not a fault - it only fails when
    # there is no server to call either
    assert _verdict(True, True, False)[1] == 0
    assert _verdict(True, False, True)[1] == 0, "API-only with a server is fine"
    assert _verdict(True, False, False)[1] == 1, "nothing can answer a question"
    assert _verdict(False, True, True)[1] == 1, "no pillow beats everything else"

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = main(["ask", "x.png", "--choice", "a=x", "b=y", "--encoder", "nope"])
    assert code == 1, code
    assert "unknown encoder" in err.getvalue() and "Traceback" not in err.getvalue(), err.getvalue()

    print("cli selftest ok - repeated flags get distinct ids, bad pairs, repeated choice "
          "names and empty asks are refused, a flag beats its variable at every value "
          "including 0, an empty --url/--api-key/--port out of range cannot start an "
          "open server or retarget a delete, columns align, doctor calls an API-only "
          "install ready and an empty one not, an unknown encoder exits 1 "
          "without a traceback")


if __name__ == "__main__":
    sys.exit(main())
