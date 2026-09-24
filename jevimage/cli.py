"""The `jev` command: try an encoder, train a head, see whether training helped.

A thin shell over the Python API - nothing is computed here that `jevimage.Jev` does
not already compute, so a number that looks wrong is wrong upstream, not in the
formatting. Plain ASCII on purpose: this output gets pasted into issues and logs.

    jev encoders
    jev ask photo.jpg --choice indoor="an indoor scene" outdoor="the outdoors"
    jev train street dataset/
    jev heads
    jev serve
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from .client import JevError

# The errors a user causes: a typo'd encoder name, a head that was never trained, a path
# that is not there. A traceback for those tells them nothing they can act on.
EXPECTED = (ValueError, KeyError, OSError, JevError)


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
        print(f"{qid}  [noul]  {a['noul']:.3f}   ({a['basis']})\n")
        return

    if a["type"] == "score":
        head = f"{a['score']:.2f} / {len(a['legend']) - 1}"
    else:
        head = f"{a['choice']}"
    head += f"   confidence {a['confidence']:.3f}"
    if a["type"] == "head":
        head += f"   (head {a['head']!r}, {a['trained_on']} examples, " \
                f"held out {_pct(a['accuracy']).strip()})"
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

def _opt(args, name, env):
    """A flag's value, else the environment, else None.

    `is None`, not truthiness: `--timeout 0` is a value the user typed, and the flags are
    documented to beat their variables. A flag that was passed but is empty - the shape
    `--url "$SERVER"` takes when $SERVER is unset - is a mistake, not a request for the
    default, and on `rm` that mistake silently retargets a delete at the local directory.
    """
    v = getattr(args, name, None)
    if v is None:
        return os.environ.get(env) or None
    if isinstance(v, str) and not v.strip():
        raise ValueError(f"--{name.replace('_', '-')} was given but is empty")
    return v


def _remote(args):
    """The server to run against, or None for local. One resolution for every command
    that branches on it, so `--url` means the same thing to `rm` as to `ask`."""
    return _opt(args, "url", "JEV_URL")


def _backend(args):
    """A remote instance when --url (or $JEV_URL) is set, otherwise a local one.

    The two expose the same methods, so every command below is written once and works
    either way - which is the whole point of `connect` mirroring `load`.
    """
    import jevimage

    url = _remote(args)
    if url:
        # training a few thousand images through a server routinely outlives the 60s
        # client default, and a job that cannot be waited for cannot be run
        raw = _opt(args, "timeout", "JEV_TIMEOUT")
        raw = 60.0 if raw is None else raw
        where = "--timeout" if getattr(args, "timeout", None) is not None else "$JEV_TIMEOUT"
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{where}={raw!r} is not a number of seconds") from None
        if timeout <= 0:
            raise ValueError(f"{where} must be a positive number of seconds; got {raw}")
        return jevimage.connect(url, api_key=os.environ.get("JEV_API_KEY"),
                                timeout=timeout)
    name = _opt(args, "encoder", "JEV_ENCODER")
    if name:
        # Checked against the registry first, which reads without torch. Otherwise a
        # typo on an API-only install answered "install torch": you would install it,
        # run the same command, and only then be told the name was wrong.
        from . import encoders

        for part in (p.strip() for p in str(name).split("+")):
            if part and part not in encoders.REGISTRY:
                raise KeyError(f"unknown encoder {part!r}. Available: "
                               f"{', '.join(encoders.available())}")
    return jevimage.load(name)


def _where(jev) -> str:
    return (f"server {jev.url}" if hasattr(jev, "url")
            else f"encoder {jev.encoder.name}")


# -- commands ------------------------------------------------------------------

def cmd_ask(args) -> int:
    questions = _questions(args)            # validated before any weights load
    jev = _backend(args)
    answers = jev.ask(args.image, questions)
    if args.json:
        print(json.dumps(answers, indent=2))
    else:
        print(f"{args.image}   {_where(jev)}\n")
        for qid, a in answers.items():
            _show(qid, a)
    return 0


def cmd_train(args) -> int:
    jev = _backend(args)
    head = jev.train(args.name, args.folder, epochs=args.epochs)
    print(f"{head.name!r}: {head.n} examples, {len(head.classes)} classes, "
          f"encoder {head.encoder}\n")

    cmp = head.comparison
    if not cmp:
        # a head restored without its embeddings has nothing to compare against; the
        # head itself is still fine, so this is a missing section, not a failure
        print("(no zero-shot comparison for this head - it was saved without the "
              "embeddings the comparison is computed from)")
        return 0
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


def cmd_heads(args) -> int:
    """Reads the head directory directly: listing what you trained should not download
    an encoder's weights."""
    broken = {}
    if _remote(args):
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
        print(f"no heads in {where}. Train one: jev train NAME FOLDER")
        return 0
    _table([[h["name"], h["encoder"], len(h["classes"]), h["n"], _pct(h["cv_accuracy"]),
             time.strftime("%Y-%m-%d", time.localtime(h["created"])) if h["created"] else "-"]
            for h in heads],
           ["head", "encoder", "classes", "n", "held out", "created"])
    print(f"\n{where}")
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

    override = os.environ.get("JEV_ENCODER")
    for name in encoders.available():
        mark = "*" if name == (override or encoders.DEFAULT) else " "
        print(f" {mark} {name}")
    note = f"$JEV_ENCODER={override}" if override else f"default ({encoders.DEFAULT})"
    print(f"\n* {note}. Join names with '+' for an ensemble, e.g. "
          f"{encoders.DEFAULT}+dfn5b-h-14-384.")
    return 0


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
            # $JEV_API_KEYS is split on commas AND whitespace, so a key containing either
            # authenticates as several shorter fragments while the real key gets a 401.
            if not k.strip():
                raise ValueError("--api-key was given an empty key, which would start an "
                                 "OPEN server; pass a real key, or omit the flag to serve "
                                 "openly on purpose")
            if re.search(r"[\s,]", k):
                raise ValueError(f"--api-key {k!r}: a key cannot contain a space or a "
                                 "comma, because $JEV_API_KEYS is split on both")
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
        raise ImportError("jev serve needs uvicorn: pip install 'jevimage[serve]'")

    # Passed through the environment rather than to the app object, so the same knobs
    # work when a container or process manager starts uvicorn itself.
    for var, value in (("JEV_ENCODER", args.encoder), ("JEV_HEADS_DIR", args.heads_dir),
                       ("JEV_API_KEYS", ",".join(keys))):
        if value:
            os.environ[var] = value
    from .server import app     # touching it builds the app, so the encoder loads here

    uvicorn.run(app, host=args.host, port=args.port)
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
    p = argparse.ArgumentParser(prog="jev", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="serve the HTTP API (needs uvicorn)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=_port, default=8000)
    s.add_argument("--encoder")
    s.add_argument("--heads-dir")
    s.add_argument("--api-key", nargs="+", action="append", metavar="KEY",
                   help="require one of these keys; without it the API is open")
    s.set_defaults(fn=cmd_serve)

    a = sub.add_parser("ask", help="ask one image typed questions")
    a.add_argument("image")
    a.add_argument("--choice", nargs="+", action="append", metavar="NAME=DESCRIPTION",
                   help='pick one, e.g. --choice indoor="an indoor scene" outdoor="outside"')
    a.add_argument("--score", action="append", metavar="LOW,...,HIGH",
                   help='ordered rubric, e.g. --score "empty,busy,packed"')
    a.add_argument("--noul", nargs="+", action="append", metavar="TEXT",
                   help='yes/no probability: two captions for a true/false pair (works '
                        'on every encoder), or one for the sigmoid readout (needs a '
                        'calibrated encoder)')
    a.add_argument("--head", action="append", metavar="NAME", help="run a trained head")
    a.add_argument("--encoder")
    a.add_argument("--url", help="run against a `jev serve` instance instead of local "
                                 "weights (also $JEV_URL, with $JEV_API_KEY)")
    a.add_argument("--timeout", type=float, metavar="SECONDS",
                   help="how long to wait on --url (default 60, also $JEV_TIMEOUT)")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_ask)

    t = sub.add_parser("train", help="fit a head on a folder of labelled images")
    t.add_argument("name")
    t.add_argument("folder", help="one sub-folder per label")
    t.add_argument("--encoder")
    t.add_argument("--url", help="train on a `jev serve` instance instead of locally")
    t.add_argument("--timeout", type=float, metavar="SECONDS",
                   help="how long to wait on --url (default 60, also $JEV_TIMEOUT); raise it for a big folder")
    t.add_argument("--epochs", type=int, default=500)
    t.set_defaults(fn=cmd_train)

    h = sub.add_parser("heads", help="list trained heads")
    h.add_argument("--url", help="list a server's heads instead of the local ones")
    h.add_argument("--json", action="store_true")
    h.set_defaults(fn=cmd_heads)

    r = sub.add_parser("rm", help="delete a trained head")
    r.add_argument("--url", help="delete on a server instead of locally")
    r.add_argument("name")
    r.set_defaults(fn=cmd_rm)

    sub.add_parser("encoders", help="list the encoder registry").set_defaults(fn=cmd_encoders)
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.fn(args)
    except ImportError as e:
        # The commonest way to land here is an API-only install running a command that
        # would execute the encoder. "No module named 'torch'" does not tell anyone
        # which of the two fixes they want, so say both.
        if "torch" in str(e) or "transformers" in str(e):
            msg = ["jev: this command runs the encoder locally, which needs torch:",
                   "       pip install 'jevimage[local]'"]
            if hasattr(args, "url"):        # only the commands that can go remote
                msg += ["     or point it at a server that already has one:",
                        f"       jev {args.command} --url http://host:8000   (also $JEV_URL)"]
            print("\n".join(msg), file=sys.stderr)
            return 1
        print(f"jev: {e}", file=sys.stderr)
        return 1
    except EXPECTED as e:
        # KeyError stringifies to the repr of its argument, which prints a sentence
        # wrapped in quotes; every other error stringifies to the sentence itself.
        print(f"jev: {e.args[0] if isinstance(e, KeyError) and e.args else e}",
              file=sys.stderr)
        # The server's 401 tells an HTTP caller to send an Authorization header, which
        # is the wrong instruction here: from the shell the header is not yours to set.
        # Say which variable is missing instead.
        if getattr(e, "status", None) == 401 and not os.environ.get("JEV_API_KEY"):
            print("     $JEV_API_KEY is not set in this shell. Note that a bare\n"
                  "     `JEV_API_KEY=...` line sets a shell variable, not an\n"
                  "     environment one - it needs `export`:\n"
                  "       export JEV_API_KEY=<key>", file=sys.stderr)
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
         "--score", "empty, busy ,packed", "--head", "street", "--noul", "it is dark"])
    qs = _questions(args)
    assert list(qs) == ["choice", "choice2", "score", "noul", "street"], list(qs)
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
    os.environ["JEV_TIMEOUT"], os.environ["JEV_URL"] = "2", "http://unused"
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
        os.environ["JEV_TIMEOUT"] = "abc"
        try:
            _backend(_parser().parse_args(["heads", "--url", "http://h"]))
            raise AssertionError("accepted JEV_TIMEOUT=abc")
        except ValueError as e:
            assert "$JEV_TIMEOUT" in str(e), str(e)
    finally:
        del os.environ["JEV_TIMEOUT"], os.environ["JEV_URL"]

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

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = main(["ask", "x.png", "--choice", "a=x", "b=y", "--encoder", "nope"])
    assert code == 1, code
    assert "unknown encoder" in err.getvalue() and "Traceback" not in err.getvalue(), err.getvalue()

    print("cli selftest ok - repeated flags get distinct ids, bad pairs, repeated choice "
          "names and empty asks are refused, a flag beats its variable at every value "
          "including 0, an empty --url/--api-key/--port out of range cannot start an "
          "open server or retarget a delete, columns align, an unknown encoder exits 1 "
          "without a traceback")


if __name__ == "__main__":
    sys.exit(main())
