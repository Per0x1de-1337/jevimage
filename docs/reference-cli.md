---
title: "CLI"
description: "Every jevimage command and flag, exit codes, environment variables and remote mode."
---

`jevimage` is a thin shell over the Python API. It computes nothing that
`jevimage.Jev` and `jevimage.connect()` do not already compute, so a number that
looks wrong here is wrong upstream, not in the formatting. The output is plain
ASCII on purpose; it is meant to survive being pasted into an issue or a log.

```
jevimage serve | ask | train | heads | rm | encoders
```

Four of those (`ask`, `train`, `heads`, `rm`) take `--url` and drive a remote
`jevimage serve` instance instead of local weights. Remote mode runs the same code
path, because `jevimage.connect(url)` exposes the same methods as
`jevimage.load()` and the command body is written once against both.

Related pages: [Install](install.md) · [Quickstart](quickstart.md) ·
[Question types](questions.md) · [Encoders](encoders.md) ·
[Serving](serving.md) · [Python API](reference-python.md) ·
[HTTP API](reference-http.md)

## About the transcripts on this page

Every block below is a command that was run, with its real output. The one
substitution: the encoder is a **deterministic toy encoder** (a 16-dim
hash-of-the-bytes model registered as `toy` / `toy-calibrated`), not SigLIP,
because this machine has no network for model weights. The images are `ds/`, the twelve
squares [`examples/make_fixtures.py`](https://github.com/Per0x1de-1337/jevimage/blob/main/examples/make_fixtures.py) writes. The toy is not
semantic (it will happily call a blue square "red"), so read the *shapes* of these
outputs, not the probabilities. Everything that is not a probability (columns, ids, error
text, exit codes, which install a command needs) is exactly what you will see.

Two cosmetic edits: long scratch-directory paths are elided to `/tmp/.../` so the
lines fit, and a repeated listing already printed in full above is cut short with
a `...` line. Nothing in a transcript is retyped.

The toy appears in `jevimage encoders` listings on this page because it was registered by a
`sitecustomize.py` on `PYTHONPATH`. That is also the only way to get a custom encoder
into the CLI at all. See [Encoders and the CLI](#custom-encoders-and-the-cli).

## What each command needs installed

| Command | `pip install jevimage` (no torch) | Needs `[local]` | Needs `[serve]` |
|---|---|---|---|
| `jevimage encoders` | works | no | no |
| `jevimage doctor` | works | no | no |
| `jevimage ask --url` / `train --url` / `heads --url` | works | no | no |
| `jevimage rm` (local) | works | no | no |
| `jevimage rm --url` | works | no | no |
| `jevimage ask` / `jevimage train` (local) | fails, exit 1 * | yes | no |
| `jevimage heads` (local) | fails, exit 1 * | yes | no |
| `jevimage serve` | fails, exit 1 | yes | yes |

\* unless a `jevimage serve` is already running on this machine, in which case the
question goes there and no torch is needed - see below.

`jevimage doctor` (below) is the same idea and answers the question this table asks:
it reports which of those three installs this machine actually has.

`jevimage encoders` listing the registry without torch is deliberate: seeing which
encoders exist should not require the stack that runs them. `jevimage rm` gets there
by accident of being an `unlink`; it never has to read a head file.

The failure is one message, not a traceback, and it names both fixes:

```console
$ jevimage ask ds/red/0.png --choice red="a red square" blue="a blue square"
jevimage: this command runs the encoder locally, which needs torch:
       pip install 'jevimage[local]'
     or point it at a server that already has one:
       jevimage ask --url http://host:8731   (also $JEVIMAGE_URL)
exit=1
```

The second half only appears on commands that can go remote. `jevimage serve` cannot,
so it gets its own message:

```console
$ jevimage serve
jevimage: jevimage serve needs uvicorn: pip install 'jevimage[serve]'
exit=1
```

`jevimage heads` and `jevimage train` produce the same two-part message as `jevimage ask`, with
the command name substituted.

> Listing *local* heads needs torch even though a head listing is only metadata:
> the head files are torch tensors, and the command imports the training module
> to format them. Listing a *server's* heads does not, which is the whole reason
> that import sits inside the function rather than at the top of the module.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. Also: `jevimage` with no arguments at all, `jevimage heads` with no heads, and Ctrl-C during `jevimage serve`. |
| `1` | An error you caused: bad flag value, unknown encoder or head, missing file, HTTP error from a server, a missing dependency. One line on stderr, no traceback. |
| `2` | argparse rejected the command line: an unknown subcommand, or a missing positional on one that exists. Usage is printed. Typing `jevimage` alone is not this - see [above](#exit-codes). |
| `130` | Ctrl-C during `ask`, `train`, `heads`, `rm`. Silent; nothing is printed. |

Exit 1 covers `ValueError`, `KeyError`, `OSError`, `ImportError` and `JevError`
(the client's HTTP error). Anything else is a bug and prints a traceback.

```console
$ jevimage frobnicate
usage: jevimage [-h] {serve,ask,train,heads,rm,encoders,doctor} ...
jevimage: error: argument command: invalid choice: 'frobnicate' (choose from 'serve', 'ask', 'train', 'heads', 'rm', 'encoders', 'doctor')
exit=2

$ jevimage
jevimage 0.1.19 - typed questions about images.

  jevimage ask photo.jpg --choice indoor="an indoor scene" outdoor="the outdoors"
  jevimage train mine dataset/      one sub-folder per label
  jevimage heads                    the trained heads you can ask for
  jevimage doctor                   what this machine can do, and what is missing

Not pointed anywhere yet. Either:
  export JEVIMAGE_URL=https://api.jevimage.org   # the hosted API
  export JEVIMAGE_API_KEY=...                    # a key from console.jevimage.org
or run your own, which needs no key:
  pip install 'jevimage[serve]' && jevimage serve

`jevimage doctor` checks all of this and names what is missing.

All commands: jevimage --help      Docs: https://docs.jevimage.org
exit=0
```

The bare command exits **0**, not 2. Nothing went wrong: someone installed it and typed
its name. A command that does not exist is still an error, and still exits 2, as above.

The middle section changes with your setup - it names the server `$JEVIMAGE_URL` points at,
or one it found running locally, or neither:

```console
$ JEVIMAGE_URL=https://api.jevimage.org jevimage
...
Pointed at https://api.jevimage.org - but $JEVIMAGE_API_KEY is not set, which a hosted server will want
```

Ctrl-C, here against a server that accepts the connection and never answers:

```console
$ jevimage heads --url http://127.0.0.1:8134      # then Ctrl-C
exit=130
```

`jevimage serve` is the exception: uvicorn installs its own SIGINT handler, shuts the
app down and returns normally, so Ctrl-C there exits `0`.

## Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `JEVIMAGE_URL` | `ask`, `train`, `heads`, `rm` | Same as `--url`. An explicit `--url` wins. |
| `JEVIMAGE_TIMEOUT` | `ask`, `train`, `heads`, `rm` | Seconds to wait on a remote instance. Default 60. `--timeout` (on `ask` and `train`) wins, including when it is `0`. Both must be a positive number; `0` or less is exit 1 either way. |
| `JEVIMAGE_ENCODER` | `ask`, `train`, `serve`, `encoders` | Default encoder name. `--encoder` wins. `jevimage encoders` marks it with `*`. |
| `JEVIMAGE_HEADS_DIR` | every local command | Where heads live. `jevimage serve --heads-dir` sets it for the server process. |
| `JEVIMAGE_HOME` | every local command | Heads go to `$JEVIMAGE_HOME/heads` when `JEVIMAGE_HEADS_DIR` is unset. Default `~/.jev`. |
| `JEVIMAGE_DEVICE` | local commands | `cuda`, `cpu`, `cuda:1`. Default: CUDA if available, else CPU. |
| `JEVIMAGE_API_KEY` | `ask`, `train`, `heads`, `rm` | Bearer token for a remote instance. **There is no flag for this**; the environment is the only way to pass it. Setting it with no `--url` also selects the hosted API - a key is only good against the service that issued it. |
| `JEVIMAGE_THREADS` | local commands, on the CPU | How many threads a forward pass uses. Default: the performance cores on a hybrid chip, torch's own default anywhere else. `0` leaves torch alone. |
| `JEVIMAGE_API_KEYS` | the server process | Keys the server accepts, comma- or space-separated. `jevimage serve --api-key` sets it. |
| `JEVIMAGE_BATCH` | local commands | Images per encoder pass. Halves itself on an out-of-memory, so it is a starting point rather than a limit. Default 32. |

That table is the complete set. `jevimage doctor` shows which of them are set on this
machine, and never prints a key's value. They are deliberately **not** in
`jevimage --help`: that answers "what can this do", and eleven variables most people
never touch made it twice as long as the commands it documents.

Any other `JEVIMAGE_`-prefixed variable is read by nobody, and the CLI says so in one line
rather than letting it look like it took effect:

```console
$ export JEVIMAGE_BASE=https://localhost:8732
$ jevimage ask photo.jpg --choice a="an a" b="a b"
jevimage: $JEVIMAGE_BASE is set, but nothing reads it. `jevimage doctor` shows the ones that are.
```

### Against the hosted API

`--url` and `$JEVIMAGE_URL` take `https://api.jevimage.org` as readily as your own server;
it needs a key, which goes in `$JEVIMAGE_API_KEY`:

```console
$ export JEVIMAGE_URL=https://api.jevimage.org
$ export JEVIMAGE_API_KEY=<key>
$ jevimage ask photo.jpg --choice indoor="an indoor scene" outdoor="an outdoor scene"
photo.jpg   server https://api.jevimage.org

choice  [choice]  outdoor   confidence 0.480
    outdoor   0.740  ################
    indoor    0.260  ######
```

Two differences it papers over, so you do not have to know which you reached: the hosted
API answers about a list of images and puts each image's answers under `results[i]`
rather than at the top level, and it keeps `/v1/health` for liveness while publishing
its capability document behind `/v1/models`.

`train`, `heads` and `rm` all work against it. A head belongs to the key that trained
it - listed and runnable only by that key - while the shared demo heads belong to
nobody, so everyone can use them and no one can delete them.

Because of that, **two keys list two different sets of heads**, and `heads` names the
account so the difference is visible rather than looking like a sync failure:

```console
$ jevimage heads
head                encoder  classes    n  held out  created
street_objects  jev-image-1        4  468     95.9%        -

https://api.jevimage.org   (account Hemanshu-codeant)
```

A head you trained in the console and cannot see here is almost always a key belonging
to a different GitHub account than the one the browser is signed in as. `jevimage
doctor` names it too.

```console
$ jevimage train street4 ds/ --url https://api.jevimage.org
trained 'street4': 428 examples, 4 classes, 96.3% accurate
  classes: bicycle, bus, hydrant, motorcycle
  encoder: jev-image-1

Use it:
  jevimage ask IMAGE --head street4
```

Training images are capped per account rather than per request - 1000 on
api.jevimage.org, and `/v1/models` publishes the figure. Retraining a head does not
count its old copy, so a head can always be retrained. When the cap is reached the
refusal says how much room is left:

```console
$ jevimage train another ds/ --url https://api.jevimage.org
jevimage: training images per account are capped at 1000; your other heads already use
859, so this one can add at most 141 (you sent 428). Delete a head, or train on fewer
images per class. [HTTP 413 from POST https://api.jevimage.org/v1/heads]
```

**Where the time goes.** Almost none of it is the server, which answers in about 10ms
and says so in `timings.total_ms`. The rest is the network, and what dominates is the
handshake rather than the bytes: three round trips are mandatory before a fresh HTTPS
connection can carry anything.

api.jevimage.org is served from a CDN, so those round trips are to a nearby edge rather
than to the machine running the model. Measured from India, one `jevimage ask` end to
end:

| | |
|---|---|
| warm | about 350 ms |
| first call after an idle spell | about 850 ms - the edge reconnects to the origin |
| server time within either | 10 ms |

The first-call figure is a low-traffic artefact: the edge keeps its connection to the
origin for two minutes, so under steady use it never lapses. What removes the handshake
altogether is opening fewer connections - pass several images to one `ask`, or from
Python keep a `connect()` object, which measures about 225 ms per call once warm.

Against your own `jevimage serve` on the same machine none of this applies: the whole
call is about 80 ms, most of it Python starting up.

`jevimage serve` speaks plain HTTP and does not terminate TLS, so an `https://` URL
pointed at one says which half is wrong:

A genuine typo of a real name is suggested; a name that merely looks similar is not,
because a confident wrong guess costs more than no guess:

```console
$ JEVIMAGE_TIMOUT=30 jevimage heads
jevimage: $JEVIMAGE_TIMOUT is set, but nothing reads it. Did you mean $JEVIMAGE_TIMEOUT?
```

```console
$ jevimage ask photo.jpg --url https://localhost:8731 --choice a="an a" b="a b"
jevimage: cannot reach a jevimage server at https://localhost:8731 ([SSL: WRONG_VERSION_NUMBER] wrong version number (_ssl.c:1000)). That server is speaking plain HTTP: use http://, or put a proxy in front of it to terminate TLS.
exit=1
```

`JEVIMAGE_HOME` at work. Note the `/heads` suffix the CLI appends:

```console
$ JEVIMAGE_HOME=/tmp/dochome2 jevimage heads
no heads in /tmp/dochome2/heads. Train one: jevimage train NAME FOLDER
exit=0
```

## Remote mode: `--url` and `$JEVIMAGE_URL`

Set either and `ask`, `train`, `heads` and `rm` run against a `jevimage serve`
instance: no torch, no weights, no GPU on this machine. `--url` beats `$JEVIMAGE_URL`.
The last line of `jevimage ask` output tells you which backend answered, so a
transcript is self-describing:

```console
$ jevimage ask ds/red/0.png --url http://127.0.0.1:8123 \
    --choice red="a red square" blue="a blue square" \
    --score "empty,half full,packed" \
    --noul "a red square" "a blue square"
ds/red/0.png   server http://127.0.0.1:8123

choice  [choice]  red   confidence 1.000
    red    1.000  ######################
    blue   0.000  

score  [score]  1.28 / 2   confidence 0.562
    half full   0.708  ################
    packed      0.285  ######
    empty       0.007  

noul  [noul]  1.000   (contrast pair)

exit=0
```

Against a local encoder the header reads `encoder toy` instead, and for this
image the numbers are identical: same engine, same arithmetic, different side of
a socket.

**`--encoder` is silently ignored when a URL is set.** The server decides which
encoder runs; nothing is downloaded here. This command answers from the remote
toy encoder and never touches `siglip2-giant-384`:

```console
$ jevimage ask ds/red/0.png --url http://127.0.0.1:8123 --encoder siglip2-giant-384 \
    --choice red="a red square" blue="a blue square"
ds/red/0.png   server http://127.0.0.1:8123

choice  [choice]  red   confidence 1.000
    red    1.000  ######################
    blue   0.000  

exit=0
```

`jevimage encoders` has no `--url`: it lists the registry compiled into *this* install,
which is not necessarily what the server runs. Ask the server itself:
`curl $JEVIMAGE_URL/v1/health` returns its encoder name and dim.

### Authentication

```console
$ jevimage heads --url http://127.0.0.1:8131
jevimage: send 'Authorization: Bearer <key>' with a key this server was started with (JEVIMAGE_API_KEYS) [HTTP 401 from GET http://127.0.0.1:8131/v1/heads]
     $JEVIMAGE_API_KEY is not set in this shell. Note that a bare
     `JEVIMAGE_API_KEY=...` line sets a shell variable, not an
     environment one - it needs `export`:
       export JEVIMAGE_API_KEY=<key>
exit=1

$ JEVIMAGE_API_KEY=s3cret jevimage heads --url http://127.0.0.1:8131
no heads in http://127.0.0.1:8131. Train one: jevimage train NAME FOLDER
exit=0
```

A wrong key gets the identical 401 *from the server* - it does not distinguish "no key"
from "wrong key", and should not. The printed output differs: the four `export` lines
above are the client's own hint, added only when `$JEVIMAGE_API_KEY` is unset, so a
`JEVIMAGE_API_KEY=wrong jevimage heads --url ...` prints the first line and nothing else.

### When the server is not there

```console
$ jevimage heads --url http://127.0.0.1:9999
jevimage: cannot reach a jevimage server at http://127.0.0.1:9999 ([Errno 111] Connection refused). Start one with `jevimage serve`, or pass the URL it is actually listening on.
exit=1

$ jevimage heads --url 127.0.0.1:8131
jevimage: url '127.0.0.1:8131' has no scheme; use 'http://127.0.0.1:8131'
exit=1
```

The scheme check is separate because urllib's own message ("unknown url type:
127.0.0.1") points at the host and sends you looking in the wrong place.

The client's default timeout is 60s. `ask` and `train` take `--timeout SECONDS`
for a cold server that is still loading a model, or a training job too big to
finish inside a minute; `$JEVIMAGE_TIMEOUT` sets it for all four remote commands,
including the two with no flag:

```console
$ jevimage ask ds/red/0.png --url http://127.0.0.1:8134 --timeout 2 \
    --choice red="a red square" blue="a blue square"
jevimage: POST http://127.0.0.1:8134/v1/ask did not answer within 2.0s; raise timeout= if the server is loading a model
exit=1

$ JEVIMAGE_TIMEOUT=2 jevimage heads --url http://127.0.0.1:8134
jevimage: GET http://127.0.0.1:8134/v1/heads did not answer within 2.0s; raise timeout= if the server is loading a model
exit=1
```

That is port 8134 again: a socket that accepts the connection and then says
nothing, which is what a timeout has to be provoked with.

---

## Where a command sends the work

`ask`, `train` and `heads` pick a backend in this order. The first line that applies
wins:

| | |
|---|---|
| `--local` | this **machine**: a `jevimage serve` running here, or the weights in this process |
| `--url URL` / `$JEVIMAGE_URL` | that server |
| `--encoder NAME` / `$JEVIMAGE_ENCODER` | this process - you named an encoder, so you meant one |
| **`$JEVIMAGE_API_KEY` is set** | **the hosted API** at `https://api.jevimage.org` |
| **`jevimage login` was run** | **the server you signed in to** |
| a `jevimage serve` running on `127.0.0.1:8731-8740` | that server |
| nothing above | the weights in this process |

**A key selects the hosted API** because a key is only good against the service that
issued it: holding one is the same as choosing it. It is not a hardcoded default,
because without a key that default is a 401, and this package is self-hostable - a
working local install must not be redirected to an endpoint it cannot authenticate to.
`$JEVIMAGE_URL` reaches a self-hosted or staging server, and beats the default.

**`--local` is about the machine, not the process.** Someone with a server running here
who asks for local means that server, not twelve seconds of reloading the same weights
beside it. `--encoder NAME` is how you insist on this process, and it reads as what it
does.

### The local probe

When no key and no URL are set, `ask`, `train` and `heads` check `127.0.0.1:8731-8740`
for a running server before loading their own copy of the encoder, and use the first
that answers. Loading SigLIP2 costs about twelve seconds and the question itself costs
twenty milliseconds, so a CLI that loads its own copy spends 99.8% of its life starting
up - and on a small GPU it collides with the server it ignored.

```console
$ time jevimage ask photo.jpg --choice indoor="an indoor scene" outdoor="the outdoors"
using http://127.0.0.1:8731   (a jevimage serve is running here; --encoder NAME to load the weights in this process)
photo.jpg   server http://127.0.0.1:8731

choice  [choice]  outdoor   confidence 0.722

real    0m0.07s
```

The notice goes to stderr, so `--json` piped to a file is unaffected.

`heads` follows the same rule so that it lists the heads `ask` will actually answer
from. `rm` does not: deleting is not a read, and silently deleting the server's head
when you meant the one in `~/.jev/heads` is not a speedup. `rm --url` still works.

The probe is a `GET /v1/health` against loopback with a quarter-second timeout; any
failure at all falls through to loading locally. It sweeps the whole range rather than
one port because [`serve` walks up](#jevimage-serve) when its own is taken, and a server
we cannot find is a server we start a second copy of. A closed port on loopback refuses
immediately instead of waiting out the timeout, so ten of them cost about a millisecond
in total.

The range is not configurable: it is `jevimage serve`'s own, and a server outside it is
one you put there, so name it with `--url`.

`jevimage doctor` ends with what is configured here, so a variable that is set is
visible without anyone reciting the whole table:

```console
$ jevimage doctor
...
  yes $JEVIMAGE_URL               http://127.0.0.1:8731
  yes $JEVIMAGE_API_KEY           set
  yes $JEVIMAGE_DEVICE            cpu
```

`$JEVIMAGE_API_KEY` reads `set` rather than its value. Configuration output that echoes a
secret is one screenshot away from leaking it.

## `jevimage login`

```
jevimage login [--key KEY] [--url URL]
jevimage logout
```

Saves a key so every shell uses the same server. Without it, a terminal that has not
exported `$JEVIMAGE_API_KEY` is indistinguishable from someone with no account, so the same
command in two windows reached two different servers and listed two different sets of
heads - the console showing one thing and `jevimage heads` another. A browser does not
ask you to re-authenticate per tab.

```console
$ jevimage login
API key (not echoed):
signed in to https://api.jevimage.org
  model jev-image-1, 2 head(s) you can use
  saved to /home/you/.jev/config.json (0600). `jevimage logout` removes it.
```

The key is checked against the server before anything is written, so a typo fails here
rather than on your next command. The file is created `0600` **before** the key goes
into it - a secret must not exist, even for an instant, at whatever the umask happens to
be. `--key` takes it from the command line, and it is read from stdin when piped, for
CI:

```console
$ echo "$JEVIMAGE_KEY" | jevimage login --url https://api.jevimage.org
```

`$JEVIMAGE_API_KEY` and `$JEVIMAGE_URL` still win when set, so one shell can point somewhere else
without signing out. `jevimage logout` deletes the file; `jevimage` on its own says
whether you are signed in.

## `jevimage doctor`

What this machine can do, and the command that fixes anything it cannot. There are two
supported installs - calling a server needs no torch at all, running models locally
needs the whole stack - so `doctor` checks both and only complains when neither is
there.

```console
$ jevimage doctor
jevimage 0.1.19 on python 3.12.3 (Linux x86_64)

  yes python >= 3.10         3.12.3
  yes pillow                 present
  yes server                 http://127.0.0.1:8731 - siglip2-base-224 on cuda, 2 head(s), auth open
  yes run models locally     torch + transformers present
  yes gpu                    NVIDIA GeForce RTX 3050 Laptop GPU, 1.9 GB free of 3.7 GB
  yes weights cached         11 model(s) in /home/you/.cache/huggingface/hub
  yes disk                   31.0 GB free in your home directory
  yes host a server          fastapi + uvicorn present

  ready: this machine can answer questions on its own and a server is reachable
exit=0
```

An API-only install is a pass, not a failure:

```console
$ jevimage doctor
jevimage 0.1.19 on python 3.12.3 (Linux x86_64)

  yes python >= 3.10         3.12.3
  yes pillow                 present
  yes server                 http://127.0.0.1:8731 - siglip2-base-224 on cuda, 2 head(s), auth open
  --  run models locally     torch no, transformers no
                             -> pip install 'jevimage[local]'
  --  host a server          not installed
                             -> pip install 'jevimage[serve]' - only to run `jevimage serve` yourself

  ready: API-only - this machine calls the server above and needs no model stack
exit=0
```

With no model stack and no server, there is nothing that can answer a question, and
that is the one case that exits 1:

```console
$ jevimage doctor
...
  --  server                 none running here, and $JEVIMAGE_URL is not set
  --  run models locally     torch no, transformers no
                             -> pip install 'jevimage[local]'   - or point $JEVIMAGE_URL at a server instead

  not ready: no local model stack and no server to call
  pick one: pip install 'jevimage[local]', or set $JEVIMAGE_URL to a server you can reach
exit=1
```

Lines marked `yes` are present, `NO` must be fixed, and `--` is a thing this install
does not have and does not need. `--url` checks a specific server instead of the one
`$JEVIMAGE_URL` names; with neither, `doctor` probes `127.0.0.1:8731` for a `jevimage serve`
you already have running.

A GPU is never required. If one is present but the model does not fit, the encoder
loads on the CPU and says so - and `doctor` names what is holding the card, because
knowing it is full does not tell you what to close:

```
  yes gpu                    NVIDIA GeForce RTX 3050 Laptop GPU, 0.0 GB free of 3.7 GB
                             -> too tight for a model; it will load on the CPU instead, which is about ten times slower
                                pid 1779614    900 MiB  python3 jevimage serve  <- STOPPED, still holding it
                                pid 2122998    896 MiB  python3 jevimage serve
                                pid 1786123    896 MiB  python3 jevimage serve  <- STOPPED, still holding it
```

A suspended process is called out by name: Ctrl-Z does not release a CUDA allocation,
and a stopped server shows up as nothing at all in `top` while still holding most of a
card. The list comes from nvidia-smi, which is installed wherever CUDA is; where it is
not, the line is simply absent.

On a laptop card this is the difference between a 0.08s answer and a 0.16s one - the
CPU forward pass is about ten times the GPU's, and is half of what a `jevimage ask`
costs end to end.

## `jevimage encoders`

Lists the registry and marks the default. No flags, no `--url`, no `--json`. Works
in an API-only install.

```console
$ jevimage encoders
   clip-vit-l-14
   dfn5b-h-14-384
   metaclip2-huge
 * siglip2-base-224
   siglip2-base-384
   siglip2-giant-384
   siglip2-so400m-384
   toy
   toy-calibrated

* default (siglip2-base-224). Join names with '+' for an ensemble, e.g. siglip2-base-224+dfn5b-h-14-384.
```

`toy` and `toy-calibrated` are the stand-ins this page's transcripts run on, registered
by a `sitecustomize.py` as described in [About the transcripts](#about-the-transcripts-on-this-page);
a stock install lists the seven above them and nothing else.

(`toy` and `toy-calibrated` are there because every transcript on this page runs with the
`sitecustomize.py` shim described under [Custom encoders and the
CLI](#custom-encoders-and-the-cli). A plain install lists the seven built-ins only.)

The `*` follows `$JEVIMAGE_ENCODER` when it is set, and the footnote says so, which is
how you catch an override you forgot was exported:

```console
$ JEVIMAGE_ENCODER=toy jevimage encoders
   clip-vit-l-14
   dfn5b-h-14-384
   metaclip2-huge
   siglip2-base-224
   siglip2-base-384
   siglip2-giant-384
   siglip2-so400m-384
 * toy
   toy-calibrated

* $JEVIMAGE_ENCODER=toy. Join names with '+' for an ensemble, e.g. siglip2-base-224+dfn5b-h-14-384.
```

Nothing here downloads weights. An entry in this list is a name plus a factory;
the model arrives on the first `ask` or `train` that uses it.

### Custom encoders and the CLI

`jevimage.register(name, factory)` is a Python call, and `jevimage` never imports your
code, so a registered encoder is invisible to the CLI unless you arrange for your
module to be imported at interpreter startup. The error says as much:

```console
$ jevimage ask ds/red/0.png --encoder nope --choice a=x b=y
jevimage: unknown encoder 'nope'. Available: clip-vit-l-14, dfn5b-h-14-384, metaclip2-huge, siglip2-base-224, siglip2-base-384, siglip2-giant-384, siglip2-so400m-384, toy, toy-calibrated. Register your own with jevimage.register(name, factory) - from Python, in the process that will serve it; the `jevimage` command only sees these built-in names, because it never imports your code.
exit=1
```

The two ways to make your own encoder reachable from a shell:

1. Put a `sitecustomize.py` on `PYTHONPATH` that imports your module and calls
   `register()`. This is what the transcripts on this page do to get `toy`.
2. Do not use the CLI for it. Build the `Jev` in Python and serve it:
   `uvicorn`-hosting `jevimage.server.build_app(jevimage.load(MyEncoder()))`.

---

## `jevimage ask`

```
jevimage ask IMAGE [IMAGE ...] [--choice NAME=DESCRIPTION ...] [--score LOW,...,HIGH]
              [--noul TEXT [TEXT]] [--head NAME]
              [--encoder ENCODER] [--url URL] [--local] [--timeout SECONDS]
              [--time] [--json]
```

One image is encoded once; every question is read against that one embedding. N
questions cost barely more than one, which is why all four flags repeat.

Several images may be given, and they share one connection - which is where the time
goes against a remote server. Measured over 8 files to a server one round trip away:

```console
$ time (for f in *.jpg; do jevimage ask "$f" --choice indoor=... outdoor=...; done)
real    0m9.10s      # 1138 ms each: TCP and TLS, eight times

$ time jevimage ask *.jpg --choice indoor=... outdoor=...
real    0m2.86s      #  358 ms each: the handshake once
```

`--json` keys the object by path when there is more than one image, and keeps the bare
`{question id: answer}` shape for one, so a script that reads a single answer is
unaffected.

| Flag | Repeats | Takes |
|---|---|---|
| `--choice` | yes | Two or more `name=description` pairs, space-separated. |
| `--score` | yes | One comma-separated string of ordered levels, lowest first. |
| `--noul` | yes | One caption (sigmoid readout), or two (`true` then `false`, the contrast pair). |
| `--head` | yes | The name of a trained head. |
| `--encoder` | no | Registry name, or `a+b` for an ensemble. Ignored with `--url`. Naming one also turns off the [local-server probe](#where-a-command-sends-the-work). |
| `--local` | no | Use this machine rather than a remote API: a `jevimage serve` running here, or the weights in this process. |
| `--time` | no | Print where the time went, on stderr. See [below](#where-the-time-goes). |
| `--url` | no | Remote instance. Also `$JEVIMAGE_URL`. Passed-but-empty is an error, not a request for local mode. |
| `--timeout` | no | Seconds to wait on `--url`. Must be positive. Default 60, also `$JEVIMAGE_TIMEOUT`. |
| `--json` | no | Print the raw answer objects instead of the table. |

Flag *syntax* is checked **before** any weights load - a malformed `name=description`
pair, a repeated `--choice` name, or three `--noul` captions costs you no model
download. The remaining bounds (option and level counts, an unknown head name) are the
core's, and it sees the question after the encoder is in memory.

### Where the time goes

`--time` splits one slow number into the parts that have different fixes:

```console
$ jevimage ask a.png --time --choice indoor="an indoor scene" outdoor="an outdoor scene"
using http://127.0.0.1:8731   (a jevimage serve is running here; --encoder NAME to load the weights in this process)
a.png   server http://127.0.0.1:8731

choice  [choice]  indoor   confidence 0.163
    indoor    0.582  #############
    outdoor   0.418  #########

  image         112 ms   2560x1440 RGBA png -> 19 KB
  answer         12 ms   server http://127.0.0.1:8731
  startup        26 ms   python, argparse, and importing pillow and http.client
                         ----
  total         150 ms
```

The same command with `--local`, which loads the weights into this process:

```console
  load        10638 ms   encoder siglip2-base-224, loaded into this process
  answer        161 ms   encoder siglip2-base-224
  startup        41 ms   python, argparse, and importing pillow and http.client
                         ----
  total       10840 ms
```

| line | what it is | when it is large |
|---|---|---|
| `image` | reading the file and shrinking it to what the encoder will look at | A big PNG. 2560x1440 RGBA inflates to 14 MB, and zlib is serial, so the other fifteen cores watch. The same picture as a JPEG measured 0.17s against the PNG's 0.26s: a JPEG decoder can scale during the DCT, a PNG decoder cannot. |
| `answer` | the request, and the forward pass at the other end | A server on the CPU. The forward is about ten times the GPU's and is matmul-bound at roughly half the CPU's peak FLOPs, so there is no trick left in it. `jevimage doctor` names what is holding the card. |
| `load` | reading the weights into this process | Only appears without a server. Ten seconds, every time, which is the whole reason `ask` uses a `jevimage serve` when one is running. |
| `startup` | python, argparse, and importing pillow and http.client | Nothing you can change; it is what a command costs to be a command. Several questions in one `jevimage ask` pay it once, which is why the flags repeat. |

The breakdown goes to stderr, so `--time --json` still pipes. Reported in stages rather
than as one profile because the numbers are only useful if you can act on them: they
distinguish "send a smaller file" from "close whatever has the GPU".

### Question ids

`--choice` becomes `choice`, a second `--choice` becomes `choice2`, and so on
per type. A `--head` question is named after the head. Ids are how you find an
answer in `--json` output.

```console
$ jevimage ask ds/red/0.png --encoder toy \
    --choice red="a red square" blue="a blue square" \
    --choice bright="a bright image" dark="a dark image" \
    --noul "a red square" "a blue square" \
    --head colours
ds/red/0.png   encoder toy

choice  [choice]  red   confidence 1.000
    red    1.000  ######################
    blue   0.000  

choice2  [choice]  bright   confidence 1.000
    bright   1.000  ######################
    dark     0.000  

noul  [noul]  1.000   (contrast pair)

colours  [head]  red   confidence 0.972   (head 'colours', 12 examples, held out 50.0%)
    red    0.986  ######################
    blue   0.014  

exit=0
```

The id and type come first, then the verdict and `confidence`, then every class with its
probability and a bar. A `score` prints `1.28 / 2`, the expected level out of the top
index, not a percentage. A `head` answer also carries its provenance: how many examples
fitted it and its held-out accuracy, so a suspiciously confident answer from a head
trained on nine images says so on the same line. `noul` prints one number and its basis,
either `contrast pair` or `sigmoid readout`.

### `--json`

The table is for humans; `--json` is the contract. It prints exactly what the
Python API returns, which is what you pipe into `jq`.

```console
$ jevimage ask ds/red/1.png --encoder toy --choice red="a red square" blue="a blue square" --json
{
  "choice": {
    "type": "choice",
    "choice": "red",
    "probabilities": {
      "red": 0.701865,
      "blue": 0.298135
    },
    "confidence": 0.40373
  }
}
```

Note that `--json` prints no header line, so it does not record whether a server
or a local encoder answered.

### Failure modes

```console
$ jevimage ask ds/red/0.png --encoder toy
jevimage: ask needs at least one question: --choice name="a description" ..., --score "low,high", --noul "..." or --head NAME
exit=1

$ jevimage ask ds/red/0.png --encoder toy --choice indoor
jevimage: --choice takes name=description pairs; got 'indoor'. Try: --choice indoor="an indoor scene" outdoor="the outdoors"
exit=1

$ jevimage ask ds/red/0.png --encoder toy --noul a b c
jevimage: --noul takes one caption, or two for a true/false pair; got 3: ['a', 'b', 'c']
exit=1

$ jevimage ask missing.png --encoder toy --choice red="a red square" blue="a blue square"
jevimage: no such image file: 'missing.png'. Pass a path that exists, raw image bytes, a PIL.Image, or a 'data:image/...;base64,...' string.
exit=1

$ jevimage ask ds/red/0.png --encoder toy --head nosuch
jevimage: unknown head 'nosuch'; trained heads: colours
exit=1
```

**Single-caption `--noul` needs a calibrated encoder.** The one-caption form
turns a cosine into a probability through a sigmoid, which requires the encoder
to publish both a `logit_scale` and a `logit_bias`. SigLIP has both. CLIP-family
models have a scale but no bias, and an ensemble has no bias either. It averages
its members' scales, but a bias fitted against one member's own negatives does not
transfer to a concatenated cosine. Rather than invent a bias and report a wrong
number, it refuses:

```console
$ jevimage ask ds/red/0.png --encoder toy --noul "a red square"
jevimage: encoder 'toy' has no calibrated sigmoid, so a single-caption noul cannot produce a probability. Give a {"true": ..., "false": ...} criteria pair instead, which works on every encoder.
exit=1

$ jevimage ask ds/red/0.png --encoder toy-calibrated --noul "a red square"
ds/red/0.png   encoder toy-calibrated

noul  [noul]  1.000   (sigmoid readout)

exit=0
```

The two-caption form works everywhere, and is what you want unless you know your
encoder is calibrated.

**A head refuses to run under a different encoder.** A head is a linear map in one
encoder's embedding space. Under another encoder the same matrix produces
confident nonsense, so it is checked, not attempted:

```console
$ jevimage ask ds/red/0.png --encoder toy-calibrated --head colours
jevimage: head 'colours' was trained on encoder 'toy' but this instance runs 'toy-calibrated'. Load 'toy', or retrain the head here.
exit=1
```

This bites most often when `$JEVIMAGE_ENCODER` is set in one shell and not another.

---

## `jevimage train`

```
jevimage train NAME FOLDER [--encoder ENCODER] [--url URL] [--local]
                      [--timeout SECONDS] [--epochs N] [--compare] [--json]
```

Fits a linear head on the frozen encoder. `FOLDER` has one sub-folder per label.
`--encoder` is ignored with `--url`, exactly as on `ask`: the head is trained in the
server's embedding space, and stamped with the server's encoder name. `--epochs`
defaults to 500; both sides require at least 1, and a server caps it at 5000. Locally, the head lands in
`$JEVIMAGE_HEADS_DIR` (see the environment table); with `--url` it stays on the server
and only its metadata comes back. Raise `--timeout` for a folder big enough that
uploading and encoding it outlasts the 60s default.

```console
$ jevimage train colours ds/ --encoder toy
trained 'colours': 12 examples, 2 classes, 50.0% accurate
  classes: blue, red
  encoder: toy

Use it:
  jevimage ask IMAGE --head colours
exit=0
```

### `--compare`: did training actually help?

The accuracy above is the head's own. Whether fitting it beat simply describing the
classes in a prompt is a different question, and `--compare` is where it is answered:

```console
$ jevimage train colours ds/ --encoder toy --compare
trained 'colours': 12 examples, 2 classes, 50.0% accurate
  ...

class   n  zero-shot  trained   delta
blue    6      50.0%    50.0%    0.0%
red     6     100.0%    50.0%  -50.0%
ALL    12      75.0%    50.0%  -25.0%

trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example.
Training did not beat the prompts here - try more examples per class, or a bigger encoder.
```

`trained` is held-out k-fold accuracy (k = min(5, the thinnest class)), never the
fit. `zero-shot` is what plain prompts scored on the same images. `delta` is the
answer to "did training actually help", per class, and it is allowed to be
negative - on classes the encoder already knows, it usually is. When the overall
delta is zero or below, the command says so instead of letting you read a table and
conclude the opposite:

```
Training did not beat the prompts here - try more examples per class, or a bigger encoder.
```

A `-` in any accuracy column means "not measured", not zero: k-fold reports no
accuracy for a class too thin to hold anything out.

The same command with `--url` (or `$JEVIMAGE_URL`) trains on the server. Identical
output, identical numbers; the images are uploaded once:

```console
$ JEVIMAGE_URL=http://127.0.0.1:8123 jevimage train colours ds/
trained 'colours': 12 examples, 2 classes, 50.0% accurate
  classes: blue, red
  encoder: toy

Use it:
  jevimage ask IMAGE --head colours
exit=0
```

### Failure modes

```console
$ jevimage train colours nosuchfolder --encoder toy
jevimage: 'nosuchfolder' is not a directory
exit=1

$ jevimage train e empty_ds/ --encoder toy
jevimage: no labelled images under 'empty_ds/'. Expected one sub-folder per label, each holding .jpeg/.jpg/.png/.webp files.
exit=1

$ jevimage train thin thin_ds/ --encoder toy
jevimage: every class needs at least 2 examples; too few for: red
exit=1

$ jevimage train colours ds/ --epochs 6000        # against a server
jevimage: {"detail": [{"type": "less_than_equal", "loc": ["body", "epochs"], "msg": "Input should be less than or equal to 5000", "input": 6000, "ctx": {"le": 5000}}]} [HTTP 422 from POST http://127.0.0.1:8123/v1/heads]
exit=1
```

That second message is the server's validation error passed through verbatim.
It is ugly, and it is the truth. The epoch bound is the server's, not the client's, so
an API-only install cannot pre-check it. Local training has no such cap; 5000 epochs on a
small dataset is seconds. The *floor* is shared: `--epochs 0` is refused either way,
because a head fitted in zero steps is all zeros and would still be saved and listed.

The other bounds are the core's, and apply locally and remotely alike: at least 2
examples per class (k-fold has to hold one out), at most 4000 examples and at
most 100 classes per head.

Head names become filenames and are checked rather than sanitised, because
silently rewriting a name means `ask` cannot find what `train` said it saved:

```console
$ jevimage rm "../etc"
jevimage: invalid head name '../etc': use 1-64 characters from [A-Za-z0-9._-], starting with a letter or digit
exit=1
```

---

## `jevimage heads`

```
jevimage heads [--url URL] [--local] [--json]
```

Lists trained heads. If a `jevimage serve` is running here it lists that server's, so
the list matches what `ask` will answer from; `--local` and `--url` override, as
[above](#where-a-command-sends-the-work). Locally it reads the head directory
directly, because listing what you trained should not download an encoder's weights,
though the local branch does need torch, for the reason in the note above.

```console
$ jevimage heads
head     encoder  classes   n  held out     created
colours      toy        2  12     50.0%  2026-09-23

/tmp/.../scratchpad/dochome
exit=0
```

The last line is where the heads came from: a directory locally, a URL with
`--url`. An empty result is not an error:

```console
$ jevimage heads
no heads in /tmp/.../scratchpad/dochome. Train one: jevimage train NAME FOLDER
exit=0
```

### Unreadable head files

A `.pt` file that will not load is reported on **stderr** and the listing
continues. It is not silently skipped, because a later "no head named x" about a
file sitting right there in the directory would be a lie:

```console
$ jevimage heads
junk.pt is present but unreadable: UnpicklingError: Weights only load failed. In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.
head     encoder  classes   n  held out     created
colours      toy        2  12     50.0%  2026-09-23

/tmp/.../scratchpad/dochome
exit=0
```

The reason is truncated to 160 characters. Torch's unpickling error is several
paragraphs, and this is a listing. `--json` carries the same file under its
`broken` key, so a script reading only stdout hears about it too.

Three things put a file here: it will not load at all; it loads but is not a head
(someone else's checkpoint, a bare tensor - anything without `W`/`classes`/`n`);
or its filename is not a valid head name, so no command could address it. All
three are reported the same way and none of them stops the listing.

A directory that cannot be *read* is a different thing and is an error, not an
empty listing - `jevimage heads` and `jevimage rm` both say `Permission denied` rather than
disagreeing about whether the head is there.

Exit status stays `0`. A broken file is a warning, not a failure of the listing.

### `--json`

```console
$ jevimage heads --json
{
  "heads": [
    {
      "name": "colours",
      "encoder": "toy",
      "classes": [
        "blue",
        "red"
      ],
      "n": 12,
      "counts": {
        "blue": 6,
        "red": 6
      },
      "cv_accuracy": 0.5,
      "per_class": {
        "blue": 0.5,
        "red": 0.5
      },
      "folds": 5,
      "created": 1790185352.5471728,
      "dim": 16,
      "bytes": 136
    }
  ],
  "directory": "/tmp/.../dochome"
}
```

An object, not a bare array: `heads` is the listing, `directory` is the same last
line the table prints, and `broken` appears when a `.pt` file would not load.
That is the one channel a `--json` script has for the warning the table puts on stderr.
It is absent when nothing is broken, as above. With `junk.pt` back in the
directory (`heads` unchanged, cut short here):

```console
$ jevimage heads --json
{
  "heads": [
    ...
  ],
  "directory": "/tmp/.../dochome",
  "broken": {
    "junk": "UnpicklingError: Weights only load failed. In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch."
  }
}
```

Never the weights. A head entry is the same metadata object the HTTP API returns.
`bytes` is what the weights would cost: 136 bytes for a 16-dim, 2-class head.
Real encoders are 768 to 1536 dims, so tens of kilobytes.

---

## `jevimage rm`

```
jevimage rm NAME [--url URL]
```

Deletes one head. No confirmation prompt and no undo; a head is cheap to retrain,
and the command is used in scripts.

```console
$ jevimage rm colours
deleted 'colours'
exit=0

$ jevimage rm colours
jevimage: no head named 'colours' in /tmp/.../scratchpad/dochome
exit=1

$ JEVIMAGE_URL=http://127.0.0.1:8123 jevimage rm colours
jevimage: no head named 'colours' on http://127.0.0.1:8123
exit=1
```

Deleting something that was not there is exit 1, not a silent success. In a
script, a typo'd head name should not look like a completed cleanup.

This is the one local command that works in an API-only install: deleting a file
never reads it.

```console
$ jevimage rm quick          # API-only venv, no torch installed
deleted 'quick'
exit=0
```

---

## `jevimage serve`

```
jevimage serve [--host HOST] [--port PORT] [--encoder ENCODER]
          [--heads-dir DIR] [--api-key KEY [KEY ...]]
```

Hosts the HTTP API. Needs `pip install 'jevimage[serve]'`. Defaults: host
`127.0.0.1`, port `8731`, the default encoder, the default heads directory, **no
authentication**.

Port 8731 is deliberately not 8000: 8000 is the default of every other Python web
server on the machine, and losing that race is how `jevimage serve` used to die on
`[Errno 98] address already in use` after spending twelve seconds loading the model. If
8731 is taken too, `serve` walks up to 8740 and says which port it settled on:

```console
$ jevimage serve
port 8731 is in use; serving on 8732 instead
INFO:     Started server process [1799834]
INFO:     Application startup complete.
```

The port is bound before uvicorn is handed it, not tested and then bound, so nothing
can take it in between. `--port` names one yourself; it walks from there on the same
rule, and `--port 0` lets the kernel choose. With all ten taken you get one error
naming the range instead of errno 98 out of uvicorn. `ask`, `train` and `heads`
[sweep the same range](#where-a-command-sends-the-work), so a server that walked
is still the one they use.

```console
$ JEVIMAGE_ENCODER=toy jevimage serve --port 8131 --heads-dir /tmp/docsrv --api-key s3cret
INFO:     Started server process [2686431]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8131 (Press CTRL+C to quit)
```

**The flags are passed through the environment**, not into the app object:
`--encoder` sets `JEVIMAGE_ENCODER`, `--heads-dir` sets `JEVIMAGE_HEADS_DIR`, `--api-key`
sets `JEVIMAGE_API_KEYS`. The same knobs therefore work when a container or a process
manager starts uvicorn itself:

```sh
JEVIMAGE_ENCODER=siglip2-base-224 JEVIMAGE_API_KEYS=k1,k2 uvicorn jevimage.server:app
```

**The encoder loads at startup**, not on the first request; importing the app
builds it. A `jevimage serve` that has printed "Application startup complete" has its
weights in memory.

**`--api-key` takes several keys**, so clients can migrate to a new key before the old
one is withdrawn. Any one of them authenticates; anything else is a 401. Withdrawing the
old key is a restart - the list is read once, at startup, from `$JEVIMAGE_API_KEYS` - and the
port refuses connections while the encoder reloads, so the rotation is seamless for
callers but the revocation is not free. See
[Serving](serving.md#api-keys) for the systemd shape. The flag also repeats, so
`--api-key old --api-key new` is the same as `--api-key old new`:

```console
$ jevimage serve --port 8135 --api-key k1 k2
...
$ JEVIMAGE_API_KEY=k1 jevimage heads --url http://127.0.0.1:8135
head     encoder  classes   n  held out     created
colours      toy        2  12     50.0%  2026-09-23

http://127.0.0.1:8135
$ JEVIMAGE_API_KEY=k2 jevimage heads --url http://127.0.0.1:8135
head     encoder  classes   n  held out     created
colours      toy        2  12     50.0%  2026-09-23

http://127.0.0.1:8135
$ JEVIMAGE_API_KEY=k3 jevimage heads --url http://127.0.0.1:8135
jevimage: send 'Authorization: Bearer <key>' with a key this server was started with (JEVIMAGE_API_KEYS) [HTTP 401 from GET http://127.0.0.1:8135/v1/heads]
exit=1
```

Without `--api-key` the API is open to anyone who can reach the port, including
`POST /v1/heads`, which trains, and `DELETE /v1/heads/{name}`, which deletes.
`--host 0.0.0.0` with no key exposes both. `GET /v1/health` stays open even when
keys are set, so a load balancer does not need a secret to ask whether the
process is alive.

A key is one argument with no space or comma in it. `$JEVIMAGE_API_KEYS` - which is how
`--api-key` reaches the server process - is split on both, so a passphrase like
`correct horse battery staple` would authenticate as four short keys while the
passphrase itself got a 401. That is refused at parse time, as is an empty key:
`--api-key "$JEVIMAGE_KEY"` with `$JEVIMAGE_KEY` unset asked for authentication, and starting
an open server instead is the one outcome that must not be silent.

### Failure modes

```console
$ jevimage serve                                   # no uvicorn installed
jevimage: jevimage serve needs uvicorn: pip install 'jevimage[serve]'
exit=1

$ jevimage serve --api-key ""                      # e.g. --api-key "$UNSET_VAR"
jevimage: --api-key was given an empty key, which would start an OPEN server; pass a real key, or omit the flag to serve openly on purpose
exit=1

$ jevimage serve --api-key "correct horse battery staple"
jevimage: --api-key 'correct horse battery staple': a key cannot contain a space or a comma, because $JEVIMAGE_API_KEYS is split on both
exit=1

$ jevimage serve --port 99999
usage: jevimage serve [-h] [--host HOST] [--port PORT] [--encoder ENCODER] [--heads-dir HEADS_DIR] [--api-key KEY [KEY ...]]
jevimage serve: error: argument --port: port 99999 is outside 0-65535
exit=2

$ jevimage serve --heads-dir ./afile               # a file, not a directory
jevimage: --heads-dir './afile' is a file, not a directory
exit=1

$ JEVIMAGE_ENCODER=bogus jevimage serve --port 8132
jevimage: unknown encoder 'bogus'. Available: clip-vit-l-14, dfn5b-h-14-384, metaclip2-huge, siglip2-base-224, ...
exit=1
```

A port already in use is uvicorn's own error, and it exits 1 after a clean
shutdown:

```console
$ jevimage serve --port 8124
INFO:     Started server process [2686170]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
ERROR:    [Errno 98] error while attempting to bind on address ('127.0.0.1', 8124): address already in use
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
exit=1
```

Note the ordering: application startup (which loads the encoder) happens before
the bind. A model download will complete before you learn the port was taken.

The routes this serves are documented in [HTTP API](reference-http.md).

---

## Things the CLI does not do

Know these before you build a script on top.

- **No `--api-key` flag on the client commands.** `$JEVIMAGE_API_KEY` only.
- **No `--heads-dir` flag except on `serve`.** `$JEVIMAGE_HEADS_DIR` or `$JEVIMAGE_HOME`.
- **No `--temperature`.** `ask` uses `1 / logit_scale` (the reciprocal of the
  encoder's learned scale), or 0.01 for an encoder that publishes none. Pass `temperature=`
  to `jevimage.load()` if you need to sharpen or flatten the distributions.
- **No batch mode.** One image per `ask`. The HTTP API takes up to 8 per request;
  a loop over `jevimage ask` re-encodes the text captions every time. For many images,
  write the four-line Python loop; see [Python API](reference-python.md).
- **No `compare` command.** `jevimage train` prints the comparison once; to re-run it
  later, use `jev.compare(name)` in Python or `GET /v1/heads/{name}/compare`.
- **`--json` is only on `ask` and `heads`.** `train` and `rm` print prose.

---

[← Serving](serving.md) · [Docs index](index.md) · [Python reference →](reference-python.md)
