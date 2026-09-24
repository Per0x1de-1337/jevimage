---
title: "CLI"
description: "Every jev command and flag, exit codes, environment variables and remote mode."
---

`jev` is a thin shell over the Python API. It computes nothing that
`jevimage.Jev` and `jevimage.connect()` do not already compute, so a number that
looks wrong here is wrong upstream, not in the formatting. The output is plain
ASCII on purpose; it is meant to survive being pasted into an issue or a log.

```
jev serve | ask | train | heads | rm | encoders
```

Four of those (`ask`, `train`, `heads`, `rm`) take `--url` and drive a remote
`jev serve` instance instead of local weights. Remote mode runs the same code
path, because `jevimage.connect(url)` exposes the same methods as
`jevimage.load()` and the command body is written once against both.

Related pages: [Install](/docs/install) · [Quickstart](/docs/quickstart) ·
[Question types](/docs/questions) · [Encoders](/docs/encoders) ·
[Serving](/docs/serving) · [Python API](/docs/reference-python) ·
[HTTP API](/docs/reference-http)

## About the transcripts on this page

Every block below is a command that was run, with its real output. The one
substitution: the encoder is a **deterministic toy encoder** (a 16-dim
hash-of-the-bytes model registered as `toy` / `toy-calibrated`), not SigLIP,
because this machine has no network for model weights. The images are `ds/`, the twelve
squares [`examples/make_fixtures.py`](https://github.com/jevimage/jevimage/blob/main/examples/make_fixtures.py) writes. The toy is not
semantic (it will happily call a blue square "red"), so read the *shapes* of these
outputs, not the probabilities. Everything that is not a probability (columns, ids, error
text, exit codes, which install a command needs) is exactly what you will see.

Two cosmetic edits: long scratch-directory paths are elided to `/tmp/.../` so the
lines fit, and a repeated listing already printed in full above is cut short with
a `...` line. Nothing in a transcript is retyped.

The toy appears in `jev encoders` listings on this page because it was registered by a
`sitecustomize.py` on `PYTHONPATH`. That is also the only way to get a custom encoder
into the CLI at all. See [Encoders and the CLI](#custom-encoders-and-the-cli).

## What each command needs installed

| Command | `pip install jevimage` (no torch) | Needs `[local]` | Needs `[serve]` |
|---|---|---|---|
| `jev encoders` | works | — | — |
| `jev ask --url` / `train --url` / `heads --url` | works | — | — |
| `jev rm` (local) | works | — | — |
| `jev rm --url` | works | — | — |
| `jev ask` / `jev train` (local) | fails, exit 1 | yes | — |
| `jev heads` (local) | fails, exit 1 | yes | — |
| `jev serve` | fails, exit 1 | yes | yes |

`jev encoders` listing the registry without torch is deliberate: seeing which
encoders exist should not require the stack that runs them. `jev rm` gets there
by accident of being an `unlink`; it never has to read a head file.

The failure is one message, not a traceback, and it names both fixes:

```console
$ jev ask ds/red/0.png --choice red="a red square" blue="a blue square"
jev: this command runs the encoder locally, which needs torch:
       pip install 'jevimage[local]'
     or point it at a server that already has one:
       jev ask --url http://host:8000   (also $JEV_URL)
exit=1
```

The second half only appears on commands that can go remote. `jev serve` cannot,
so it gets its own message:

```console
$ jev serve
jev: jev serve needs uvicorn: pip install 'jevimage[serve]'
exit=1
```

`jev heads` and `jev train` produce the same two-part message as `jev ask`, with
the command name substituted.

> Listing *local* heads needs torch even though a head listing is only metadata:
> the head files are torch tensors, and the command imports the training module
> to format them. Listing a *server's* heads does not, which is the whole reason
> that import sits inside the function rather than at the top of the module.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. Also: `jev heads` with no heads, and Ctrl-C during `jev serve`. |
| `1` | An error you caused: bad flag value, unknown encoder or head, missing file, HTTP error from a server, a missing dependency. One line on stderr, no traceback. |
| `2` | argparse rejected the command line (unknown subcommand, missing positional). Usage is printed. |
| `130` | Ctrl-C during `ask`, `train`, `heads`, `rm`. Silent; nothing is printed. |

Exit 1 covers `ValueError`, `KeyError`, `OSError`, `ImportError` and `JevError`
(the client's HTTP error). Anything else is a bug and prints a traceback.

```console
$ jev frobnicate
usage: jev [-h] {serve,ask,train,heads,rm,encoders} ...
jev: error: argument command: invalid choice: 'frobnicate' (choose from 'serve', 'ask', 'train', 'heads', 'rm', 'encoders')
exit=2

$ jev
usage: jev [-h] {serve,ask,train,heads,rm,encoders} ...
jev: error: the following arguments are required: command
exit=2
```

Ctrl-C, here against a server that accepts the connection and never answers:

```console
$ jev heads --url http://127.0.0.1:8134      # then Ctrl-C
exit=130
```

`jev serve` is the exception: uvicorn installs its own SIGINT handler, shuts the
app down and returns normally, so Ctrl-C there exits `0`.

## Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `JEV_URL` | `ask`, `train`, `heads`, `rm` | Same as `--url`. An explicit `--url` wins. |
| `JEV_API_KEY` | `ask`, `train`, `heads`, `rm` | Bearer token for the remote instance. **There is no flag for this**; the environment is the only way to pass it. |
| `JEV_TIMEOUT` | `ask`, `train`, `heads`, `rm` | Seconds to wait on a remote instance. Default 60. `--timeout` (on `ask` and `train`) wins, including when it is `0`. Both must be a positive number; `0` or less is exit 1 either way. |
| `JEV_ENCODER` | `ask`, `train`, `serve`, `encoders` | Default encoder name. `--encoder` wins. `jev encoders` marks it with `*`. |
| `JEV_HEADS_DIR` | every local command | Where heads live. `jev serve --heads-dir` sets it for the server process. |
| `JEV_HOME` | every local command | Heads go to `$JEV_HOME/heads` when `JEV_HEADS_DIR` is unset. Default `~/.jev`. |
| `JEV_DEVICE` | local commands | `cuda`, `cpu`, `cuda:1`. Default: CUDA if available, else CPU. |
| `JEV_API_KEYS` | the server process | Keys the server accepts, comma- or space-separated. `jev serve --api-key` sets it. |

`JEV_HOME` at work. Note the `/heads` suffix the CLI appends:

```console
$ JEV_HOME=/tmp/dochome2 jev heads
no heads in /tmp/dochome2/heads. Train one: jev train NAME FOLDER
exit=0
```

## Remote mode: `--url` and `$JEV_URL`

Set either and `ask`, `train`, `heads` and `rm` run against a `jev serve`
instance: no torch, no weights, no GPU on this machine. `--url` beats `$JEV_URL`.
The last line of `jev ask` output tells you which backend answered, so a
transcript is self-describing:

```console
$ jev ask ds/red/0.png --url http://127.0.0.1:8123 \
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
$ jev ask ds/red/0.png --url http://127.0.0.1:8123 --encoder siglip2-giant-384 \
    --choice red="a red square" blue="a blue square"
ds/red/0.png   server http://127.0.0.1:8123

choice  [choice]  red   confidence 1.000
    red    1.000  ######################
    blue   0.000  

exit=0
```

`jev encoders` has no `--url`: it lists the registry compiled into *this* install,
which is not necessarily what the server runs. Ask the server itself:
`curl $JEV_URL/v1/health` returns its encoder name and dim.

### Authentication

```console
$ jev heads --url http://127.0.0.1:8131
jev: send 'Authorization: Bearer <key>' with a key this server was started with (JEV_API_KEYS) [HTTP 401 from GET http://127.0.0.1:8131/v1/heads]
     $JEV_API_KEY is not set in this shell. Note that a bare
     `JEV_API_KEY=...` line sets a shell variable, not an
     environment one - it needs `export`:
       export JEV_API_KEY=<key>
exit=1

$ JEV_API_KEY=s3cret jev heads --url http://127.0.0.1:8131
no heads in http://127.0.0.1:8131. Train one: jev train NAME FOLDER
exit=0
```

A wrong key gets the identical 401 *from the server* - it does not distinguish "no key"
from "wrong key", and should not. The printed output differs: the four `export` lines
above are the client's own hint, added only when `$JEV_API_KEY` is unset, so a
`JEV_API_KEY=wrong jev heads --url ...` prints the first line and nothing else.

### When the server is not there

```console
$ jev heads --url http://127.0.0.1:9999
jev: cannot reach jev at http://127.0.0.1:9999 ([Errno 111] Connection refused). Start one with `jev serve`, or pass the URL it is actually listening on.
exit=1

$ jev heads --url 127.0.0.1:8131
jev: url '127.0.0.1:8131' has no scheme; use 'http://127.0.0.1:8131'
exit=1
```

The scheme check is separate because urllib's own message ("unknown url type:
127.0.0.1") points at the host and sends you looking in the wrong place.

The client's default timeout is 60s. `ask` and `train` take `--timeout SECONDS`
for a cold server that is still loading a model, or a training job too big to
finish inside a minute; `$JEV_TIMEOUT` sets it for all four remote commands,
including the two with no flag:

```console
$ jev ask ds/red/0.png --url http://127.0.0.1:8134 --timeout 2 \
    --choice red="a red square" blue="a blue square"
jev: POST http://127.0.0.1:8134/v1/ask did not answer within 2.0s; raise timeout= if the server is loading a model
exit=1

$ JEV_TIMEOUT=2 jev heads --url http://127.0.0.1:8134
jev: GET http://127.0.0.1:8134/v1/heads did not answer within 2.0s; raise timeout= if the server is loading a model
exit=1
```

That is port 8134 again: a socket that accepts the connection and then says
nothing, which is what a timeout has to be provoked with.

---

## `jev encoders`

Lists the registry and marks the default. No flags, no `--url`, no `--json`. Works
in an API-only install.

```console
$ jev encoders
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

The `*` follows `$JEV_ENCODER` when it is set, and the footnote says so, which is
how you catch an override you forgot was exported:

```console
$ JEV_ENCODER=toy jev encoders
   clip-vit-l-14
   dfn5b-h-14-384
   metaclip2-huge
   siglip2-base-224
   siglip2-base-384
   siglip2-giant-384
   siglip2-so400m-384
 * toy
   toy-calibrated

* $JEV_ENCODER=toy. Join names with '+' for an ensemble, e.g. siglip2-base-224+dfn5b-h-14-384.
```

Nothing here downloads weights. An entry in this list is a name plus a factory;
the model arrives on the first `ask` or `train` that uses it.

### Custom encoders and the CLI

`jevimage.register(name, factory)` is a Python call, and `jev` never imports your
code, so a registered encoder is invisible to the CLI unless you arrange for your
module to be imported at interpreter startup. The error says as much:

```console
$ jev ask ds/red/0.png --encoder nope --choice a=x b=y
jev: unknown encoder 'nope'. Available: clip-vit-l-14, dfn5b-h-14-384, metaclip2-huge, siglip2-base-224, siglip2-base-384, siglip2-giant-384, siglip2-so400m-384, toy, toy-calibrated. Register your own with jevimage.register(name, factory) - from Python, in the process that will serve it; the `jev` command only sees these built-in names, because it never imports your code.
exit=1
```

The two ways to make your own encoder reachable from a shell:

1. Put a `sitecustomize.py` on `PYTHONPATH` that imports your module and calls
   `register()`. This is what the transcripts on this page do to get `toy`.
2. Do not use the CLI for it. Build the `Jev` in Python and serve it:
   `uvicorn`-hosting `jevimage.server.build_app(jevimage.load(MyEncoder()))`.

---

## `jev ask`

```
jev ask IMAGE [--choice NAME=DESCRIPTION ...] [--score LOW,...,HIGH]
              [--noul TEXT [TEXT]] [--head NAME]
              [--encoder ENCODER] [--url URL] [--timeout SECONDS] [--json]
```

One image is encoded once; every question is read against that one embedding. N
questions cost barely more than one, which is why all four flags repeat.

| Flag | Repeats | Takes |
|---|---|---|
| `--choice` | yes | Two or more `name=description` pairs, space-separated. |
| `--score` | yes | One comma-separated string of ordered levels, lowest first. |
| `--noul` | yes | One caption (sigmoid readout), or two (`true` then `false`, the contrast pair). |
| `--head` | yes | The name of a trained head. |
| `--encoder` | no | Registry name, or `a+b` for an ensemble. Ignored with `--url`. |
| `--url` | no | Remote instance. Also `$JEV_URL`. Passed-but-empty is an error, not a request for local mode. |
| `--timeout` | no | Seconds to wait on `--url`. Must be positive. Default 60, also `$JEV_TIMEOUT`. |
| `--json` | no | Print the raw answer objects instead of the table. |

Flag *syntax* is checked **before** any weights load - a malformed `name=description`
pair, a repeated `--choice` name, or three `--noul` captions costs you no model
download. The remaining bounds (option and level counts, an unknown head name) are the
core's, and it sees the question after the encoder is in memory.

### Question ids

`--choice` becomes `choice`, a second `--choice` becomes `choice2`, and so on
per type. A `--head` question is named after the head. Ids are how you find an
answer in `--json` output.

```console
$ jev ask ds/red/0.png --encoder toy \
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
$ jev ask ds/red/1.png --encoder toy --choice red="a red square" blue="a blue square" --json
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
$ jev ask ds/red/0.png --encoder toy
jev: ask needs at least one question: --choice name="a description" ..., --score "low,high", --noul "..." or --head NAME
exit=1

$ jev ask ds/red/0.png --encoder toy --choice indoor
jev: --choice takes name=description pairs; got 'indoor'. Try: --choice indoor="an indoor scene" outdoor="the outdoors"
exit=1

$ jev ask ds/red/0.png --encoder toy --noul a b c
jev: --noul takes one caption, or two for a true/false pair; got 3: ['a', 'b', 'c']
exit=1

$ jev ask missing.png --encoder toy --choice red="a red square" blue="a blue square"
jev: no such image file: 'missing.png'. Pass a path that exists, raw image bytes, a PIL.Image, or a 'data:image/...;base64,...' string.
exit=1

$ jev ask ds/red/0.png --encoder toy --head nosuch
jev: unknown head 'nosuch'; trained heads: colours
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
$ jev ask ds/red/0.png --encoder toy --noul "a red square"
jev: encoder 'toy' has no calibrated sigmoid, so a single-caption noul cannot produce a probability. Give a {"true": ..., "false": ...} criteria pair instead, which works on every encoder.
exit=1

$ jev ask ds/red/0.png --encoder toy-calibrated --noul "a red square"
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
$ jev ask ds/red/0.png --encoder toy-calibrated --head colours
jev: head 'colours' was trained on encoder 'toy' but this instance runs 'toy-calibrated'. Load 'toy', or retrain the head here.
exit=1
```

This bites most often when `$JEV_ENCODER` is set in one shell and not another.

---

## `jev train`

```
jev train NAME FOLDER [--encoder ENCODER] [--url URL] [--timeout SECONDS]
                      [--epochs N]
```

Fits a linear head on the frozen encoder. `FOLDER` has one sub-folder per label.
`--encoder` is ignored with `--url`, exactly as on `ask`: the head is trained in the
server's embedding space, and stamped with the server's encoder name. `--epochs`
defaults to 500; both sides require at least 1, and a server caps it at 5000. Locally, the head lands in
`$JEV_HEADS_DIR` (see the environment table); with `--url` it stays on the server
and only its metadata comes back. Raise `--timeout` for a folder big enough that
uploading and encoding it outlasts the 60s default.

```console
$ jev train colours ds/ --encoder toy
'colours': 12 examples, 2 classes, encoder toy

class   n  zero-shot  trained   delta
blue    6      50.0%    50.0%    0.0%
red     6     100.0%    50.0%  -50.0%
ALL    12      75.0%    50.0%  -25.0%

trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example.
Training did not beat the prompts here - try more examples per class, or a bigger encoder.
exit=0
```

`trained` is held-out k-fold accuracy (k = min(5, the thinnest class)), never the
fit. `zero-shot` is what plain prompts scored on the same images. `delta` is the
answer to "did training actually help", per class, and it is allowed to be
negative. When the overall delta is zero or below, the command says so instead of
letting you read a table and conclude the opposite:

```
Training did not beat the prompts here - try more examples per class, or a bigger encoder.
```

A `-` in any accuracy column means "not measured", not zero: k-fold reports no
accuracy for a class too thin to hold anything out.

The same command with `--url` (or `$JEV_URL`) trains on the server. Identical
output, identical numbers; the images are uploaded once:

```console
$ JEV_URL=http://127.0.0.1:8123 jev train colours ds/
'colours': 12 examples, 2 classes, encoder toy

class   n  zero-shot  trained   delta
blue    6      50.0%    50.0%    0.0%
red     6     100.0%    50.0%  -50.0%
ALL    12      75.0%    50.0%  -25.0%

trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example.
Training did not beat the prompts here - try more examples per class, or a bigger encoder.
exit=0
```

### Failure modes

```console
$ jev train colours nosuchfolder --encoder toy
jev: 'nosuchfolder' is not a directory
exit=1

$ jev train e empty_ds/ --encoder toy
jev: no labelled images under 'empty_ds/'. Expected one sub-folder per label, each holding .jpeg/.jpg/.png/.webp files.
exit=1

$ jev train thin thin_ds/ --encoder toy
jev: every class needs at least 2 examples; too few for: red
exit=1

$ jev train colours ds/ --epochs 6000        # against a server
jev: {"detail": [{"type": "less_than_equal", "loc": ["body", "epochs"], "msg": "Input should be less than or equal to 5000", "input": 6000, "ctx": {"le": 5000}}]} [HTTP 422 from POST http://127.0.0.1:8123/v1/heads]
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
$ jev rm "../etc"
jev: invalid head name '../etc': use 1-64 characters from [A-Za-z0-9._-], starting with a letter or digit
exit=1
```

---

## `jev heads`

```
jev heads [--url URL] [--json]
```

Lists trained heads. Locally it reads the head directory directly, because listing what
you trained should not download an encoder's weights, though the local branch does
need torch, for the reason in the note above.

```console
$ jev heads
head     encoder  classes   n  held out     created
colours      toy        2  12     50.0%  2026-09-23

/tmp/.../scratchpad/dochome
exit=0
```

The last line is where the heads came from: a directory locally, a URL with
`--url`. An empty result is not an error:

```console
$ jev heads
no heads in /tmp/.../scratchpad/dochome. Train one: jev train NAME FOLDER
exit=0
```

### Unreadable head files

A `.pt` file that will not load is reported on **stderr** and the listing
continues. It is not silently skipped, because a later "no head named x" about a
file sitting right there in the directory would be a lie:

```console
$ jev heads
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
empty listing - `jev heads` and `jev rm` both say `Permission denied` rather than
disagreeing about whether the head is there.

Exit status stays `0`. A broken file is a warning, not a failure of the listing.

### `--json`

```console
$ jev heads --json
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
$ jev heads --json
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

## `jev rm`

```
jev rm NAME [--url URL]
```

Deletes one head. No confirmation prompt and no undo; a head is cheap to retrain,
and the command is used in scripts.

```console
$ jev rm colours
deleted 'colours'
exit=0

$ jev rm colours
jev: no head named 'colours' in /tmp/.../scratchpad/dochome
exit=1

$ JEV_URL=http://127.0.0.1:8123 jev rm colours
jev: no head named 'colours' on http://127.0.0.1:8123
exit=1
```

Deleting something that was not there is exit 1, not a silent success. In a
script, a typo'd head name should not look like a completed cleanup.

This is the one local command that works in an API-only install: deleting a file
never reads it.

```console
$ jev rm quick          # API-only venv, no torch installed
deleted 'quick'
exit=0
```

---

## `jev serve`

```
jev serve [--host HOST] [--port PORT] [--encoder ENCODER]
          [--heads-dir DIR] [--api-key KEY [KEY ...]]
```

Hosts the HTTP API. Needs `pip install 'jevimage[serve]'`. Defaults: host
`127.0.0.1`, port `8000`, the default encoder, the default heads directory, **no
authentication**.

```console
$ JEV_ENCODER=toy jev serve --port 8131 --heads-dir /tmp/docsrv --api-key s3cret
INFO:     Started server process [2686431]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8131 (Press CTRL+C to quit)
```

**The flags are passed through the environment**, not into the app object:
`--encoder` sets `JEV_ENCODER`, `--heads-dir` sets `JEV_HEADS_DIR`, `--api-key`
sets `JEV_API_KEYS`. The same knobs therefore work when a container or a process
manager starts uvicorn itself:

```sh
JEV_ENCODER=siglip2-base-224 JEV_API_KEYS=k1,k2 uvicorn jevimage.server:app
```

**The encoder loads at startup**, not on the first request; importing the app
builds it. A `jev serve` that has printed "Application startup complete" has its
weights in memory.

**`--api-key` takes several keys**, so clients can migrate to a new key before the old
one is withdrawn. Any one of them authenticates; anything else is a 401. Withdrawing the
old key is a restart - the list is read once, at startup, from `$JEV_API_KEYS` - and the
port refuses connections while the encoder reloads, so the rotation is seamless for
callers but the revocation is not free. See
[Serving](/docs/serving#api-keys) for the systemd shape. The flag also repeats, so
`--api-key old --api-key new` is the same as `--api-key old new`:

```console
$ jev serve --port 8135 --api-key k1 k2
...
$ JEV_API_KEY=k1 jev heads --url http://127.0.0.1:8135
head     encoder  classes   n  held out     created
colours      toy        2  12     50.0%  2026-09-23

http://127.0.0.1:8135
$ JEV_API_KEY=k2 jev heads --url http://127.0.0.1:8135
head     encoder  classes   n  held out     created
colours      toy        2  12     50.0%  2026-09-23

http://127.0.0.1:8135
$ JEV_API_KEY=k3 jev heads --url http://127.0.0.1:8135
jev: send 'Authorization: Bearer <key>' with a key this server was started with (JEV_API_KEYS) [HTTP 401 from GET http://127.0.0.1:8135/v1/heads]
exit=1
```

Without `--api-key` the API is open to anyone who can reach the port, including
`POST /v1/heads`, which trains, and `DELETE /v1/heads/{name}`, which deletes.
`--host 0.0.0.0` with no key exposes both. `GET /v1/health` stays open even when
keys are set, so a load balancer does not need a secret to ask whether the
process is alive.

A key is one argument with no space or comma in it. `$JEV_API_KEYS` - which is how
`--api-key` reaches the server process - is split on both, so a passphrase like
`correct horse battery staple` would authenticate as four short keys while the
passphrase itself got a 401. That is refused at parse time, as is an empty key:
`--api-key "$JEV_KEY"` with `$JEV_KEY` unset asked for authentication, and starting
an open server instead is the one outcome that must not be silent.

### Failure modes

```console
$ jev serve                                   # no uvicorn installed
jev: jev serve needs uvicorn: pip install 'jevimage[serve]'
exit=1

$ jev serve --api-key ""                      # e.g. --api-key "$UNSET_VAR"
jev: --api-key was given an empty key, which would start an OPEN server; pass a real key, or omit the flag to serve openly on purpose
exit=1

$ jev serve --api-key "correct horse battery staple"
jev: --api-key 'correct horse battery staple': a key cannot contain a space or a comma, because $JEV_API_KEYS is split on both
exit=1

$ jev serve --port 99999
usage: jev serve [-h] [--host HOST] [--port PORT] [--encoder ENCODER] [--heads-dir HEADS_DIR] [--api-key KEY [KEY ...]]
jev serve: error: argument --port: port 99999 is outside 0-65535
exit=2

$ jev serve --heads-dir ./afile               # a file, not a directory
jev: --heads-dir './afile' is a file, not a directory
exit=1

$ JEV_ENCODER=bogus jev serve --port 8132
jev: unknown encoder 'bogus'. Available: clip-vit-l-14, dfn5b-h-14-384, metaclip2-huge, siglip2-base-224, ...
exit=1
```

A port already in use is uvicorn's own error, and it exits 1 after a clean
shutdown:

```console
$ jev serve --port 8124
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

The routes this serves are documented in [HTTP API](/docs/reference-http).

---

## Things the CLI does not do

Know these before you build a script on top.

- **No `--api-key` flag on the client commands.** `$JEV_API_KEY` only.
- **No `--heads-dir` flag except on `serve`.** `$JEV_HEADS_DIR` or `$JEV_HOME`.
- **No `--temperature`.** `ask` uses `1 / logit_scale` (the reciprocal of the
  encoder's learned scale), or 0.01 for an encoder that publishes none. Pass `temperature=`
  to `jevimage.load()` if you need to sharpen or flatten the distributions.
- **No batch mode.** One image per `ask`. The HTTP API takes up to 8 per request;
  a loop over `jev ask` re-encodes the text captions every time. For many images,
  write the four-line Python loop; see [Python API](/docs/reference-python).
- **No `compare` command.** `jev train` prints the comparison once; to re-run it
  later, use `jev.compare(name)` in Python or `GET /v1/heads/{name}/compare`.
- **`--json` is only on `ask` and `heads`.** `train` and `rm` print prose.

---

[← Serving](/docs/serving) · [Docs index](/docs/index) · [Python reference →](/docs/reference-python)
