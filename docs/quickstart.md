---
title: "Quickstart"
description: "From nothing to a working answer, twice: running the models yourself and calling a server."
---

Ten minutes, from an empty environment to a calibrated answer and a trained head.
Everything here is shown twice: once with `jevimage.load()`, which runs the encoder in
your process, and once with `jevimage.connect(url)`, which calls a `jev serve` instance
over HTTP. The two have the same method names and return the same objects, so the
choice is one line in your code.

**About the numbers on this page.** The outputs below were produced with a deterministic
toy encoder substituted for SigLIP, so that every block is reproducible in a second and
costs no GPU. The shapes, keys, types and error messages are real; the probabilities are
the toy's, not SigLIP's, and mean nothing about red squares. Where a line would differ
under a real encoder, it says so.

## Install

```bash
pip install jevimage              # API-only: pillow + stdlib. No torch, no weights.
pip install 'jevimage[local]'     # + torch, transformers. load() runs the encoder here.
pip install 'jevimage[serve]'     # + fastapi, uvicorn. `jev serve` hosts it.
```

**Not on PyPI yet.** Until it is, those three lines mean the equivalent against a clone:
`git clone <repo> && cd jevimage-oss`, then `pip install .`, `pip install '.[local]'`,
`pip install '.[serve]'`. See [Install](/docs/install#from-source).

The base install is the whole client. `jevimage.connect(url)` works fully under it,
including `train()`, and so does `jev ask/train/heads/rm --url`. What it cannot do is
run a model: `jevimage.load()` raises an `ImportError` that names the fix. See
[When it goes wrong](#when-it-goes-wrong).

There is a fourth extra, `'jevimage[openclip]'`, needed only for the `dfn5b-h-14-384`
registry entry; those weights are Apple-licensed, which is why they are opt-in.

## Ask one image a question

A question is a dict: a `type` and some `criteria`. `criteria` for a `choice` maps your
label to a *caption* — dual encoders were trained on captions, not questions, so
"an indoor scene" reads better than "is it indoors?".

The images below are a 12-square fixture, `ds/red/0..5.png` and `ds/blue/0..5.png`,
written by [`examples/make_fixtures.py`](https://github.com/jevimage/jevimage/blob/main/examples/make_fixtures.py) - six lines of
Pillow, no download:

```python
from PIL import Image
import os

for label, base, jitter in (("red",  (200, 40, 40), (4, 6, -18, -4, 12, 11)),
                            ("blue", (40, 40, 200), (5, -1, 10, 2, 17, -7))):
    os.makedirs(f"ds/{label}", exist_ok=True)
    for i, d in enumerate(jitter):
        Image.new("RGB", (64, 64), tuple(c + d for c in base)).save(f"ds/{label}/{i}.png")
```

Run it in the directory you run the blocks from. The toy encoder standing in for SigLIP
hashes the raw pixels, so those exact squares are what the numbers below are numbers
*of*; your own photos give the same shapes and different numbers.

### Running the encoder yourself

```python
import json

import jevimage

jev = jevimage.load()                      # siglip2-base-224; downloads once, then cached
a = jev.ask("ds/red/0.png", {
    "colour": {"type": "choice",
               "criteria": {"red": "a red square", "blue": "a blue square"}},
})
print(a["colour"])
print(a["colour"].choice, a["colour"].probabilities["red"], a["colour"].confidence)
```

```
<choice 'red' p=1.000>
red 1.0 0.999999
```

(The line actually run was `jevimage.load(Toy(), heads_dir="qs_heads")` — the toy
encoder standing in for SigLIP, at its default temperature. A real encoder will not
report 1.0 on a question like this; it would be somewhere short of certain.)

`ask` takes a path, a `Path`, raw JPEG/PNG/WebP bytes, a `PIL.Image`, or a
`data:image/...;base64,...` string. A string that is neither an existing path nor a data
URL raises `FileNotFoundError` rather than being guessed at as base64.

### Against a server

Same script, one line different, no torch on this machine at all:

```python
import jevimage

jev = jevimage.connect("http://127.0.0.1:8123")     # api_key="..." if the server has one
a = jev.ask("ds/red/0.png", {
    "colour": {"type": "choice",
               "criteria": {"red": "a red square", "blue": "a blue square"}},
})
print(a["colour"])
print(jev.health())
```

```
<choice 'red' p=1.000>
{'encoder': 'toy', 'dim': 16, 'device': 'cpu', 'primitives': ['choice', 'score', 'noul', 'head'], 'heads': 0, 'auth': 'open', 'version': '0.1.0'}
```

Identical to the local answer above here, digit for digit. The same engine, on the other
side of a socket. (`heads` counts the heads that server holds, so yours will report its
own; this one was empty.) `health()` is the one method that exists only on the remote
side: it is how you find out which encoder your answers and heads are coming from, since
there is no encoder object on this end of the wire.

One caveat before you write `assert local == remote` in a test. The answers here matched
digit for digit, and probabilities are rounded to 6 decimals before you see them, but the
two sides are not promised to be bit-identical: captions are encoded in whatever batch a
request happens to need, float reductions are not associative, and a GPU encoder runs
bf16 while a CPU one runs fp32. Compare with a tolerance rather than `==`, and pin
`JEV_DEVICE=cpu` on both sides if you need the same digits every time.

To start the server yourself: `pip install 'jevimage[serve]'` then `jev serve`
(`--host`, `--port`, `--encoder`, `--api-key`). The same thing from the CLI, against
the same server:

```bash
jev ask ds/red/0.png --choice red="a red square" blue="a blue square" --url http://127.0.0.1:8123
```

```
ds/red/0.png   server http://127.0.0.1:8123

choice  [choice]  red   confidence 1.000
    red    1.000  ######################
    blue   0.000
```

`--url` is also `$JEV_URL`, with `$JEV_API_KEY` for the key. Without either, `jev ask`
loads a local encoder.

## What an answer is

`ask` returns `{question id: Answer}`. `Answer` is a `dict` subclass with attribute
access, so `a.choice` and `a["choice"]` are the same thing and `json.dumps(a)` works
unchanged. `print()` gives the short repr, which is *not* the whole answer:

```python
print(a["colour"])
print(json.dumps(a["colour"], indent=2))
```

```
<choice 'red' p=1.000>
{
  "type": "choice",
  "choice": "red",
  "probabilities": {
    "red": 1.0,
    "blue": 0.0
  },
  "confidence": 0.999999
}
```

`confidence` is not `max(probabilities)`. It is rescaled so that a uniform distribution
is 0 and certainty is 1, because 0.5 over two options is lukewarm and 0.5 over fifty is
emphatic. That rescaling is what lets you use one threshold across differently shaped
questions. Asking an `Answer` for a key it does not carry raises `AttributeError`
listing the keys it does have.

## Many questions, one encode

The image is encoded once; every question is a matmul against that one vector. So ask
them together rather than in a loop. If you will come back to the same image later,
keep the embedding and use `ask_embedding` (local only: its entire purpose is skipping
the encode, and over HTTP you would be uploading the embedding anyway).

`jev` here is the local one from the first example, not the `connect()` one. The
`trained` question reads the head trained in [the next
section](#train-a-head-on-your-own-images); run this before that one and the whole call
raises `ValueError: unknown head 'colours'; trained heads: none trained yet`, because
every question is planned before any is answered.

```python
emb = jev.embed("ds/red/0.png")          # the only thing that touches the encoder
print(emb.shape, emb.dtype)

answers = jev.ask_embedding(emb, {
    "colour": {"type": "choice", "criteria": {"red": "a red square",
                                              "blue": "a blue square"}},
    "busy":   {"type": "score", "criteria": ["empty", "half full", "packed"]},
    "outside":{"type": "noul", "criteria": {"true": "outdoors", "false": "indoors"}},
    "trained":{"type": "head", "head": "colours"},
})
for qid, a in answers.items():
    print(qid, a)
print(answers["busy"]["legend"], answers["busy"].score)
```

```
torch.Size([1, 16]) torch.float32
colour <choice 'red' p=1.000>
busy <score 1.28/2>
outside <noul 0.046>
trained <head 'red' p=0.986>
{'0': 'empty', '1': 'half full', '2': 'packed'} 1.2774
```

Four question types, all with that shape: `choice` (2..255 named options), `score` (an
ordered 2..10 level rubric, answered as an expected level index), `noul` (one yes/no
probability), and `head` (a classifier you trained). A `score`'s `probabilities` are
keyed by level number, and `legend` maps those numbers back to your rubric text.

The `noul` form above gives `criteria` a `true`/`false` caption pair, which works on
every encoder. The single-caption form (`instructions` instead of `criteria`) needs an
encoder that publishes both `logit_scale` and `logit_bias`, and raises rather than
inventing a probability. SigLIP publishes both; CLIP publishes neither, and an ensemble
has no bias. Prefer the pair.

## Train a head on your own images

`train()` fits a linear head on the **frozen** encoder: seconds of work, tens of
kilobytes on disk, one extra matmul at query time. The layout is one sub-folder per
label (`ds/red/*.png`, `ds/blue/*.png`), or a `{label: [images]}` mapping, or a list of
`(image, label)` pairs.

```python
head = jev.train("colours", "ds/")
print(head)
print(head.accuracy, head.counts)
print(json.dumps(head.comparison, indent=2))
```

```
<Head 'colours' 2 classes, 12 examples, 50.0% held out>
0.5 {'blue': 6, 'red': 6}
{
  "overall": {
    "zero_shot": 0.75,
    "trained": 0.5,
    "delta": -0.25
  },
  "per_class": [
    {
      "label": "blue",
      "n": 6,
      "zero_shot": 0.5,
      "trained": 0.5
    },
    {
      "label": "red",
      "n": 6,
      "zero_shot": 1.0,
      "trained": 0.5
    }
  ],
  "basis": "trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example"
}
```

`accuracy` is stratified k-fold held-out accuracy, with k = min(5, thinnest class) —
never the accuracy on the images it was fitted to. With 6 examples per class that number
is worth very little, and the toy encoder makes it worth nothing at all; read it next to
`n` and `counts` before believing it. A class with fewer than two examples is refused
outright, so k is always at least 2 and a head `train()` returns always carries a number
here.

`comparison` is the part to look at: your trained head against plain zero-shot prompts,
on the same images, **per class**. Training does not always win. That is why this is
computed every time instead of being left to you to discover in production.

Now use it; the question type is `head`:

```python
a = jev.ask("ds/blue/2.png", {"c": {"type": "head", "head": "colours"}})
print(a["c"])
print(json.dumps(a["c"], indent=2))
```

```
<head 'blue' p=0.843>
{
  "type": "head",
  "head": "colours",
  "choice": "blue",
  "probabilities": {
    "blue": 0.843197,
    "red": 0.156803
  },
  "confidence": 0.686394,
  "trained_on": 12,
  "accuracy": 0.5
}
```

The answer carries `trained_on` and `accuracy`, so a downstream consumer can see how
much evidence is behind the number it is reading.

Locally, heads are files under `$JEV_HEADS_DIR`, else `$JEV_HOME/heads`, else
`~/.jev/heads` (one file per head, so a head can be copied or committed on its own);
`jevimage.load(..., heads_dir=...)` overrides it. `jev heads` lists them, `jev rm NAME`
deletes one.

### The same training, through a server

Uploads the images, fits on the server's encoder, leaves the weights there:

```python
import json

import jevimage

jev = jevimage.connect("http://127.0.0.1:8123")
head = jev.train("colours", "ds/")
print(head)
print(json.dumps(head.comparison["overall"], indent=2))
print(jev.ask("ds/blue/2.png", {"c": {"type": "head", "head": "colours"}})["c"])
```

```
<Head 'colours' 2 classes, 12 examples, 50.0% held out>
{
  "zero_shot": 0.75,
  "trained": 0.5,
  "delta": -0.25
}
<head 'blue' p=0.843>
```

Same `Head`, same numbers, no torch installed on the client. What crosses the wire is
metadata, never the weights: `list_heads()` gives you `name`, `encoder`, `classes`,
`n`, `counts`, `cv_accuracy`, `per_class`, `folds`, `created`, `dim`, `bytes`.

## When it goes wrong

These are the four you are most likely to hit first. All four messages below are verbatim.

**`load()` on an API-only install.** Nothing is broken; you asked to run a model in a
process that has no model runtime:

```
ImportError: jevimage.load runs the encoder in this process, which needs torch and
transformers: pip install 'jevimage[local]'
To use a server instead - no torch required - call jevimage.connect(url).
(underlying import error: No module named 'torch')
```

`jev encoders` still works there (it lists the registry without loading anything), and
so does every `--url` command.

**Single-caption `noul` on an encoder with no calibration.**

```
ValueError: encoder 'toy' has no calibrated sigmoid, so a single-caption noul cannot
produce a probability. Give a {"true": ..., "false": ...} criteria pair instead, which
works on every encoder.
```

**A head run under a different encoder.** A head is a matrix in one encoder's embedding
space; running it against another's would be silently wrong wherever the widths happen
to match, so the encoder name is recorded at training time and checked at query time:

```
ValueError: head 'colours' was trained on encoder 'toy' but this instance runs
'toy-calibrated'. Load 'toy', or retrain the head here.
```

**No server there.** `connect()` does not connect (it builds a URL), so the first
actual call is where you find out:

```
JevError: cannot reach jev at http://127.0.0.1:8199 ([Errno 111] Connection refused).
Start one with `jev serve`, or pass the URL it is actually listening on.
```

`JevError` subclasses `ValueError`, deliberately: a local `Jev` raises `ValueError` for
the same refusals, so an existing `except ValueError` keeps working when your `load()`
line becomes a `connect()` line. It carries `.status` (the HTTP code) and `.detail`
(the server's own sentence).

## Where to go next

- The four question types in full, including the shape of every answer and both `noul`
  forms: [Question types](/docs/questions).
- Choosing an encoder, joining two with `+`, or registering your own object with
  `.name` / `.dim` / `.img()` / `.txt()`: [Choosing an encoder](/docs/encoders).
- Serving: `jev serve`, API keys and deployment shapes in [Serving](/docs/serving); the
  wire contract route by route in [the HTTP reference](/docs/reference-http).
- Runnable versions of everything above: [`examples/quickstart.py`](https://github.com/jevimage/jevimage/blob/main/examples/quickstart.py)
  and [`examples/train_your_own.py`](https://github.com/jevimage/jevimage/blob/main/examples/train_your_own.py).

---

[← Install](/docs/install) · [Docs index](/docs/index) · [Question types →](/docs/questions)
