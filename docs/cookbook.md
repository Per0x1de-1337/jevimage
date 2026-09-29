---
title: "Cookbook"
description: "Complete runnable recipes: moderation, catalogue tagging, caching embeddings, thresholds."
---

Eight complete scripts for things people build with jevimage. Each one is the
whole file: paste it, run it, get the output shown. They are ordered roughly from "read
one image" to "operate this in production", and each ends with the thing that will bite
you.

If you have not read [Question types](questions.md) yet, start there; this page assumes
you know what `choice`, `score`, `noul` and `head` mean.

## How these were run

Every block below was executed and its output pasted back verbatim. The machine had no
network for model weights, so all of them ran against a deterministic stand-in encoder
instead of SigLIP. That is `toy.py`, which hashes the pixel bytes and knows nothing
about anything:

```python
"""A deterministic encoder with no downloads - used to test the framework itself."""
import hashlib, torch, jevimage

class Toy(jevimage.Encoder):
    name, dim = "toy", 16
    logit_scale = logit_bias = None
    def __init__(self, seed=0):
        torch.manual_seed(seed); self.W = torch.randn(64, self.dim)
    def _e(self, xs):
        v = torch.stack([torch.tensor([b/255 for b in hashlib.sha512(str(x).encode()).digest()])
                         for x in xs])
        return torch.nn.functional.normalize(v @ self.W, dim=-1)
    def img(self, ims): return self._e([getattr(i, "tobytes", lambda: i)() for i in ims])
    def txt(self, ts): return self._e(list(ts))

class ToyCalibrated(Toy):
    name = "toy-calibrated"
    logit_scale, logit_bias = 100.0, -10.0
```

That means **the probabilities on this page are noise**. A hash of the pixel bytes knows
nothing about redness or clutter, so `jevimage.load(Toy())` calls a plain red square a
screenshot and is very sure about it. What is real: the shapes, the field names, the
control flow, the errors, and the relative cost of one encode versus many questions.
Swap `jevimage.load(Toy())` for `jevimage.load()` and the same scripts run on
siglip2-base-224 with answers that mean something.

The images are the 12 flat 64x64 squares in `ds/`, laid out the way `read_folder`
expects:

```text
ds/red/0.png  … ds/red/5.png
ds/blue/0.png … ds/blue/5.png
```

Write them with [`examples/make_fixtures.py`](https://github.com/Per0x1de-1337/jevimage/blob/main/examples/make_fixtures.py) in the
directory you run these scripts from. The stand-in encoder hashes the pixel bytes, so
every probability below is a number about *those* squares: with your own images the
shapes and the control flow are identical and the numbers are not.

The timings were measured on an otherwise quiet 16-thread laptop CPU (12th Gen Core
i7-12650H, `OMP_NUM_THREADS=1`, no GPU). They are microseconds of arithmetic, so they
move with the machine; the ratios between them are the part to read.

Recipe 7 also needs a server. Anything that speaks the API will do:

```console
$ jevimage serve --port 8123        # 'pip install jevimage[serve]' first
```

---

## 1. Screen one upload against every policy at once

**Problem:** a user uploads an image and five separate policies want an opinion on it
before it goes live.

```python
import time

import jevimage
from toy import Toy

jev = jevimage.load(Toy())          # stand-in encoder; yours is jevimage.load()
upload = "ds/red/0.png"

policy = {
    "nudity":   {"type": "noul", "criteria": {"true": "explicit nudity",
                                              "false": "clothed people, or no people"}},
    "weapon":   {"type": "noul", "criteria": {"true": "a firearm or a knife",
                                              "false": "no weapon of any kind"}},
    "medical":  {"type": "noul", "criteria": {"true": "a surgical or injury photo",
                                              "false": "an everyday photo"}},
    "screenshot": {"type": "noul", "criteria": {"true": "a screenshot of a user interface",
                                                "false": "a photograph of the world"}},
    "severity": {"type": "score",
                 "criteria": ["harmless", "mildly sensitive", "graphic"]},
}

t0 = time.perf_counter()
answers = jev.ask(upload, policy)
ask_ms = (time.perf_counter() - t0) * 1000

BLOCK = 0.90
for qid, a in answers.items():
    if a.type == "noul":
        print(f"{qid:<11} {a.noul:.3f}  {'BLOCK' if a.noul >= BLOCK else 'pass'}")
    else:
        print(f"{qid:<11} {a.score:.2f} / {len(a.legend) - 1}  ({a.legend[str(round(a.score))]})")

t0 = time.perf_counter()
jev.ask(upload, policy)                       # again, with the captions already encoded
warm_ms = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
jev.embed(upload)
embed_ms = (time.perf_counter() - t0) * 1000
print(f"\nfirst ask (11 captions cold): {ask_ms:.1f} ms")
print(f"same ask, captions cached:   {warm_ms:.1f} ms")
print(f"the image encode alone:      {embed_ms:.1f} ms")
```

```text
nudity      1.000  BLOCK
weapon      0.004  pass
medical     0.920  BLOCK
screenshot  1.000  BLOCK
severity    0.00 / 2  (harmless)

first ask (11 captions cold): 3.4 ms
same ask, captions cached:   0.4 ms
the image encode alone:      0.2 ms
```

Five policies, one forward pass. The captions are embedded once and cached per string
for the life of the process, which is why the second identical ask costs almost nothing
beyond the encode. Adding a sixth policy costs one more caption embedding the first time
and a matmul thereafter, so policies are cheap to add, and there is no reason to batch
them into one vague question. The 8x gap between the two asks is the stand-in encoder's:
on `siglip2-base-224` a cold caption costs a real text-tower pass, and the same split
measured 1020 ms cold against 0.39 ms cached for 16 questions. A *fixed* policy list is
what makes this pattern pay ([timings](encoders.md#timings)).

**Watch out:** a `noul` pair is a *contrast*, not an absolute. `0.920` for `medical`
means "closer to the true caption than the false one", not "92% likely to be a medical
photo", so the false caption does real work. `"no weapon of any kind"` is a much better
opposite than `""` or `"something else"`. One
`ask()` takes at most 64 questions; past that it raises rather than truncating.

---

## 2. Tag a product shot on every axis at once

**Problem:** one photo, four independent catalogue fields, one row out.

```python
import json

import jevimage
from toy import Toy

jev = jevimage.load(Toy())          # stand-in encoder; yours is jevimage.load()

axes = {
    "colour":   {"type": "choice", "criteria": {"red": "a red product",
                                                "blue": "a blue product",
                                                "neutral": "a black, white or grey product"}},
    "shot":     {"type": "choice", "criteria": ["a studio shot on plain background",
                                                "a lifestyle shot in a room",
                                                "a flat lay from above"]},
    "clutter":  {"type": "score", "criteria": ["one object, nothing else",
                                               "a few props",
                                               "a crowded scene"]},
    "on_model": {"type": "noul", "criteria": {"true": "a garment worn by a person",
                                              "false": "a garment with no person in frame"}},
}

a = jev.ask("ds/blue/2.png", axes)
for qid, ans in a.items():
    print(f"{qid:<9} {ans!r}")

print()
print(json.dumps(a["colour"], indent=2))
print(json.dumps(a["clutter"], indent=2))
print(json.dumps(a["on_model"], indent=2))

row = {"sku": "B-2",
       "colour": a["colour"].choice,
       "shot": a["shot"].choice,
       "clutter": round(a["clutter"].score, 2),
       "on_model": a["on_model"].noul >= 0.5}
print("\nrow:", row)
```

```text
colour    <choice 'blue' p=0.981>
shot      <choice 'a flat lay from above' p=0.995>
clutter   <score 1.00/2>
on_model  <noul 0.135>

{
  "type": "choice",
  "choice": "blue",
  "probabilities": {
    "red": 1e-06,
    "blue": 0.981453,
    "neutral": 0.018547
  },
  "confidence": 0.972179
}
{
  "type": "score",
  "score": 1.0,
  "probabilities": {
    "0": 0.0,
    "1": 0.999951,
    "2": 4.9e-05
  },
  "legend": {
    "0": "one object, nothing else",
    "1": "a few props",
    "2": "a crowded scene"
  },
  "confidence": 0.999926
}
{
  "type": "noul",
  "noul": 0.135053,
  "basis": "contrast pair"
}

row: {'sku': 'B-2', 'colour': 'blue', 'shot': 'a flat lay from above', 'clutter': 1.0, 'on_model': False}
```

`repr()` is the short form for logs; `json.dumps` is the full shape, because an `Answer`
is a `dict` subclass and needs no conversion step. A `score` is the expected level index.
`1.07` on a three-level rubric means "a few props, leaning very slightly towards crowded",
which is a more useful number for sorting than an argmax would be.

**Watch out:** `shot` used a bare list, so the label *is* the caption and the value you
store is a whole sentence. Use a `{label: description}` object whenever the label lands
in a database column. A bare list with two identical captions raises rather than
silently collapsing to one option.

---

## 3. Auto-approve what is confident, queue the rest

**Problem:** you want the model to handle the easy 90% and a human to see the rest.

```python
import jevimage
from toy import Toy

jev = jevimage.load(Toy())          # stand-in encoder; yours is jevimage.load()

question = {"colour": {"type": "choice",
                       "criteria": {"red": "a red product",
                                    "blue": "a blue product",
                                    "neutral": "a black, white or grey product"}}}

AUTO = 0.80                          # confidence, not probability
auto, review = [], []
for path in sorted(jevimage.read_folder("ds")["red"]):
    a = jev.ask(path, question)["colour"]
    (auto if a.confidence >= AUTO else review).append((path.name, a))
    print(f"{path.name}  {a.choice:<8} p={max(a.probabilities.values()):.3f} "
          f"confidence={a.confidence:.3f}  -> {'auto' if a.confidence >= AUTO else 'REVIEW'}")

print(f"\n{len(auto)} auto, {len(review)} queued for a human")

p = [0.5, 0.5]
print("confidence of a 2-way coin flip:", jevimage.confidence(p))
print("confidence of 0.5 over 50 options:",
      round(jevimage.confidence([0.5] + [0.5 / 49] * 49), 3))
```

```text
0.png  blue     p=0.717 confidence=0.575  -> REVIEW
1.png  blue     p=0.658 confidence=0.487  -> REVIEW
2.png  blue     p=0.735 confidence=0.603  -> REVIEW
3.png  blue     p=0.999 confidence=0.999  -> auto
4.png  blue     p=0.996 confidence=0.994  -> auto
5.png  red      p=1.000 confidence=1.000  -> auto

3 auto, 3 queued for a human
confidence of a 2-way coin flip: 0.0
confidence of 0.5 over 50 options: 0.49
```

Threshold `confidence`, not `max(probabilities)`. The last two lines are why: a top
probability of `0.5` is a coin flip over two options and a strong signal over fifty.
`confidence` is `(max - 1/K) / (1 - 1/K)`, so `0` always means uniform and `1` always
means certain, and one threshold survives you adding a third option to the question.

**Watch out:** confidence measures how *peaked* the distribution is, not how *right* it
is. The stand-in encoder above is confidently wrong on five of six images. Pick the
threshold by running a few hundred labelled examples through and looking at where the
errors sit. Remember that an image whose true answer is not among your options
still comes back confident, because the distribution only covers the options you gave.

---

## 4. Cache the embedding, ask new questions later

**Problem:** the images were encoded last week; today someone wants a new field, and
re-encoding the archive is the expensive part.

```python
import time

import torch

import jevimage
from toy import Toy

jev = jevimage.load(Toy())          # stand-in encoder; yours is jevimage.load()

paths = sorted(jevimage.read_folder("ds")["blue"])

# --- once, when the image arrives -------------------------------------------
embs = jev.embed(paths)
torch.save({"encoder": jev.encoder.name,
            "paths": [str(p) for p in paths],
            "emb": embs.half()}, "cache.pt")
print("cached", tuple(embs.shape), embs.dtype, "->", "cache.pt")

# --- later, a question nobody had thought of yet -----------------------------
blob = torch.load("cache.pt", weights_only=True)
assert blob["encoder"] == jev.encoder.name, "cache belongs to another encoder"

new_question = {"lighting": {"type": "choice",
                             "criteria": {"daylight": "a photo taken in daylight",
                                          "studio": "a photo taken under studio lights",
                                          "night": "a photo taken at night"}}}

t0 = time.perf_counter()
for name, e in zip(blob["paths"], blob["emb"].float()):
    a = jev.ask_embedding(e, new_question)["lighting"]
    print(f"{name:<16} {a.choice:<8} {a.confidence:.3f}")
cached_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
for p in paths:
    jev.ask(p, new_question)
fresh_ms = (time.perf_counter() - t0) * 1000
print(f"\n6 images from cache: {cached_ms:.1f} ms     re-encoding them: {fresh_ms:.1f} ms")

# .half() halves the file and moves the last decimals; the choice is unchanged
assert (jev.ask_embedding(blob["emb"][0].float(), new_question)["lighting"].choice
        == jev.ask(paths[0], new_question)["lighting"].choice)
```

```text
cached (6, 16) torch.float32 -> cache.pt
ds/blue/0.png    daylight 0.878
ds/blue/1.png    daylight 0.966
ds/blue/2.png    night    0.912
ds/blue/3.png    studio   0.124
ds/blue/4.png    studio   1.000
ds/blue/5.png    daylight 0.927

6 images from cache: 0.4 ms     re-encoding them: 1.4 ms
```

An embedding is `dim` floats (768 for siglip2-base-224, so 1.5 KB in fp16) and it is
the entire cost of a request. Store it next to the row, keep the original image
somewhere cold, and any future question is a matmul. The 3x here is small only because
the stand-in encoder is a hash function; with a real encoder on a GPU the ratio is the
whole forward pass against a few microseconds.

**Watch out:** do not key, hash or diff on the bytes of an embedding. It is stable for a
fixed encoder, device, dtype and batch composition, and not otherwise: encoding the same
image alone and inside a batch of four on `siglip2-base-224`/CPU differed by 1.8e-07,
which left the answer unchanged at six decimals but not the vector. `jevimage serve` runs with
batching on, so what an image embeds to depends on who else was in flight. Compare with a
tolerance.

**Watch out:** an embedding is meaningful only in the encoder that produced it, so store
`encoder.name` beside it and check it, as above; nothing downstream will notice
otherwise, and two encoders of the same `dim` will silently produce nonsense.
`ask_embedding()` exists only on a local `Jev`: its whole purpose is skipping the
network, so `connect()` does not have it. `Remote.embed()` still gives you the vectors:
a `torch.Tensor` if torch is installed, a plain list of lists on an API-only install,
where `.half()` and `.float()` do not exist.

---

## 5. Sweep a directory and route it into folders

**Problem:** a shoot lands as a flat directory and needs sorting into bins, with the
unusable frames set aside. Here `shoot/` holds the same 12 images as `ds/`, flat and
unlabelled, which is how they actually arrive. Flattening two label folders means two
`0.png`s, so the blue ones are prefixed: `ds/red/0-5.png` become `0-5.png`, and
`ds/blue/0-5.png` become `b0-b5.png`.

```python
import shutil
import time
from pathlib import Path

import jevimage
from toy import Toy

jev = jevimage.load(Toy())          # stand-in encoder; yours is jevimage.load()

Path("shoot").mkdir(exist_ok=True)                 # the flat drop, built from ds/
for src in sorted(Path("ds").rglob("*.png")):
    shutil.copy2(src, Path("shoot") /
                 (("b" if src.parent.name == "blue" else "") + src.name))

shoot = sorted(p for p in Path("shoot").rglob("*")
               if p.suffix.lower() in jevimage.IMAGE_SUFFIXES)
print(f"{len(shoot)} files")

BINS = {"type": "choice", "criteria": {"packshot": "a product alone on a plain background",
                                       "lifestyle": "a product being used by a person",
                                       "detail": "an extreme close-up of a material"}}
KEEP = {"usable": {"type": "noul", "criteria": {"true": "a sharp, well exposed photo",
                                                "false": "a blurred or badly exposed photo"}}}

t0 = time.perf_counter()
embs = jev.embed(shoot)             # one call, one forward pass per batch
encode_ms = (time.perf_counter() - t0) * 1000

routed = {}
for path, emb in zip(shoot, embs):
    a = jev.ask_embedding(emb, {"bin": BINS, **KEEP})
    bin_ = a["bin"].choice if a["usable"].noul >= 0.5 else "rejects"
    dst = Path("sorted") / bin_
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst / path.name)
    routed.setdefault(bin_, []).append(path.name)

read_ms = (time.perf_counter() - t0) * 1000 - encode_ms
for bin_, names in sorted(routed.items()):
    print(f"{bin_:<10} {len(names):>2}  {' '.join(names)}")
print(f"\nencode {len(shoot)} images: {encode_ms:.1f} ms   "
      f"2 questions x {len(shoot)} reads: {read_ms:.1f} ms")
```

```text
12 files
detail      4  1.png 4.png b1.png b5.png
packshot    7  0.png 2.png 3.png 5.png b2.png b3.png b4.png
rejects     1  b0.png

encode 12 images: 1.8 ms   2 questions x 12 reads: 1.8 ms
```

`embed(list)` hands the whole list to the encoder in one go; a loop of `ask()` calls
would hand it one image at a time and waste the GPU. Encode in bulk, then read each
embedding. The loop that follows does no model work at all, so it can do arbitrary
routing logic without thinking about cost.

**Watch out:** `Jev.embed()` has no size cap, so a directory of 50,000 images is 50,000
decoded bitmaps in RAM at once, so slice it into chunks of a few hundred. Going through a
server, `Remote.embed()` chunks for you at 8 images per request, which is the server's
own limit. `IMAGE_SUFFIXES` is deliberately short (`.jpg .jpeg .png .webp`); a `.gif` or
a `.tif` in the folder is skipped here and refused with a 415 by a server.

---

## 6. Train a head when the prompts plateau

**Problem:** zero-shot prompts get one class right and the other wrong, and you have a
folder of labelled examples.

```python
import jevimage
from toy import Toy

jev = jevimage.load(Toy(), heads_dir="heads")   # stand-in encoder; yours is jevimage.load()

zero_shot = {"colour": {"type": "choice", "criteria": {"red": "a red product",
                                                       "blue": "a blue product"}}}
for label, paths in jevimage.read_folder("ds").items():
    hits = sum(jev.ask(p, zero_shot)["colour"].choice == label for p in paths)
    print(f"zero-shot {label}: {hits}/{len(paths)}")

head = jev.train("colour", "ds", template="a {} product")   # same prompts as above
print("\n", head, sep="")
print("classes", head.classes, "counts", head.counts, "encoder", head.encoder)
print("held out", head.accuracy, "per class", head.per_class)

c = head.comparison
print(f"\n{'class':<8}{'n':>3}{'zero-shot':>11}{'trained':>9}")
for r in c["per_class"]:
    print(f"{r['label']:<8}{r['n']:>3}{r['zero_shot']:>11.3f}{r['trained']:>9.3f}")
o = c["overall"]
print(f"{'ALL':<8}{head.n:>3}{o['zero_shot']:>11.3f}{o['trained']:>9.3f}"
      f"   delta {o['delta']:+.3f}")
print(c["basis"])

a = jev.ask("ds/red/0.png", {"c": {"type": "head", "head": "colour"}})["c"]
print("\n", a, sep="")
```

```text
zero-shot blue: 4/6
zero-shot red: 1/6

<Head 'colour' 2 classes, 12 examples, 50.0% held out>
classes ['blue', 'red'] counts {'blue': 6, 'red': 6} encoder toy
held out 0.5 per class {'blue': 0.5, 'red': 0.5}

class     n  zero-shot  trained
blue      6      0.667    0.500
red       6      0.167    0.500
ALL      12      0.417    0.500   delta +0.083
trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example

<head 'red' p=0.986>
```

Read the table before you ship the head. Overall it won by 8 points, which looks like a
success, and it got there by taking `red` from 0.167 to 0.500 while pulling `blue` *down*
from 0.667 to 0.500. On this data you would keep the prompts for `blue` and the head for
`red`, or go and find more `blue` examples. That per-class split is the reason the
comparison exists; an overall delta alone would have hidden it.

**Watch out:** `accuracy` is stratified k-fold with `k = min(5, thinnest class)`, so six
examples per class means folds of one or two images and a number with enormous error
bars. 0.500 on 12 examples is a direction, not a measurement. The baseline uses
`"a photo of {}"` unless you pass `template=`, as this script does; comparing a tuned
head against an untuned prompt you would never have written is how a head looks better
than it is. Training on a server works the same way (`Remote.train`), but it has no
`template` parameter, so the baseline there is always the default one.

---

## 7. The same script against a local encoder and a server

**Problem:** develop against a server your team already runs, then move the encoder
in-process for a batch job, without touching the code in between.

```python
import os

import jevimage
from toy import Toy

QUESTIONS = {"colour": {"type": "choice", "criteria": {"red": "a red product",
                                                       "blue": "a blue product"}},
             "usable": {"type": "noul", "criteria": {"true": "a sharp, well exposed photo",
                                                     "false": "a blurred photo"}}}

def screen(jev, path):                       # knows nothing about where the encoder runs
    a = jev.ask(path, QUESTIONS)
    return a["colour"].choice, round(a["colour"].confidence, 4), round(a["usable"].noul, 4)

url = os.environ.get("JEVIMAGE_URL")
local = jevimage.load(Toy())                 # stand-in encoder; yours is jevimage.load()
remote = jevimage.connect(url, api_key=os.environ.get("JEVIMAGE_API_KEY"))

print("remote health:", remote.health())
print("local        :", screen(local, "ds/red/2.png"))
print("remote       :", screen(remote, "ds/red/2.png"))
assert screen(local, "ds/red/2.png") == screen(remote, "ds/red/2.png")

head = remote.train("colour", "ds")       # the images upload, the head stays there
print("\ntrained on the server:", head, head.encoder)
print("listed there:", "colour" in [h["name"] for h in remote.list_heads()])
print("head ask   :", remote.ask("ds/blue/0.png", {"c": {"type": "head",
                                                            "head": "colour"}})["c"])

e = remote.embed(["ds/red/2.png", "ds/blue/0.png"])
print("embeddings :", type(e).__name__, tuple(e.shape))

try:
    remote.ask_embedding(e[0], QUESTIONS)
except AttributeError as err:
    print("\nremote has no ask_embedding:", err)
try:
    local.health()
except AttributeError as err:
    print("local has no health        :", err)

print("deleted:", remote.delete_head("colour"), "again:", remote.delete_head("colour"))
```

```console
$ JEVIMAGE_URL=http://127.0.0.1:8123 python3 r7.py
```

```text
remote health: {'encoder': 'toy', 'dim': 16, 'device': 'cpu', 'primitives': ['choice', 'score', 'noul', 'head'], 'heads': 0, 'auth': 'open', 'version': '0.1.19'}
local        : ('blue', 0.4711, 0.9485)
remote       : ('blue', 0.4711, 0.9485)

trained on the server: <Head 'colour' 2 classes, 12 examples, 50.0% held out> toy
listed there: True
head ask   : <head 'blue' p=0.942>
embeddings : Tensor (2, 16)

remote has no ask_embedding: 'Remote' object has no attribute 'ask_embedding'
local has no health        : 'Jev' object has no attribute 'health'
deleted: True again: False
```

`screen()` takes whichever object it is handed. Training goes over the wire too: the
labelled images upload, the encoder that fits the head is the server's, and what comes
back is the same `Head` object with the same `comparison`; the weights never leave.
`delete_head` returns `True`/`False` on both sides rather than raising on the second
call, because that is what the local one does.

**Watch out:** two gaps are deliberate and both are visible above. `ask_embedding` is
local-only, because skipping the network is its entire point, and `health()` is
remote-only, because there is no server to report on locally. A third asymmetry is
quieter and this script does not show it: `Jev.train(template=)` and
`Jev.compare(template=)` have no remote equivalent. The numbers agree only because both
sides run the same encoder, so check `health()["encoder"]` before you assume a server's
answers are comparable with your local ones, and remember heads live where the encoder
is, so a head you trained locally is not on the server.

---

## 8. Tell which thing is which, not just what is there

This is the case where `choice` and `noul` quietly give you the wrong answer. Both read one
vector for the whole image, which records what is in the picture and loses the arrangement. So
"the red wire goes to the black terminal" and "the black wire goes to the red terminal" look
nearly identical to them -- same words, and the pooled vector has nothing left to separate.

You do not switch anything on. When your options are the same words in a different order,
`choice` and `noul` read the patch grid instead of the pooled vector, because that is the only
thing that can tell them apart. Same forward pass.

```python
import jevimage

jev = jevimage.load()          # any single siglip2-* or clip-* encoder

# Each pair is the SAME WORDS rearranged. That is what routes it to the grid.
CHECKS = {
    "wiring":   ("the red wire is on the black terminal",
                 "the black wire is on the red terminal"),
    "stacking": ("the blue crate is on top of the yellow crate",
                 "the yellow crate is on top of the blue crate"),
    "labels":   ("the label with the barcode is on the left box",
                 "the label with the barcode is on the right box"),
}

questions = {k: {"type": "noul", "criteria": {"true": t, "false": f}}
             for k, (t, f) in CHECKS.items()}

a = jev.ask("panel.jpg", questions)          # one encode, every check read off it

for name, ans in a.items():
    flag = "as described" if ans.noul > 0.5 else "the other way round"
    print(f"{name:9} {ans.noul:.3f}  {flag}   ({ans.basis})")
```

```
wiring    0.938  as described          (grid readout)
stacking  0.104  the other way round   (grid readout)
labels    0.771  as described          (grid readout)
```

`basis` is `grid readout` rather than `contrast pair`, which is how you confirm the routing
fired. Ask the same binding as a `choice` and you get a `choice` back with `readout: "grid"`.

### The limitation to design around

The router fires on a rearrangement and nothing else, because deciding it from the text alone
is what makes it free and what keeps it from damaging ordinary classification -- the grid is
*worse* on presence questions (SugarCrepe `replace_obj`: pooled 92.01, grid 75.24).

So a pair that differs in **which** words it contains reads pooled, even when it is about
arrangement:

```python
# reads pooled - "terracotta" and "black" are different words, not a rearrangement
{"true": "the plant is in a terracotta pot", "false": "the plant is in a black pot"}
```

On one measured photo pooled got that confidently wrong (0.104) where the grid would have got
it right (0.970). If your question is about arrangement, **phrase it as a swap** -- name both
things in both options and change only their order. That is not a workaround, it is the
clearest statement of the question:

```python
# reads the grid
{"true": "a terracotta pot holding a green plant",
 "false": "a green pot holding a terracotta plant"}
```

Measured on BiVLC `swap` -- 359 real photos whose wrong caption has the same words in a
different order -- the two-image protocol favours the grid, but on one image the pooled
reading wins (SugarCrepe swap n=911: pooled 70.25, this routing 68.50), which is why the
routing is off unless `JEVIMAGE_GRID=1`. On controlled
two-object images where the captions differ only in the relation, pooled sits at chance
(0.525 left/right, 0.470 above/below) and the grid scores 1.000 on both.

Attribute binding is the strong half. Relation wording on a cluttered photo is the weak half:
on a busy shelf "the plant is to the left of the camera" scores 0.466, below chance. On
controlled images with two separated objects the same question is exact, so the limit is
clutter and scale, not the idea.

### Two refusals you should expect

An ensemble has no single patch grid, and since the pooled vector is at chance on exactly this
question, the call is refused by name rather than quietly falling back to it:

```
jevimage: these options are the same words in a different order, which only the patch grid
can separate, and encoder 'siglip2-base-224+dfn5b-h-14-384' cannot expose one. Use a single
transformers encoder, not an ensemble.
```

And a stored embedding no longer records where anything was, so recipe 4's cache-and-ask-later
pattern does not extend to a question phrased as a swap:

```
jevimage: these options are the same words in a different order, so telling them apart needs
where things are in the picture, which a stored embedding no longer records. Call ask() with
the image instead.
```

Both are raised before the image is encoded, so neither costs you a forward pass.

## 9. The failures you will hit

**Problem:** knowing in advance what jevimage refuses to do, and what it says when it
refuses.

```python
import jevimage
from toy import Toy, ToyCalibrated

jev = jevimage.load(Toy(), heads_dir="heads")            # head 'colour' was fitted here
other = jevimage.load(ToyCalibrated(), heads_dir="heads")

# 1. a head is a matrix in one encoder's space, and says so
try:
    other.ask("ds/red/0.png", {"c": {"type": "head", "head": "colour"}})
except ValueError as e:
    print("1.", e)

# 2. a single-caption noul needs an encoder with logit_scale AND logit_bias
one_caption = {"q": {"type": "noul", "instructions": "a red product"}}
try:
    jev.ask("ds/red/0.png", one_caption)
except ValueError as e:
    print("2.", e)
print("   on a calibrated encoder:", other.ask("ds/red/0.png", one_caption)["q"])
print("   the pair form, anywhere:",
      jev.ask("ds/red/0.png", {"q": {"type": "noul",
                                        "criteria": {"true": "a red product",
                                                     "false": "a blue product"}}})["q"])

# 3. bad questions are refused - after the image encode, not instead of it
for bad in ({"q": {"type": "choice", "criteria": {"only": "one option"}}},
            {"q": {"type": "choice", "prompt": "a photo", "criteria": ["a", "b"]}},
            {"q": {"type": "guess", "criteria": ["a", "b"]}}):
    try:
        jev.ask("ds/red/0.png", bad)
    except ValueError as e:
        print("3.", e)

# 4. a server that is not there names itself and the fix
try:
    jevimage.connect("http://127.0.0.1:9", timeout=2).list_heads()
except jevimage.JevError as e:
    print("4.", e)

# 5. a server that is there, refusing the request, forwards its own words
remote = jevimage.connect("http://127.0.0.1:8123")
try:
    remote.ask("ds/red/0.png", {"c": {"type": "head", "head": "nope"}})
except jevimage.JevError as e:
    print("5.", e.status, "|", e.detail)
```

```text
1. head 'colour' was trained on encoder 'toy' but this instance runs 'toy-calibrated'. Load 'toy', or retrain the head here.
2. encoder 'toy' has no calibrated sigmoid, so a single-caption noul cannot produce a probability. Give a {"true": ..., "false": ...} criteria pair instead, which works on every encoder.
   on a calibrated encoder: <noul 1.000>
   the pair form, anywhere: <noul 0.283>
3. choice needs a criteria object of 2..255 options, e.g. {"indoor": "an indoor scene", "outdoor": "the outdoors"}
3. prompt 'a photo' has no '{}' placeholder, so every option would get the same caption; write e.g. 'a photo of {}'
3. unknown question type 'guess'; expected one of: choice, score, noul, head
4. cannot reach a jevimage server at http://127.0.0.1:9 ([Errno 111] Connection refused). Start one with `jevimage serve`, or pass the URL it is actually listening on.
5. 422 | unknown head 'nope'; trained heads: none trained yet
```

Five refusals, all of them things a working system will hit eventually. The head/encoder
check is the one that matters most: two encoders with the same `dim` would otherwise
produce plausible nonsense, so a head records the encoder that fitted it and refuses to
run under another. The single-caption `noul` needs both `logit_scale` and `logit_bias`:
SigLIP publishes both, CLIP neither, and an `Ensemble` averages its members' scale but
has no bias. The `{"true", "false"}` pair form works everywhere and is better
calibrated, so prefer it.

**Watch out:** catch `ValueError`. Locally that is what the engine raises; remotely
`JevError` is a subclass of it, carrying `.status` and the server's own `.detail`, so
one `except ValueError` covers both and an existing handler keeps working when a
`load()` line becomes a `connect()` line. A dead server and a rejected request are both
`JevError`; branch on `.status is None` if you want to retry only the former. Two
things `ask()` can raise are not refusals of the question and are outside `ValueError`:
a list passed as `image` is a `TypeError`, and a head file that is present but will not
load is a `KeyError` - see
[`ask() takes one image`](troubleshooting.md#ask-takes-one-image-got-a-list-of-2) and
[A head file that will not load](troubleshooting.md#a-head-file-that-will-not-load).

---

## Where to go next

- [Question types](questions.md): every field of every question and answer.
- [Training a head](training.md): what the comparison measures, and when to trust it.
- [Choosing an encoder](encoders.md): the registry, ensembles, and bringing your own.
- [Python reference](reference-python.md): the exact public surface.
- [HTTP reference](reference-http.md): the routes, if you are writing a client.
- [Install](install.md): which of the four installs you want.

---

[← HTTP reference](reference-http.md) · [Docs index](index.md) · [Troubleshooting →](troubleshooting.md)
