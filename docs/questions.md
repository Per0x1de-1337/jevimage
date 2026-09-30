---
title: "Question types"
description: "choice, score, noul and head - the exact request shape, answer shape and failure modes of each, plus the grid readout that choice and noul switch to on their own."
---

A question is a JSON object. You send one image and up to 64 of them; you get back one
`Answer` per question, each a probability distribution over things you named yourself.
Nothing is generated, so nothing has to be parsed: you compare a number to a threshold.

There are four types.

| type | you give | you get | works on |
|------|----------|---------|----------|
| `choice` | 2..255 named options | the winner, a distribution over your names, a confidence | every encoder |
| `score` | an ordered rubric of 2..10 levels | a float position on that rubric, plus the distribution and legend | every encoder |
| `noul` | one or two captions | a single probability in 0..1 | pair form: every encoder. single-caption form: SigLIP only |
| `head` | the name of a head you trained | the winner, a distribution over your trained classes, a confidence | the encoder that fitted the head |

`noul` is "no-or-yes level": the one number between no and yes.

`choice` and `noul` each have two readouts and you do not choose between them: when the
options you give are the same words in a different order, they are read off the patch grid
instead of the pooled vector. See [the grid readout](#the-grid-readout-same-words-different-order).

## How the examples here were run

There is no network for model weights in the environment these docs were written in, so
every example below runs a **deterministic toy encoder** in place of SigLIP: a 16-dim
hash of the input bytes. It produces real embeddings, so every shape, every field and
every error message on this page is real output from a real run. **The numbers mean
nothing.** The toy has no idea what "indoor" looks like. Swap `Toy()` for
`jevimage.load()` and the shapes are identical; only the numbers become meaningful.

```python
# toy.py - a stand-in encoder, no downloads. See encoders.md for the real ones.
import hashlib, torch, jevimage

class Toy(jevimage.Encoder):
    name, dim = "toy", 16
    logit_scale = logit_bias = None          # no calibrated sigmoid: see noul, below
    def __init__(self, seed=0):
        torch.manual_seed(seed); self.W = torch.randn(64, self.dim)
    def _e(self, xs):
        v = torch.stack([torch.tensor([b/255 for b in hashlib.sha512(str(x).encode()).digest()])
                         for x in xs])
        return torch.nn.functional.normalize(v @ self.W, dim=-1)
    def img(self, ims): return self._e([getattr(i, "tobytes", lambda: i)() for i in ims])
    def txt(self, ts): return self._e(list(ts))

class ToyCalibrated(Toy):                    # pretends to be SigLIP-shaped
    name = "toy-calibrated"
    logit_scale, logit_bias = 100.0, -10.0
```

Every example loads it at `temperature=0.5`. A real encoder's cosines live in a narrow
band and the default temperature is tuned for that; the toy's cosines are spread over
the whole range, so at the default every answer saturates to `1.000` / `0.000` and the
printed distributions stop being readable. See [Temperature](#temperature).

```python
import json

import jevimage
from toy import Toy, ToyCalibrated

jev = jevimage.load(Toy(), temperature=0.5)
img = "ds/red/0.png"
```

`ds/` is the 12-image fixture these pages share: flat 64x64 squares, `ds/red/0..5.png`
and `ds/blue/0..5.png`. Write it next to `toy.py` with
[`examples/make_fixtures.py`](https://github.com/Per0x1de-1337/jevimage/blob/main/examples/make_fixtures.py) before running anything
below. The toy hashes the image's raw pixels, so the numbers on this page are a function
of those exact squares - point `img` at a photo of your own and every shape stays the
same while every number moves.

## The shape every question takes

```json
{
  "type": "choice | score | noul | head",
  "criteria": "...",
  "prompt": "a photo of {}",
  "instructions": "...",
  "head": "my_head"
}
```

| field | required | default | meaning |
|-------|----------|---------|---------|
| `type` | no | `"choice"` | which of the four. Case-insensitive. |
| `criteria` | for `choice`, `score`, and the `noul` pair form | none | the options, levels or captions. Shape depends on `type`. |
| `prompt` | no | `"{}"` | a template each caption is substituted into. Must substitute exactly one `{}`: a template that cannot vary per option is refused. |
| `instructions` | only for single-caption `noul` | `""` | the caption for the sigmoid form. Also names a head if `head` is absent. **Ignored by `choice` and `score`.** `instruction` is accepted as a spelling of the same key. |
| `head` | for `head` | none | which trained head to run. `name` is accepted as a spelling of the same key. |

`instructions` on a `choice` or a `score` is silently dead weight; it is read but never
used, so a question phrased as instructions plus bare labels will not do what it looks
like it does:

```python
crit = {"indoor": "an indoor scene", "outdoor": "an outdoor scene"}
bare  = jev.ask(img, {"q": {"type": "choice", "criteria": crit}})["q"]
withi = jev.ask(img, {"q": {"type": "choice", "criteria": crit,
                            "instructions": "Is this picture taken inside or outside?"}})["q"]
print("without instructions:", bare["probabilities"])
print("with instructions   :", withi["probabilities"])
print("identical:", bare == withi)
```

```
without instructions: {'indoor': 0.438078, 'outdoor': 0.561922}
with instructions   : {'indoor': 0.438078, 'outdoor': 0.561922}
identical: True
```

Everything the model sees is in `criteria` and `prompt`. Put the discriminating words
there.

### Questions arrive as a dict or a list

```python
jev.ask(img, {"scene": {...}, "blur": {...}})     # -> {"scene": Answer, "blur": Answer}
jev.ask(img, [{...}, {...}])                      # -> {"q1": Answer, "q2": Answer}
jev.ask(img, [{"id": "scene", ...}])              # -> {"scene": Answer}
```

A list gets `q1`, `q2`, ... unless an entry carries its own `id`. Two entries with the
same id is an error, not a silent overwrite.

### Answers

An `Answer` is a `dict` subclass with attribute access, so `a.choice` and `a["choice"]`
are the same thing and `json.dumps(a)` needs no encoder. Printing one gives a short
repr, not the full record:

```python
a = jev.ask(img, {"scene": {"type": "choice",
                            "criteria": {"indoor": "an indoor scene",
                                         "outdoor": "an outdoor scene"}}})
print(a["scene"])
print(json.dumps(a["scene"], indent=2))
```

```
<choice 'outdoor' p=0.562>
{
  "type": "choice",
  "choice": "outdoor",
  "probabilities": {
    "indoor": 0.438078,
    "outdoor": 0.561922
  },
  "confidence": 0.123844
}
```

Reading a field that this answer type does not have raises `AttributeError` naming what
it does have. `a.score` on a choice answer says `'choice' answers have no 'score'; this
one has: type, choice, probabilities, confidence`.

---

## choice

Pick exactly one of 2..255 named options. Use it whenever the options are mutually
exclusive and you want the winner plus how sure the model was.

**Request**

```json
{"type": "choice",
 "criteria": {"indoor": "an indoor scene", "outdoor": "an outdoor scene"},
 "prompt": "a photo of {}"}
```

`criteria` is an object mapping **the label you want back** to **the caption the model
sees**. Your downstream code branches on `"indoor"`, and you can rewrite the caption to
something wordier without touching that code.

A bare list is shorthand for "label == caption":

```python
a = jev.ask(img, {"c": {"type": "choice", "criteria": ["a red square", "a blue square"]}})
print(json.dumps(a["c"], indent=2))
```

```json
{
  "type": "choice",
  "choice": "a red square",
  "probabilities": {
    "a red square": 0.574551,
    "a blue square": 0.425449
  },
  "confidence": 0.149102
}
```

Two identical captions in that list is an error rather than a distribution over fewer
options than you listed.

**Answer**

| field | type | meaning |
|-------|------|---------|
| `type` | str | `"choice"` |
| `choice` | str | the label with the highest probability |
| `probabilities` | `{label: float}` | sums to 1, rounded to 6 dp, in the order you listed |
| `confidence` | float | 0 = uniform, 1 = certain. See [Confidence](#confidence-and-thresholds). |

**Choosing a threshold.** Branch on `confidence`, not on `probabilities`. The maximum
probability is not comparable between a 2-option question and a 30-option one;
`confidence` is rescaled so it is. Start at `confidence >= 0.5` for "act on it" and
route the rest to a human, then move it once you have looked at a few hundred of your
own images. Also check `choice` against the runner-up when two options are near-
synonyms. The model splitting 0.45/0.44 between "bus" and "coach" is a caption problem,
not an uncertainty problem.

---

## score

Place the image on an **ordered** rubric of 2..10 levels. Use it when the answer is a
magnitude rather than a category: empty → packed, pristine → destroyed, clean → filthy.

**Request**

```json
{"type": "score",
 "criteria": ["an empty street", "a few people", "a busy street", "a packed crowd"],
 "prompt": "a photo of {}"}
```

`criteria` is an **array**, low end first. Order is meaningful: the score is the expected
index, so putting the levels out of order gives a number that averages across a rubric
that has no direction.

```python
a = jev.ask(img, {"crowd": {"type": "score", "prompt": "a photo of {}",
                            "criteria": ["an empty street", "a few people",
                                         "a busy street", "a packed crowd"]}})
print(a["crowd"])
print(json.dumps(a["crowd"], indent=2))
```

```
<score 1.30/3>
{
  "type": "score",
  "score": 1.2987,
  "probabilities": {
    "0": 0.315071,
    "1": 0.267825,
    "2": 0.220409,
    "3": 0.196695
  },
  "legend": {
    "0": "an empty street",
    "1": "a few people",
    "2": "a busy street",
    "3": "a packed crowd"
  },
  "confidence": 0.086761
}
```

**Answer**

| field | type | meaning |
|-------|------|---------|
| `type` | str | `"score"` |
| `score` | float | `sum(i * p_i)` - the expected level index, in `0 .. len(criteria)-1`, rounded to 4 dp |
| `probabilities` | `{str(index): float}` | keys are level **indices as strings**, not your captions |
| `legend` | `{str(index): str}` | index → the caption you gave. Keep it: the probabilities are unreadable without it |
| `confidence` | float | peakedness of the distribution, same scale as `choice` |

**Choosing a threshold.** `score` is a mean, so it hides its own shape: a bimodal
distribution over "empty" and "packed" lands mid-rubric and looks like "moderately
busy". Threshold on `score` **and** require `confidence` above a floor, or read
`probabilities` directly and sum the top levels (`p["2"] + p["3"] > 0.6`) when you need
"at least busy" rather than a point estimate. Note the scale ends at `len(criteria)-1`,
not at 1 or 100, so three levels means `0..2`.

---

## noul

One probability, no distribution to interpret. Use it for the yes/no gates that make up
most real pipelines: is this blurry, is there a person in frame, is this a screenshot.

`noul` has two forms, and they are not interchangeable.

### The contrast pair (works on every encoder)

```json
{"type": "noul", "criteria": {"true": "a blurry photo", "false": "a sharp, in-focus photo"}}
```

Two captions that mean opposite things is a real two-way decision, scored exactly like a
2-option `choice`, and `noul` is the probability of the `true` side. Needs no
calibration, so it works on SigLIP, CLIP, MetaCLIP, DFN5B and every ensemble.

```python
a = jev.ask(img, {"blur": {"type": "noul",
                           "criteria": {"true": "a blurry photo",
                                        "false": "a sharp, in-focus photo"}}})
print(a["blur"]); print(json.dumps(a["blur"]))
```

```
<noul 0.494>
{"type": "noul", "noul": 0.494006, "basis": "contrast pair"}
```

Write the `false` side as a real alternative, not as a negation. "a sharp, in-focus
photo" is a caption an image can match; "not a blurry photo" is not. Dual encoders have
no negation operator, and that caption embeds close to the one it was meant to oppose.

A pair with only one side filled in is an error, not a fallback to the sigmoid form.

### The single-caption sigmoid (SigLIP only)

```json
{"type": "noul", "instructions": "a blurry photo"}
```

One caption, and the probability comes from `sigmoid(cosine * logit_scale + logit_bias)`,
the encoder's own trained readout. This needs an encoder publishing **both**
`logit_scale` and `logit_bias`. SigLIP and SigLIP2 have both, because they were trained
with a pairwise sigmoid loss that had to output a calibrated probability per caption.
CLIP-family models publish a scale and no bias. An `Ensemble` publishes the mean of its
members' scales when every member has one (the concatenated cosine is the mean of the
members' cosines, so the mean scale is what reads it), but never a bias: a bias was
fitted against one member's own negatives and does not average across a concatenation.
So an ensemble still refuses the single-caption form, whatever its members are.

Asked of an encoder that cannot do it, you get a refusal rather than a plausible-looking
number:

```python
jev.ask(img, {"blur": {"type": "noul", "instructions": "a blurry photo"}})
```

```
ValueError: encoder 'toy' has no calibrated sigmoid, so a single-caption noul cannot
produce a probability. Give a {"true": ..., "false": ...} criteria pair instead, which
works on every encoder.
```

With a calibrated encoder the same question answers:

```python
cal = jevimage.load(ToyCalibrated())
a = cal.ask(img, {"blur": {"type": "noul", "instructions": "a blurry photo"}})
print(a["blur"]); print(json.dumps(a["blur"]))
```

```
<noul 1.000>
{"type": "noul", "noul": 1.0, "basis": "sigmoid readout"}
```

`1.000000` there is the toy being a hash function: its random cosines are far outside
the range SigLIP's scale and bias were fitted for, so the sigmoid saturates.

A real SigLIP does not saturate, but it does not give you a usable probability either.
Measured on `siglip2-base-224`, against a photo of a man using a laptop at an outdoor
picnic table:

```
0.198581  a person using a laptop
0.095418  a laptop
0.000172  a coffee cup
0.000051  a photo taken outdoors
0.000000  an elephant in space
```

The ordering is right, and the range is not. SigLIP's bias is `-16.75`, fitted for
retrieval against millions of negatives, so everything lands near zero: a threshold of
0.5 reads every caption as false, and two captions that are both true ("a person using a
laptop", "a photo taken outdoors") differ by a factor of 4000. Use the sigmoid form for
ranking captions against one image, and the `{true, false}` pair for anything you
threshold.

`basis` tells the two forms apart in the answer, so a consumer can log which readout
produced a number. **Temperature does not apply to the sigmoid form.** The scale is the
encoder's, not yours.

**Answer** (both forms)

| field | type | meaning |
|-------|------|---------|
| `type` | str | `"noul"` |
| `noul` | float | 0..1, rounded to 6 dp |
| `basis` | str | `"contrast pair"` or `"sigmoid readout"` |

There is no `confidence` on a `noul`: with one number, the distance from 0.5 is the
confidence.

**Choosing a threshold.** The pair form is a contrast, not an absolute: 0.5 means the
two captions fit equally well, which is not the same as "50% chance this is blurry".
Sweep the threshold on 50-100 of your own labelled images and pick the point that gives
you the precision you need. For a gate that drops images, err high (0.8+); for a gate
that flags for review, err low. The `criteria` pair is also the form to prefer if you
care about stability across encoder swaps, since the sigmoid form silently stops being
available the moment you move to an ensemble.

---

## head

Run a linear head you trained on your own labelled images. Same answer shape as
`choice`, plus provenance.

**Request**

```json
{"type": "head", "head": "colour"}
```

`head` names a head in the head directory. `name` works as a spelling of the same field,
and so does `instructions` if neither is present, a leftover convenience rather than the
shape to write.

There is no `criteria`, no `prompt`: the classes and their meanings are in the head
file, fixed when you trained it. See [training.md](training.md).

```python
head = jev.train("colour", "ds/")            # ds/red/*.png, ds/blue/*.png
print(repr(head))
a = jev.ask(img, {"c": {"type": "head", "head": "colour"}})
print(a["c"]); print(json.dumps(a["c"], indent=2))
```

```
<Head 'colour' 2 classes, 12 examples, 50.0% held out>
<head 'red' p=0.986>
{
  "type": "head",
  "head": "colour",
  "choice": "red",
  "probabilities": {
    "blue": 0.014136,
    "red": 0.985864
  },
  "confidence": 0.971727,
  "trained_on": 12,
  "accuracy": 0.5
}
```

**Answer**

| field | type | meaning |
|-------|------|---------|
| `type` | str | `"head"` |
| `head` | str | the head's name |
| `choice` | str | the highest-probability class |
| `probabilities` | `{class: float}` | over the head's classes, in the order the head stores them |
| `confidence` | float | same scale as `choice` |
| `trained_on` | int | how many examples fitted it |
| `accuracy` | float or null | **held-out** k-fold accuracy, `null` when a class was too thin to hold anything out |

`accuracy` rides along in every answer on purpose: a 0.99-confidence answer from a head
whose held-out accuracy is 0.62 is a confident answer from a bad classifier, and you can
only see that if both numbers are in front of you.

A head records the encoder that fitted it and refuses to run under a different one:

```
ValueError: head 'colour' was trained on encoder 'toy' but this instance runs
'toy-calibrated'. Load 'toy', or retrain the head here.
```

That check is on the encoder **name**, not the width, because two encoders of the same
dimensionality produce embeddings in unrelated spaces and the answer would be silently
wrong rather than loudly broken.

**Choosing a threshold.** Use `confidence`, as with `choice`, but calibrate it against
`accuracy` and `per_class` from [`compare()`](training.md) rather than against intuition.
A head trained on 12 examples is a suggestion; a head with 200 per class and 0.95
held-out is a decision.

---

## What a question cannot ask for

Everything above answers **what is present**. None of it answers **how things are
arranged**, and the difference is not obvious from the outside because the model is
equally confident either way.

Measured on one photo set:

| image | `"the man is wearing a hat"` vs `"the man is bare-headed"` | correct |
|---|---|---|
| a man wearing a hat | 0.993 | yes |
| a bare-headed man, no hat in frame | 0.092 | yes |
| a man **holding** a hat above his head | 1.000 | **no** |

The third row is the whole problem. The hat is in the picture, the pooled vector records
that, and it is not aligned to *where* the hat is. So a question phrased as presence gets
a presence answer, however you word it.

### The fix is `head`, not better wording

The arrangement is not destroyed by pooling, only unreachable from text. On 400 controlled
images of two objects with the sides swapped:

| how you ask | accuracy |
|---|---|
| in words, zero-shot | 50.00% (chance) |
| a linear head on the same embeddings | **80.83%** held out |

Identical vectors, both times. So if you need *on his head* rather than *in the picture*,
sort thirty examples into two folders and [train a head](training.md). The head reports
the accuracy it reached on examples it had not seen, so you get a number measured on your
data instead of trusting one measured on ours.

The same applies to *on* vs *under*, *plugged in* vs *unplugged*, *worn* vs *carried*.

## Captions: what the model reads

### The template

Each caption is substituted into `prompt` before it is embedded. The default is `"{}"`
(the caption, unchanged). The template must contain `{}`; without a placeholder every
option would get the same string, so the distribution would be uniform over identical
captions and `choice` would be whichever you listed first. That is refused:

```
ValueError: prompt 'a photo' has no '{}' placeholder, so every option would get the
same caption; write e.g. 'a photo of {}'
```

**Write captions, not questions.** Dual encoders are trained on image/caption pairs
scraped from the web. Their text tower has seen "a photo of a golden retriever on a
lawn" tens of millions of times and "Is there a dog in this image?" approximately never.
A question embeds near other questions, which is not where your image is. Phrase every
criterion as the caption someone would have written under the photo:

```
good:  "a photo of an empty supermarket aisle"
bad:   "Is the supermarket aisle empty?"
bad:   "empty"                                  (no context, embeds near anything)
```

Keep the shared words in `prompt` and the discriminating words in `criteria`: the shared
prefix contributes the same component to every option's embedding and cancels in the
softmax, while the differing part is what the answer is made of.

### Flattening: a criterion can be a string, a list or an object

Any criterion value is flattened to exactly one caption before embedding:

| value | becomes |
|-------|---------|
| `"a red square"` | `a red square` |
| `["a cracked screen", "a shattered display"]` | `a cracked screen, a shattered display` |
| `{"front": "an intact screen", "back": "an unmarked case"}` | `front: an intact screen; back: an unmarked case` |
| `null` (or an empty string) | the label itself |

Nesting composes: lists inside objects inside lists all collapse by the same two rules.
This is visible in the planner (internal, shown here only to make the flattening
concrete; `jevimage.core.plan` is not part of the public API):

```python
from jevimage.core import plan
print(plan({"type": "choice", "prompt": "a photo of {}", "criteria": {
    "damage": ["a cracked screen", "a shattered display"],
    "clean":  {"front": "an intact screen", "back": "an unmarked case"},
    "unknown": None}}))
```

```
('choice', ['a photo of a cracked screen, a shattered display',
            'a photo of front: an intact screen; back: an unmarked case',
            'a photo of unknown'],
 ['damage', 'clean', 'unknown'])
```

The object form injects your keys (`front:`, `back:`) into the caption, which is text the
encoder scores, useful when the keys are meaningful words and noise when they are
`a`/`b`/`c`. The `null` fallback means a criterion you forgot to fill in does not error;
it quietly becomes its own label. If your labels are already good captions, the bare-list
form says so more clearly.

---

## Confidence and thresholds

`confidence` is not the maximum probability. It is that maximum rescaled so **uniform is
0 and certain is 1** at any number of options:

```
confidence = (max(p) - 1/K) / (1 - 1/K)        K = number of options
```

Raw max-probability is not comparable across questions. 0.5 is lukewarm over two
options and emphatic over fifty. The rescaling removes K, so one threshold works for
every question you ask:

```python
from jevimage import confidence
for p, label in [([0.5, 0.5], "2 options, uniform"),
                 ([0.7, 0.3], "2 options, 0.70 max"),
                 ([0.9, 0.1], "2 options, 0.90 max"),
                 ([0.02]*50, "50 options, uniform"),
                 ([0.7] + [0.3/49]*49, "50 options, 0.70 max"),
                 ([1.0, 0.0], "certain")]:
    print(f"{label:22} max={max(p):.3f}  confidence={confidence(p):.3f}")
```

```
2 options, uniform     max=0.500  confidence=0.000
2 options, 0.70 max    max=0.700  confidence=0.400
2 options, 0.90 max    max=0.900  confidence=0.800
50 options, uniform    max=0.020  confidence=0.000
50 options, 0.70 max   max=0.700  confidence=0.694
certain                max=1.000  confidence=1.000
```

The same 0.70 maximum is 0.400 over two options and 0.694 over fifty, which is the
correct ordering: being 70% sure among fifty candidates is much stronger evidence than
being 70% sure between two.

`jevimage.confidence(p)` is public and takes any sequence of probabilities, so you can
apply the same scale to distributions you computed yourself. A single-element sequence
returns 1.0 by definition.

What confidence is **not**: a probability of being correct. It measures how peaked the
distribution is, and a badly-worded caption set can be sharply peaked on the wrong
option. Calibrate a threshold against your own labelled sample; do not port one from
another project's question set.

---

## Asking many questions at once

N questions about one image cost barely more than one. The image is encoded **once**;
each question is then a matmul against that vector. Captions are cached per string across
calls, so a repeated question set costs no text encoding at all after the first call.

Counting the encoder's forward passes directly (subclassing the toy to count):

```python
class Counting(Toy):
    img_calls = txt_calls = txt_strings = 0
    def img(self, ims):
        Counting.img_calls += 1
        return super().img(ims)
    def txt(self, ts):
        Counting.txt_calls += 1; Counting.txt_strings += len(ts)
        return super().txt(ts)

jev = jevimage.load(Counting(), temperature=0.5)
qs = {"scene":  {"type": "choice", "criteria": {"indoor": "an indoor scene",
                                                "outdoor": "an outdoor scene"}},
      "blur":   {"type": "noul", "criteria": {"true": "a blurry photo",
                                              "false": "a sharp photo"}},
      "crowd":  {"type": "score", "criteria": ["an empty street", "a busy street",
                                               "a packed crowd"]},
      "colour": {"type": "head", "head": "colour"}}

def report(when):
    print(f"{when}: {Counting.img_calls} image passes, {Counting.txt_calls} "
          f"text passes, {Counting.txt_strings} captions encoded")

jev.ask(img, {"scene": qs["scene"]}); report("after 1 question ")
jev.ask(img, qs);                     report("after 4 questions")
jev.ask(img, qs);                     report("asked again      ")
print("engine text cache:", jev.engine.stats)
```

```
after 1 question : 1 image passes, 1 text passes, 2 captions encoded
after 4 questions: 2 image passes, 2 text passes, 7 captions encoded
asked again      : 3 image passes, 2 text passes, 7 captions encoded
engine text cache: {'text_hits': 25, 'text_misses': 7}
```

The four-question call did **one** image pass and **one** text pass for the five new
captions (2 + 2 + 3 = 7 total, minus the 2 already cached). Asking the identical set again
encoded no text at all. The `head` question contributes zero captions; it is pure
matmul.

The ceiling is 64 questions per call. Above that, split the call and reuse the
embedding:

```python
emb = jev.embed(img)                 # the expensive part, once
a = jev.ask_embedding(emb, qs)       # local only - its purpose is skipping the network
b = jev.ask_embedding(emb, more_qs)
```

`ask_embedding` exists on `Jev` and not on `Remote`, deliberately: over HTTP there is
nothing to skip, so a remote caller sends the image and asks everything in one `/v1/ask`.

The text cache holds 50,000 captions, evicting oldest-first, a bound that matters only
to a long-lived server whose callers generate captions dynamically.

---

## Temperature

Temperature divides the logits before the softmax. Lower is sharper, higher is flatter.
It never changes which option wins, only how confident the answer looks.

```python
q = {"scene": {"type": "choice", "criteria": {"indoor": "an indoor scene",
                                              "outdoor": "an outdoor scene"}}}
for t in [None, 0.1, 0.5, 2.0]:
    a = jev.ask(img, q, temperature=t)["scene"]
    print(f"temperature={str(t):5} p={a['probabilities']} confidence={a['confidence']}")
```

```
temperature=None  p={'indoor': 0.438078, 'outdoor': 0.561922} confidence=0.123844
temperature=0.1   p={'indoor': 0.223597, 'outdoor': 0.776403} confidence=0.552807
temperature=0.5   p={'indoor': 0.438078, 'outdoor': 0.561922} confidence=0.123844
temperature=2.0   p={'indoor': 0.484445, 'outdoor': 0.515555} confidence=0.031111
```

(`None` here equals 0.5 because that is what this `Jev` was loaded with.)

**You almost certainly should not touch it.** The default is the encoder's own
`1 / logit_scale` (the temperature the model was trained to be read at), falling back to
`0.01` for encoders that publish no scale, which is the CLIP-family convention. A raw
cosine lives in roughly `[0, 0.3]` and softmaxes to nearly uniform, which is why some
scaling is needed at all and why the model's own is the right one.

Turning temperature down to make answers "more confident" does not make them more
correct; it makes your confidence threshold meaningless and hides exactly the
uncertainty you built a probability-first API to see. The honest reasons to set it:

- you are reproducing another system's calibration,
- you swapped in a custom encoder with no `logit_scale` and `0.01` is visibly wrong for
  its cosine range (as with the toy on this page),
- you are deliberately flattening a distribution to feed an ensemble vote downstream.

Where it applies:

| set on | applies to |
|--------|------------|
| `load(temperature=...)` | `choice`, `score`, `noul` pair. **Not** `head` (heads use 1.0), **not** the `noul` sigmoid. |
| `ask(..., temperature=...)` | the same three, **and** `head` - a per-call temperature overrides the head's 1.0. |
| neither | the encoder's `1 / logit_scale`, or `0.01` |

That split surprises people, so here it is measured on the `colour` head above:

```
load(temp=0.5): {'blue': 0.014136, 'red': 0.985864}
load(temp=5.0): {'blue': 0.014136, 'red': 0.985864}
ask(temp=0.5) : {'blue': 0.000206, 'red': 0.999794}
```

Must be a finite number greater than 0. Zero, negative, infinite, `True` and strings are
all refused: a negative temperature inverts every softmax so the *least* similar option
wins while still reporting high confidence, and an infinite one returns a perfectly
well-formed answer carrying no signal. Both are worse than an error, because nothing
downstream can tell.

---

## Over HTTP

The wire shape is the same object. `POST /v1/ask`:

```json
{"image": "data:image/png;base64,iVBORw0KGgo...",
 "questions": {"scene": {"type": "choice",
                         "criteria": {"indoor": "an indoor scene",
                                      "outdoor": "an outdoor scene"}},
               "blur":  {"type": "noul",
                         "criteria": {"true": "a blurry photo", "false": "a sharp photo"}}},
 "temperature": 0.5}
```

```json
{
    "answers": {
        "scene": {
            "type": "choice",
            "choice": "outdoor",
            "probabilities": {"indoor": 0.438078, "outdoor": 0.561922},
            "confidence": 0.123844
        },
        "blur": {"type": "noul", "noul": 0.492855, "basis": "contrast pair"}
    },
    "timing": {"embed_ms": 0.5, "total_ms": 0.9}
}
```

Those `scene` numbers are identical to the local run earlier in this page, against the
same encoder: `connect()` and `load()` answer the same question the same way. Send
`images` (a list, up to 8) instead of `image` and `answers` becomes a list, one entry per
image, and the questions are asked of every image. One field or the other: sending both is
a 422, `send 'image' or 'images', not both: 'image' returns one answer object, 'images'
returns a list`, because the choice picks the response shape as well as the pixels.

Every error in the table below arrives as HTTP 422 with the same sentence in `detail`,
except the two the engine never sees: `questions` that is neither an object nor a list,
and a `temperature` that is not a number, are both typed fields on the request model, so
FastAPI rejects them with its own field report instead. The unreadable head file is the
third exception: it is a 404, with the heads directory stripped out of the sentence.
(`temperature: 0` does reach the engine, but as a float, so the sentence reads
`got 0.0` over the wire and `got 0` in Python.)

```json
{"detail": "choice needs a criteria object of 2..255 options, e.g. {\"indoor\": \"an indoor scene\", \"outdoor\": \"the outdoors\"}"}
```

## From the CLI

`jevimage ask` builds these questions from flags; each flag repeats, and all of them ride the
one encode.

```bash
JEVIMAGE_URL=http://127.0.0.1:8123 jevimage ask ds/red/0.png \
  --choice indoor="an indoor scene" outdoor="an outdoor scene" \
  --score "an empty street,a busy street,a packed crowd" \
  --noul "a blurry photo" "a sharp photo"
```

```
ds/red/0.png   server http://127.0.0.1:8123

choice  [choice]  outdoor   confidence 1.000
    outdoor   1.000  ######################
    indoor    0.000

score  [score]  0.28 / 2   confidence 0.582
    an empty street   0.721  ################
    a busy street     0.279  ######
    a packed crowd    0.000

noul  [noul]  0.193   (contrast pair)
```

(The `choice` saturates because this server runs the toy encoder at its default
temperature, rather than the 0.5 the Python blocks above load it at; see above.) `--noul` takes two captions for the pair form or one for the sigmoid form.
Repeated flags get ids `choice`, `choice2`, ...; `--head NAME` uses the head's name as
its id. The CLI covers the common shapes only; `prompt` templates, nested criteria and
per-call temperature are Python and HTTP.

---

## The grid readout: same words, different order

`choice` and `noul` read one vector for the whole image. That vector records what is in the
picture and loses the arrangement, which is why it can tell you a helmet is present and not
whether the helmet is on anybody's head.

When the options you give are **the same words in a different order**, what is present cannot
separate them at all -- both readings contain a helmet and a head. So `choice` and `noul` stop
reading the pooled vector and read the patch grid instead. The encoder computes 196 patch
vectors on the way to the pooled one and throws them away; this asks each part of your caption
which patch it belongs to, then compares your options on that. Same forward pass, same 196
patches, nothing trained.

**There is no type for this and no flag.** It is decided from your `criteria`, because your
criteria are what make it necessary. You keep the answer shape you asked for.

```bash
jevimage ask shelf.jpg \
  --noul "a black camera beside a green plant" \
         "a green camera beside a black plant"
```

```
noul  [noul]  0.828   (grid readout, confidence 0.452)
```

```python
jev.ask(img, {"colours": {"type": "noul", "criteria": {
    "true":  "a black camera beside a green plant",
    "false": "a green camera beside a black plant"}}})
```
```
{"type": "noul", "noul": 0.828, "basis": "grid readout", "confidence": 0.452}
```

The same binding as a `choice` routes the same way and answers as a `choice`:

```python
jev.ask(img, {"colours": {"type": "choice", "criteria": {
    "camera_black": "a black camera beside a green plant",
    "plant_black":  "a green camera beside a black plant"}}})
```
```
{"type": "choice", "readout": "grid", "choice": "camera_black",
 "probabilities": {...}, "confidence": 0.452}
```

### How to tell which readout ran

| type | field | value |
|------|-------|-------|
| `noul` | `basis` | `grid readout` instead of `contrast pair` |
| `choice` | `readout` | `grid`; the key is absent on a pooled choice |

`score` never routes here. An ordinal rubric's levels differ in which words they contain --
"empty", "busy", "packed" -- so they are never a rearrangement of each other.

**Read the confidence relatively.** A grid-read answer runs lower than a pooled one, because
the grid scores sit in a narrower range. On a real photo, a binding question spread 0.0149
and reported confidence 0.452, while a relation question the readout genuinely could not
decide spread 0.0036 and reported 0.086. That flatness is a true report: rescaling the scores
to a unit spread turned the undecided case into 0.551 confidence on the **wrong** option, so
the scores are deliberately left unscaled. (The mass over the grid *is* standardised, because
a location always has an answer and only its sharpness was mis-scaled.)

### When it fires

Exactly when the options are word-rearrangements of each other, ignoring articles and other
stopwords, and never when two options are identical. That test is the whole routing decision
and it needs no image, so it happens in the planner before anything is encoded.

Measured over 2933 BiVLC instances it fires on 249, of which 248 are genuine swaps:
precision 0.996, nothing fitted.

It is deliberately narrow, because the grid is **worse** on presence questions -- it can find
some region that weakly matches almost any noun. Measured on SugarCrepe `replace_obj`, where
the wrong caption swaps in a different object: the pooled vector scores 92.01 and the grid
75.24. On `swap_att`, where the wrong caption is the same words rearranged, the pooled vector
scores 54.80 -- below chance -- and the grid 70.42. Each readout is used where it wins.

Measured on BiVLC `swap` (359 real photos whose negative caption has the same words in a
different order, so a bag-of-words reading is at chance):

On the task this library performs -- one image, two captions -- the pooled reading wins,
measured on SugarCrepe swap (n=911) against a whole-caption baseline: pooled **70.25**,
this routing 68.50. The larger figures once reported for this readout came from the
two-image benchmark protocol and from a baseline that scored a caption as the mean of its
segment embeddings, about 18 points weaker than encoding the caption. The readout is
therefore **off by default**; set `JEVIMAGE_GRID=1` to try it.

On controlled two-object images where the captions differ only in the relation, the pooled
vector scores 0.525 on left/right and 0.470 on above/below - chance is 0.500 - and the grid
scores 1.000 on both.

**Attribute binding is the strong half; relation wording is the weak half.** Asking which
property belongs to which object works well:

```
"a black camera beside a green plant"  vs  "a green camera beside a black plant"
     grid 0.824      pooled 0.566   (a coin flip)
```

Asking about left/right or above/below on a cluttered photo is much weaker -- on that same
image "the plant is to the left of the camera" scores 0.466, below chance. On controlled
images with two separated objects the same question is exact, so the limit is clutter and
scale, not the idea. Measured on VSR, a spatial-reasoning benchmark of real photographs:
left/right goes from 0.46 to 0.77, while "on" and "under" do not improve, because in a real
photograph those usually mean physical support rather than a position in the frame.

Note what this means for a pair that differs in **which** words it contains, like
`"the plant is in a terracotta pot"` against `"the plant is in a black pot"`. That is a
presence question, it reads pooled, and on one measured photo pooled got it confidently wrong
(0.104) where the grid would have got it right (0.970). The router is tuned for the case that
is common and decidable from the text alone; a pair like that is neither, and guessing at it
would cost more on ordinary classification than it wins here.

### Never write a negation

This is the mistake that produces a confident-looking number with nothing behind it.

```
"the girl is wearing pink dress"  vs  "the girl is not wearing pink dress"
```

That is not two alternatives. It is one caption with a word inserted, and dual encoders do not
represent "not": measured on a real photo those two captions are **0.927** similar to each
other, where two genuine alternatives sit at 0.83-0.85. There is nothing to separate, so the
answer is noise -- 0.652 on a photo where the dress is plainly white.

Name the **alternative** instead, never the absence:

```
"the girl is wearing a pink dress"  vs  "the girl is wearing a white dress"
     -> 0.095, correctly saying she is not in pink
```

Inserting "not" changes which words are present, so a negation pair is **not** a
rearrangement and reads pooled. That is exactly where the warning matters, because nothing
else about a `contrast pair` answer tells you the number is noise -- so every `noul` pair
carries a `warning` when it is unanswerable, whichever readout ran. It fires on two cases:

| case | what it says |
|------|--------------|
| the captions differ only by a negation word | dual encoders do not represent "not"; give two real alternatives |
| the captions are more than 0.90 similar | there is little to separate; differ in what they claim, not in wording |

The warning belongs to the pooled path. A rearrangement has the same words on both sides, so
it is near-identical to its swap by construction -- 0.975 measured on a controlled two-object
image -- and running the 0.90 similarity check on the grid path would fire on essentially
every question the readout exists for. High pooled similarity is the *precondition* for
reading the grid, not a reason to distrust it, so a grid-read answer always reports
`confidence` and never a warning.

### Requirements

The grid readout needs one transformers encoder so there is a patch grid to read. An ensemble
(`a+b`) has no single grid, and since the pooled vector is at chance on exactly this question,
the call is refused by name rather than quietly answering from it:

```
jevimage: these options are the same words in a different order, which only the patch grid
can separate, and encoder 'siglip2-base-224+dfn5b-h-14-384' cannot expose one. Use a single
transformers encoder (any siglip2-* or clip-* entry in `jevimage encoders`), not an ensemble.
```

It also needs the image itself. `ask_embedding` refuses it, because a stored embedding no
longer records where anything was:

```
jevimage: these options are the same words in a different order, so telling them apart needs
where things are in the picture, which a stored embedding no longer records. Call ask() with
the image instead.
```

Both are decided before the image is encoded, so a request that cannot be answered this way
fails without paying for a forward pass.

### How it works

The pooled vector is `out_proj(sum_i a_i * V x_i)` plus the pooling head's tail. Dropping
the attention weights `a_i` and keeping `V x_i` for each patch gives that patch's own
contribution, in the space the text already lives in.

Reading `last_hidden_state` directly does not work. Measured on 96 controlled images with
one object on a textured background, asking which patch a phrase peaks on:

| readout | correct patch | centroid error |
|---------|---------------|----------------|
| raw `last_hidden_state` | 12.5% | 2.68 patches |
| the value projection | **100%** | **0.43 patches** |

A random guess averages 5.3 patches of error, so the raw tokens are barely above noise.
That is why patch tokens have a reputation for being unusable.


## Errors a malformed question can raise

All of these are `ValueError` unless marked, and surface as HTTP 422 with the text below
(see the two exceptions above). The one `KeyError` is a 404 instead, and the server
strips the heads directory out of its text on the way to the client. Each message is
verbatim.

None of them is free: `ask()` encodes the image first and plans the questions
afterwards, so one bad question in a batch of 64 fails the whole call after paying for
the encode rather than instead of it. Most are raised by the planner, which is before the
*captions* are encoded. Seven are not: the single-caption `noul` refusal, the unreadable
head file, the four head checks and the `nan` guard are raised while the answer is being
read, after every caption in the request has been encoded too. The embedding-norm check
sits in between, after the image and before the text.

| what you did | error |
|--------------|-------|
| question is not an object | `a question must be an object, got str` |
| unknown `type` | `unknown question type 'boolean'; expected one of: choice, score, noul, head` |
| `prompt` without a placeholder | `prompt 'a photo' has no '{}' placeholder, so every option would get the same caption; write e.g. 'a photo of {}'` |
| `choice` with fewer than 2 or more than 255 options, or a non-object `criteria` | `choice needs a criteria object of 2..255 options, e.g. {"indoor": "an indoor scene", "outdoor": "the outdoors"}` |
| `choice` list form repeating a caption | `choice criteria repeats 'a red square'; each option needs its own caption` |
| `score` `criteria` not an array of 2..10 levels | `score needs an ordered criteria array of 2..10 levels, low end first, e.g. ["empty", "busy", "packed"]` |
| `head` with no name | `head needs a name: {"type": "head", "head": "my_head"}` |
| `head` naming an untrained head | `unknown head 'nope'; trained heads: colour` |
| `head` naming a head file that is there but will not load | **KeyError:** `junk.pt is present in <heads dir> but unreadable: <the loader's first line>. Delete it, or retrain the head.` (the only templated message here; over HTTP it is a 404 and the `in <heads dir>` is dropped) |
| `noul` pair with one side empty | `noul criteria has an empty 'false' caption; both sides of a pair need text` |
| `noul` with neither a pair nor instructions | `noul needs instructions, or a {"true": ..., "false": ...} criteria pair (the pair is better calibrated)` |
| single-caption `noul` on an uncalibrated encoder | `encoder 'toy' has no calibrated sigmoid, so a single-caption noul cannot produce a probability. Give a {"true": ..., "false": ...} criteria pair instead, which works on every encoder.` |
| two questions sharing an id | `two questions share the id 'a'; ids must be unique` |
| `questions` is neither dict nor list | **TypeError:** `questions must be a {id: question} object or a list of questions` |
| more than 64 questions | `at most 64 questions per request` |
| `temperature` 0 or negative | `temperature must be greater than 0, got 0` |
| `temperature` infinite | `temperature must be finite, got inf; an infinite temperature makes every option equally likely` |
| `temperature` not a number (`True` counts as not a number) | `temperature must be a number greater than 0, got 'hot'` |
| a head from another encoder | `head 'colour' was trained on encoder 'toy' but this instance runs 'toy-calibrated'. Load 'toy', or retrain the head here.` |
| a head whose weights no longer match the encoder width | `head 'colour' expects 32-dim embeddings, this encoder produces 16` |
| a head file with no encoder recorded | `head 'colour' records no encoder, so it cannot be checked against this one; retrain it` |
| a head file with more classes than weight columns | `head 'colour' lists 3 classes but its weights have 2 columns; the file is damaged, retrain it` |
| `ask_embedding` with a non-unit embedding | `embeddings must be L2-normalised, got norm 0; wrap your encoder's output in jevimage.encoders.unit()` |
| an answer that came out `nan` or `inf` | `<question id>.<field> is nan: the image could not be scored (its embedding is not finite)` |

The last two only reach you through a custom encoder: `jevimage.encoders.unit()` is what
the built-in encoders wrap their output in, and a non-finite answer is refused rather
than emitted because `json.dumps` turns `nan` into invalid JSON that fails three layers
downstream.

---

See also: [encoders.md](encoders.md) for what to ask these questions of, and
[training.md](training.md) for where `head` questions come from.

---

[← Quickstart](quickstart.md) · [Docs index](index.md) · [Training a head →](training.md)
