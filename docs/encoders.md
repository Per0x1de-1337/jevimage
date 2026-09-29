---
title: "Choosing an encoder"
description: "The registry, ensembling with '+', writing your own, and why a head belongs to the encoder that fitted it."
---

The encoder is the one part of jevimage you are meant to replace. Everything above it
(`choice`, `score`, `noul`, heads, training, the HTTP routes) works on unit-norm vectors
and never asks where they came from. So the encoder is what decides your accuracy, your
latency, your embedding width, your licence obligations, and whether a single-caption
`noul` is available at all.

There are four ways to get one:

```python
jevimage.load()                                   # the default, siglip2-base-224
jevimage.load("siglip2-so400m-384")               # another registry entry
jevimage.load("siglip2-base-224+dfn5b-h-14-384")  # an ensemble, '+' joins names
jevimage.load(MyEncoder())                        # your own object, no registration
```

> **How the examples on this page were run.** Every block below ran against a stand-in
> encoder over a folder of plain red and blue
> PNGs (`ds/red/*.png`, `ds/blue/*.png`), the twelve squares
> [`examples/make_fixtures.py`](https://github.com/Per0x1de-1337/jevimage/blob/main/examples/make_fixtures.py) writes. The stand-ins are `MeanColour` and `Grey`, both
> written out in full under [Writing your own encoder](#writing-your-own-encoder)
> (`Grey` exists only so the ensemble examples have a second member), and `Toy`, a
> deterministic 16-dimensional hash encoder of the kind the package's own self-tests use
> (`demo()` in [`jevimage/server.py`](https://github.com/Per0x1de-1337/jevimage/blob/main/jevimage/server.py)); `ToyCalibrated` is that
> same toy with `logit_scale` and `logit_bias` filled in, to exercise the calibrated
> path. The outputs are real and unedited. They show the plumbing, not SigLIP's accuracy:
> a 3-dimensional colour space saturates to `p=1.000`, which a real encoder does not. No
> throughput number on this page was measured, and none is quoted.

## The registry

`jevimage.available()` is the list; `jevimage encoders` prints it with the current default
marked. Both work in an API-only install, because listing encoders should not require the
stack that runs them:

```console
$ jevimage encoders
   clip-vit-l-14
   dfn5b-h-14-384
   metaclip2-huge
 * siglip2-base-224
   siglip2-base-384
   siglip2-giant-384
   siglip2-so400m-384

* default (siglip2-base-224). Join names with '+' for an ensemble, e.g. siglip2-base-224+dfn5b-h-14-384.
```

Set `$JEVIMAGE_ENCODER` and the marker follows it. That variable is what `jevimage.load(None)`
reads, so the listing and the loader cannot disagree:

```console
$ JEVIMAGE_ENCODER=siglip2-so400m-384 jevimage encoders
   clip-vit-l-14
   ...
 * siglip2-so400m-384

* $JEVIMAGE_ENCODER=siglip2-so400m-384. Join names with '+' for an ensemble, e.g. siglip2-base-224+dfn5b-h-14-384.
```

| name | checkpoint | input | family | sigmoid `noul` | needs |
|---|---|---|---|---|---|
| `siglip2-base-224` | `google/siglip2-base-patch16-224` | 224px | SigLIP2 base - the default | yes | `[local]` |
| `siglip2-base-384` | `google/siglip2-base-patch16-384` | 384px | same backbone, more pixels | yes | `[local]` |
| `siglip2-so400m-384` | `google/siglip2-so400m-patch16-384` | 384px | shape-optimised 400M vision tower | yes | `[local]` |
| `siglip2-giant-384` | `google/siglip2-giant-opt-patch16-384` | 384px | the largest entry | yes | `[local]` |
| `clip-vit-l-14` | `openai/clip-vit-large-patch14` | 224px | the original CLIP, as a baseline | no | `[local]` |
| `metaclip2-huge` | `facebook/metaclip-2-worldwide-huge-quickgelu` | 224px | CLIP recipe, worldwide data | no | `[local]` |
| `dfn5b-h-14-384` | `apple/DFN5B-CLIP-ViT-H-14-378` | 378px | Apple's data-filtered CLIP | no | `[openclip]` |

**Relative cost follows the checkpoint name, not this page.** A ViT spends its time on
image tokens, and the token count is `(resolution / patch)²`: 224/16 gives 196 tokens,
384/16 gives 576, so the same backbone at 384px does roughly three times the vision work
of the 224px one. Across rows, `base` \< `so400m` \< `giant` in the vision tower, and
`clip-vit-l-14` (ViT-L/14 at 224 → 256 tokens) sits below `metaclip2-huge` (ViT-H/14 at
224, so the same token count but a wider, deeper tower) and `dfn5b-h-14-384` (ViT-H/14 at
378 → 729 tokens). That is arithmetic off the names. It is
not a benchmark, and it says nothing about which one is more accurate on *your* images.
For that, label a hundred of them and read the per-class table from
[`train`](training.md).

**The "sigmoid `noul`" column is the checkpoint's own claim, not a setting.** `HFCLIP`
reads `logit_scale` and `logit_bias` off the model when it loads it, and leaves them
`None` when the model has no such parameter
([`jevimage/encoders.py`](https://github.com/Per0x1de-1337/jevimage/blob/main/jevimage/encoders.py)). SigLIP-family models were trained
with a pairwise sigmoid loss and carry both; CLIP and MetaCLIP carry a scale only; the
`OpenCLIP` wrapper sets `logit_bias = None` outright, because an open_clip contrastive
model has no pairwise bias to publish. A single-caption `noul` needs **both**, so it is a
SigLIP-family feature. The `siglip2-base-224` row is measured; the rest of the column is
read from the loader's rule and the models' training objective. One line confirms it on
yours:

```python
enc = jevimage.load("siglip2-base-224").encoder
print(enc.name, enc.dim, enc.logit_scale, enc.logit_bias)
```
```text
siglip2-base-224 768 112.66889953613281 -16.771724700927734
```

**Registry keys are identifiers, not descriptions.** A head records the encoder's `.name`,
and `load()` stamps the registry key onto whatever it loads, so that name *is* the key for
anything that came out of the registry. Renaming an entry would orphan every head trained
under it, so the rule in the source is *add, do not rename*. `dfn5b-h-14-384` is the
visible scar: Apple's checkpoint is 378px, and the repository was renamed from `-384` to
`-378`, but the key keeps the old spelling because heads in the field point at it.

A registry entry is a model *plus its handling*, not just a repository id. The SigLIP
entries tokenise captions to a fixed 64 tokens, because SigLIP was trained on fixed-length
captions and measurably degrades with dynamic padding; the CLIP and MetaCLIP entries want
the opposite and use dynamic padding up to 77. You never set that, but it is why pointing
`HFCLIP` at some new repository id means knowing which of the two it expects.

Those numbers are a ceiling, not an error: a caption longer than the entry's token limit
is **truncated to fit**, so the tail of a very long caption is not read. Criteria are
descriptions ("a red front door"), not paragraphs, so this is a limit you have to go
looking for - but if you are generating captions, that is where they stop mattering.

`available()` lists registry entries only. Ensembles are formed on demand and never
appear.

## What changes when you swap

Swapping the encoder is one argument, but five things move with it.

**1. The temperature moves.** A raw cosine lives in a narrow band and would softmax to
nearly uniform, so the engine divides by a temperature, and it takes that temperature
from the encoder: `1 / logit_scale` when the encoder publishes one, otherwise `0.01`, the
CLIP-family convention.

```python
class Scaled(Toy):
    name = 'toy-scaled'; logit_scale = 50.0; logit_bias = -12.0

print('logit_scale 50.0 ->', jevimage.load(Scaled()).engine.temperature)
print('no logit_scale  ->', jevimage.load(Toy()).engine.temperature)
```

```text
logit_scale 50.0 -> 0.02
no logit_scale  -> 0.01
```

The consequence: the same question over the same image gives differently *shaped*
distributions under two encoders, even when both pick the same option. If you threshold on
a probability, re-check the threshold after a swap. `load(..., temperature=...)` overrides
it when you want one number across encoders.

**2. `dim` moves, so cached embeddings die.** If you keep embeddings from `embed()` in a
vector store, they belong to one encoder. Nothing recomputes them for you.

**3. Trained heads stop.** See
[A head belongs to the encoder that fitted it](#a-head-belongs-to-the-encoder-that-fitted-it).

**4. Single-caption `noul` may vanish.** On an encoder without both calibration numbers it
raises rather than inventing a probability; the `{"true": ..., "false": ...}` pair form
works everywhere. See [Question types](questions.md).

**5. Device and dtype are the built-in wrappers' business.** `HFCLIP` and `OpenCLIP` pick
`$JEVIMAGE_DEVICE`, else CUDA if present, else CPU, and run bf16 on GPU and fp32 on CPU. A
custom encoder decides for itself; nothing in the framework will move your model.

## Ensembles: `'+'`

Any two or more registry names joined by `+` load as one encoder:

```python
jev = jevimage.load("siglip2-base-224+dfn5b-h-14-384")
```

### What it computes

`Ensemble` concatenates the members' embeddings and renormalises. Because each member is
already unit-norm, the cosine of the concatenation is exactly the **mean of the members'
cosines**. An ensemble is an average of opinions, not a longer vector with some arbitrary
weighting. That is worth checking rather than believing:

```python
from jevimage.encoders import Ensemble

a, b = Toy(0), Toy(1); b.name = 'toy-b'
e = Ensemble([a, b])
print(e.name, e.dim, e.logit_scale, e.logit_bias)

ims, txt = [Image.open('ds/red/0.png').convert('RGB')], ['a red photo', 'a blue photo']
ci = e.img(ims) @ e.txt(txt).T
mi = (a.img(ims) @ a.txt(txt).T + b.img(ims) @ b.txt(txt).T) / 2
print('ensemble cosines', ci.tolist())
print('mean of members ', mi.tolist())
print('max abs diff', float((ci - mi).abs().max()))
```

```text
toy+toy-b 32 None None
ensemble cosines [[0.7787474393844604, 0.7259311676025391]]
mean of members  [[0.7787473797798157, 0.7259311676025391]]
max abs diff 5.960464477539063e-08
```

Two consequences fall straight out of that. **Every member gets equal weight, whatever its
`dim`**. The members are normalised before concatenation, so a 1152-dimensional member
does not outvote a 768-dimensional one, and there is no knob to change that. And with
three members you get the mean of three cosines, not a vote: one averaged similarity that
the usual softmax reads, so a member that is confidently wrong drags the average rather
than being outvoted.

### What it costs, and what it loses

The latency is the **sum** of the members' latency; both models run on every image and on
every caption. There is no early exit. A two-member ensemble of large encoders is the
slowest configuration in this library.

It does lose the sigmoid, permanently and by design:

```python
m1, m2 = ToyCalibrated(0), ToyCalibrated(1); m2.name = 'toy-calibrated-b'
e = Ensemble([m1, m2])
print(e.name, 'logit_scale', e.logit_scale, 'logit_bias', e.logit_bias)
jev = jevimage.load(e)
print('temperature', jev.engine.temperature)
jev.ask('ds/red/0.png', {'q': {'type': 'noul', 'instructions': 'a red photo'}})
```

```text
toy-calibrated+toy-calibrated-b logit_scale 100.0 logit_bias None
temperature 0.01
ValueError: encoder 'toy-calibrated+toy-calibrated-b' has no calibrated sigmoid, so a single-caption noul cannot produce a probability. Give a {"true": ..., "false": ...} criteria pair instead, which works on every encoder.
```

Both members were calibrated and the ensemble still refuses the single-caption `noul`,
because `logit_bias` is `None` on every ensemble. That is correct: a bias fitted against
one model's own negatives says nothing about the average of two models' cosines, and
pretending otherwise would return a number that looks like a probability and is not one.

The *scale* does survive. `logit_scale` is the mean of the members' scales when every
member publishes one, and `None` (so the `0.01` CLIP-family fallback) as soon as one
does not. Both stand-ins here carry `100.0`, so the `0.01` printed above is `1 / 100`, the
ensemble's own temperature rather than the fallback; an ensemble of SigLIPs reads at a
SigLIP temperature.

**When it is worth it:** you have measured that one encoder is systematically wrong on a
slice of your images in a way the other is not, your latency budget has room for both, and
you do not need the single-caption `noul`. Otherwise prefer one better encoder, or a
trained head on the cheap one, which costs a matmul instead of a second forward pass.

### Ensemble names

The name is the joined string, and **order is part of it**:

```python
print(encoders.load('mean-colour+grey').name)   # -> mean-colour+grey
print(encoders.load('grey+mean-colour').name)   # -> grey+mean-colour
print(encoders.load(' mean-colour + grey ').name)   # -> mean-colour+grey  (spaces stripped)
```

Two ensembles of the same members in different orders are two different encoder names, so
a head fitted under one is refused under the other, even though the embeddings differ only
in column order. Pick an order and keep it.

A malformed name is rejected rather than repaired. A leading or trailing `+`, a doubled
`++`, a `+ +`, and a bare `+` all raise; none of them quietly collapses to a smaller
ensemble:

```python
encoders.load('mean-colour+')
```

```text
ValueError: malformed ensemble name 'mean-colour+': join two or more encoder names with single '+', e.g. siglip2-base-224+dfn5b-h-14-384
```

## Writing your own encoder

Nothing needs to subclass anything. An encoder is any object with four members:

| member | type | contract |
|---|---|---|
| `.name` | `str` | Stable identity. Every head fitted here records this string. |
| `.dim` | `int` | Width of what `img` and `txt` return. |
| `.img(images)` | `list[PIL.Image]` → `(n, dim)` | fp32, **L2-normalised**, one row per image, in order. |
| `.txt(texts)` | `list[str]` → `(n, dim)` | fp32, **L2-normalised**, one row per caption, in order. |

Both return `torch.Tensor`. Both are called with lists and must preserve order; the
engine zips rows back to labels positionally.

Two optional attributes, and only these two, make the single-caption `noul` available:

| member | type | contract |
|---|---|---|
| `.logit_scale` | `float` | Multiplies the cosine before the sigmoid. Also sets the engine's temperature to `1 / logit_scale`. |
| `.logit_bias` | `float` | Shifts it. |

Leave them off entirely, or set them to `None`, and the framework reads them as absent
(`getattr`, not attribute access - a duck-typed encoder that publishes neither must load,
not crash). Without them, a single-caption `noul` refuses:

```text
ValueError: encoder 'mean-colour' has no calibrated sigmoid, so a single-caption noul cannot produce a probability. Give a {"true": ..., "false": ...} criteria pair instead, which works on every encoder.
```

With them, the same question reads straight off the sigmoid:

```python
class MeanColourSigmoid(MeanColour):
    name = 'mean-colour-sigmoid'
    logit_scale, logit_bias = 4.0, -2.0

jev = jevimage.load(MeanColourSigmoid())
for p in ('red/0.png', 'blue/0.png'):
    a = jev.ask('ds/' + p, {'q': {'type': 'noul', 'instructions': 'a red photo'}})
    print(p, a['q'])
```

```text
red/0.png <noul 0.861>
blue/0.png <noul 0.238>
```

Those two numbers are wiring, not calibration: they were picked here to land the output
somewhere readable. Publish `logit_scale` and `logit_bias` **only** when they came from
fitting a sigmoid on this model's cosines. SigLIP has them because it was trained with
that loss. Two numbers chosen by taste give a confident, wrong probability, which is the
one thing a probability-first library must not do. The `{"true": ..., "false": ...}`
pair form needs no calibration and works on every encoder.

### A complete encoder

This is the whole thing. No weights, no download, and it answers colour questions:

```python
"""An encoder with no weights: images become their mean RGB direction, captions become
the direction of whatever colour word they mention."""
import torch
from PIL import Image

from jevimage.encoders import unit

PALETTE = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1),
           "yellow": (1, 1, 0), "cyan": (0, 1, 1), "magenta": (1, 0, 1)}


class MeanColour:
    name = "mean-colour"      # provenance: every head fitted here records this string
    dim = 3                   # width of what img() and txt() return
    # no logit_scale / logit_bias: this encoder was never fitted with a sigmoid loss,
    # so it does not claim a calibration it does not have

    def img(self, images):
        rows = [im.convert("RGB").resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
                for im in images]
        return unit(torch.tensor(rows, dtype=torch.float32))

    def txt(self, texts):
        rows = [[sum(c) for c in zip(*(PALETTE[w] for w in PALETTE if w in t.lower()))]
                or [1, 1, 1] for t in texts]
        return unit(torch.tensor(rows, dtype=torch.float32))


class Grey:               # a second stand-in, so this page has something to ensemble
    name = "grey"
    dim = 3

    def img(self, images):
        rows = [list(im.convert("L").resize((3, 1), Image.Resampling.BOX).getdata())
                for im in images]
        return unit(torch.tensor(rows, dtype=torch.float32))

    def txt(self, texts):
        rows = [[t.lower().count("dark"), len(t) % 3, t.lower().count("light")]
                for t in texts]
        return unit(torch.tensor(rows, dtype=torch.float32))
```

`jevimage.encoders.unit()` is the helper the built-in encoders use: it pulls the tensor out
of whatever a model returned, mean-pools a token-level `(n, tokens, dim)` output down to
one row per item, casts to fp32 and L2-normalises with a clamp so a zero row becomes zeros
rather than `nan`. Always fp32: bf16 has too few mantissa bits for a stable softmax, and a
probability that changes with the dtype is not a probability.

Hand the instance straight to `load()`, no registration needed:

```python
import jevimage
from mean_colour import MeanColour

jev = jevimage.load(MeanColour())
print(jev)
a = jev.ask('ds/red/0.png', {
    'colour': {'type': 'choice', 'criteria': {'red': 'a red photo', 'blue': 'a blue photo'}},
})
print(a['colour'])
print(a['colour'].probabilities)
```

```text
<Jev encoder='mean-colour' dim=3 heads=0>
<choice 'red' p=1.000>
{'red': 1.0, 'blue': 0.0}
```

That `1.000` is the stand-in, not the library: three dimensions and a hand-made palette
separate red from blue completely, and with no `logit_scale` the temperature is `0.01`,
which sharpens it further. A real encoder in 768 dimensions spreads the mass out.

### Registering it under a name

`register(name, factory)` adds it to the same registry the built-ins live in. The factory
is called with no arguments the first time `load(name)` asks for it, so a class is itself
a valid factory:

```python
jevimage.register('mean-colour', MeanColour)
jevimage.register('grey', Grey)
print(jevimage.available())
```

```text
['clip-vit-l-14', 'dfn5b-h-14-384', 'grey', 'mean-colour', 'metaclip2-huge', 'siglip2-base-224', 'siglip2-base-384', 'siglip2-giant-384', 'siglip2-so400m-384']
```

Registering buys three things: `load("name")` by string, membership in `'+'` ensembles,
and the key itself as the provenance string in a head record. It does **not** make the
name visible to the `jevimage` command, which never imports your code:

```console
$ jevimage ask ds/red/0.png --encoder mean-colour --choice red="a red photo" blue="a blue photo"
jevimage: unknown encoder 'mean-colour'. Available: clip-vit-l-14, dfn5b-h-14-384, metaclip2-huge, siglip2-base-224, siglip2-base-384, siglip2-giant-384, siglip2-so400m-384. Register your own with jevimage.register(name, factory) - from Python, in the process that will serve it; the `jevimage` command only sees these built-in names, because it never imports your code.
```

A registry name may not contain `+`, which is reserved for ensembles:

```python
jevimage.register('a+b', MeanColour)
```

```text
ValueError: '+' is reserved for ensembles, so a name cannot contain it
```

And the key you register under *becomes* the encoder's `.name`: `load()` stamps it onto
whatever the factory returned, so the string you load by and the string a head records are
always the same one.

```python
jevimage.register('colours', MeanColour)
print('colours', '->', encoders.load('colours').name)   # -> colours
```

The factory's own `.name` survives only when you skip the registry and hand an instance to
`load()` yourself. `MeanColour().name` is still `mean-colour` there. What that leaves you
to avoid is registering one model under two keys: those are two provenance strings, and a
head fitted under one is refused under the other. One key per model.

### A wrapper is an encoder too

The cheapest useful custom encoder is a wrapper around a model you already have, because
`.img()` and `.txt()` are the only two things the engine calls. Here is prompt ensembling,
the classic CLIP trick: each caption is embedded under several templates and averaged
before it ever meets an image.

```python
import torch, jevimage
from jevimage.encoders import load as load_encoder, unit


class PromptEnsemble(jevimage.Encoder):
    """Any encoder, with each caption averaged over several templates."""
    TEMPLATES = ["a photo of {}.", "a cropped photo of {}.",
                 "a blurry photo of {}.", "a close-up photo of {}."]

    def __init__(self, inner, name="toy-prompts"):
        self.inner, self.name, self.dim = inner, name, inner.dim
        self.logit_scale = self.logit_bias = None   # averaging breaks the trained scale

    def img(self, images):
        return self.inner.img(images)

    def txt(self, texts):
        return unit(torch.stack([
            self.inner.txt([t.format(x) for t in self.TEMPLATES]).mean(0)
            for x in texts]))


jevimage.register("toy-prompts", lambda: PromptEnsemble(load_encoder("toy")))

jev = jevimage.load("toy-prompts")
print(jev)
```

```text
<Jev encoder='toy-prompts' dim=16 heads=0>
```

That ran here over a deterministic stand-in registered as `toy`; substitute
`load_encoder("siglip2-base-224")` for the real thing. Note `logit_scale = logit_bias =
None`: averaging four text vectors invalidates the sigmoid the model was trained with, so
the wrapper drops the calibration rather than passing on a number that no longer means
what it says. Single-caption `noul` then refuses, which is the correct outcome.

Measured by the library's author on a 350-image, 7-class expression set (not re-run for
these docs), that wrapper lifted zero-shot accuracy from 22.3% to 32.3% on exactly the
same embeddings and the same base captions. It costs four text encodes per label instead
of one, and the engine caches caption vectors per string, so the cost is paid once.

You can also skip the registry and hand an instance straight to `load()`:

```python
jev = jevimage.load(PromptEnsemble(load_encoder("toy")))
```

### Serving your encoder

`jevimage serve` can only name built-in encoders, for the same reason `jevimage ask --encoder`
cannot: it never imports your code. Build the app yourself instead. `build_app(jev)`
takes a `Jev` you already made, and everything else about the server is unchanged:

```python
# serve_colour.py - uvicorn serve_colour:app --port 8140
import jevimage
from mean_colour import MeanColour
from jevimage.server import build_app

app = build_app(jevimage.load(MeanColour(), heads_dir="/var/lib/jev/heads", batch=True))
```

```console
$ python -m uvicorn serve_colour:app --port 8140
$ curl -s http://127.0.0.1:8140/v1/health
{"encoder":"mean-colour","dim":3,"device":"cpu","primitives":["choice", "score", "noul", "head"],"heads":0,"auth":"open","version":"0.1.19"}
```

Give it an explicit `heads_dir`. Without one the server publishes `~/.jev/heads` - whatever is in it, including heads trained under a different encoder, which it will then
refuse to answer with. The `"heads":0` above is an empty directory, not a guarantee.

The client side does not change at all, and does not need your encoder module, torch, or
any weights. This ran in a venv holding nothing but `jevimage` and pillow:

```python
import jevimage

jev = jevimage.connect('http://127.0.0.1:8140')
print(jev.health()['encoder'])
print(jev.ask('ds/blue/0.png',
              {'colour': {'type': 'choice',
                          'criteria': {'red': 'a red photo', 'blue': 'a blue photo'}}})['colour'])
```

```text
mean-colour
<choice 'blue' p=1.000>
```

`batch=True` is worth setting on a server and not in a script: it holds concurrent
requests for a few milliseconds so they ride one forward pass, and skips the wait
entirely while traffic is thin, so a lone caller pays nothing. It only pays off if your
encoder is cheaper per image in batches. A neural network is; `MeanColour` is not.
Pass `api_keys=[...]` to `build_app` to require a bearer token; see [Serving](serving.md).

### If your encoder returns the wrong thing

The engine checks the norm before it computes anything, because a rescaled embedding
saturates every softmax to exactly `1.0` and `0.0` - the one number this API must never
emit by accident:

```python
class Unnormalised(MeanColour):
    name = "unnormalised"
    def img(self, images):      # forgot to normalise
        return super().img(images) * 3.0

jevimage.load(Unnormalised()).ask(
    'ds/red/0.png',
    {'q': {'type': 'choice', 'criteria': {'red': 'a red photo', 'blue': 'a blue photo'}}})
```

```text
ValueError: embeddings must be L2-normalised, got norm 3; wrap your encoder's output in jevimage.encoders.unit()
```

The tolerance is 1e-2, so ordinary float error passes and a genuine scaling bug does not.
`txt()` is not norm-checked; an unnormalised text row silently reweights that one option
against the others, so normalise both sides.

Subclassing `jevimage.Encoder` gets you a readable `__repr__` and `logit_scale` /
`logit_bias` defaulting to `None`; duck-typing the four members is equally supported.

## A head belongs to the encoder that fitted it

A head is a matrix in one encoder's embedding space. Run it on another encoder's
embeddings and the answer is confident nonsense whenever the widths happen to match, so
the framework checks the encoder *name*, not just the shape, and refuses:

```python
import tempfile

d = tempfile.mkdtemp()                        # any head directory
jev = jevimage.load(MeanColour(), heads_dir=d)
h = jev.train('colour', 'ds/')
print(h)
print('encoder recorded:', h.encoder)

other = jevimage.load(Toy(), heads_dir=d)     # same head directory, different encoder
other.ask('ds/red/0.png', {'q': {'type': 'head', 'head': 'colour'}})
```

```text
<Head 'colour' 2 classes, 12 examples, 100.0% held out>
encoder recorded: mean-colour
ValueError: head 'colour' was trained on encoder 'mean-colour' but this instance runs 'toy'. Load 'mean-colour', or retrain the head here.
```

Four checks run, in this order, and each has its own message: that an encoder is
recorded at all, then that it is this one, then the embedding width, then the class count
against the weight columns. A head saved with **no** recorded encoder is refused outright
rather than assumed compatible. Provenance that cannot be checked is not provenance.

The same rule catches the quieter case: an ensemble is a distinct encoder, and one of its
own members is not a substitute for it.

```python
jev = jevimage.load(Ensemble([MeanColour(), Grey()]), heads_dir=d)
h = jev.train('colour', 'ds/')
print('head encoder:', h.encoder, '| dim:', h.to_dict()['dim'])

solo = jevimage.load(MeanColour(), heads_dir=d)
solo.ask('ds/red/0.png', {'q': {'type': 'head', 'head': 'colour'}})
```

```text
head encoder: mean-colour+grey | dim: 6
ValueError: head 'colour' was trained on encoder 'mean-colour+grey' but this instance runs 'mean-colour'. Load 'mean-colour+grey', or retrain the head here.
```

By the same string comparison, `b+a` is not `a+b` either, even though the two embedding
spaces differ only in column order.

**Migrating a head to a new encoder means retraining it.** There is no conversion. The
head file does keep the embeddings it was fitted on, but those are in the *old* encoder's
space, which is exactly what the new one cannot read, so you need the original labelled
images again. In practice:

1. Keep the old encoder loadable until the new heads exist. Two `Jev` instances can share
   one heads directory; only the mismatched *head* is refused, not the directory.
2. Retrain under the new encoder - `jevimage train NAME FOLDER --encoder NEW`, or `jev.train()`
   from a `Jev` loaded with it. Under a different head name if you want to A/B them.
3. Read the per-class comparison before you cut over. A bigger encoder is not automatically
   better on your classes; that is what [`compare()`](training.md) is for.
4. `jevimage rm OLD` when the new head wins.

Keeping labelled folders around is therefore part of using heads. Treat the head file as a
cache of a fit, not as the only copy of your data.

## Caching and instance reuse

`jevimage.encoders.load(name)` caches one instance per name, for the life of the process:

```python
a = encoders.load('mean-colour'); b = encoders.load('mean-colour')
print('cached:', a is b)
ens = encoders.load('mean-colour+grey')
print(ens, ens.dim, [m.name for m in ens.members])
print('member reused:', ens.members[0] is a)
```

```text
cached: True
<Ensemble mean-colour+grey dim=6> 6 ['mean-colour', 'grey']
member reused: True
```

What follows from that:

- **Weights load once.** Two `jevimage.load("siglip2-base-224")` calls in one process share
  one model. Two *processes* share nothing but the huggingface disk cache.
- **Ensemble members are the same objects** as the standalone entries, so holding
  `"a"` and `"a+b"` costs one copy of `a`, not two.
- **Nothing is ever evicted.** Loading four registry names in one process keeps four
  models resident. If you are probing encoders in a loop, do it in separate processes.
- **An instance you pass to `jevimage.load(MyEncoder())` is not cached at all** - it never
  went through the registry. You own its lifetime.
- **`register()` drops the cached instance for that name**, so re-registering during
  development takes effect on the next `load()`. It does not touch instances already
  handed out, and it does not retroactively fix an ensemble that already holds the old
  member.

The text side has its own cache, inside the engine rather than the encoder: every caption
is embedded once per `Jev` and reused across questions and requests, bounded at 50,000
strings, oldest evicted first. That is why a fixed question set gets cheaper after the
first call, and why `ask_embedding()` on an embedding you already hold does no image
encoding at all.

## Timings

Two sets of numbers, from two machines. Both are a shape, not a benchmark, and not a
prediction about yours.

Measured here, `siglip2-base-224` on CPU with 4 torch threads, each question a `choice`
with two fresh captions:

| What | Measured |
|---|---|
| encode one image | 110.65 ms |
| 1 question, captions not yet seen | 68.87 ms |
| 16 questions, captions not yet seen | 1020.11 ms (9.2x one encode) |
| 16 questions, captions cached | 0.39 ms (0.4% of one encode) |
| 1 question, captions cached | 0.05 ms |

That is the whole economics of this library in five rows, and the cold rows are the half
that gets skipped. A caption is a forward pass through the text tower the first time the
process sees it; after that it is a cached vector and the question is a matmul. So the
amortisation claim is about a **repeated** question set, not about questions in general.
Thirty-two fresh captions (16 two-option questions) cost more than nine image encodes;
the same sixteen questions asked again cost 0.4% of one.

The regime where this design loses is therefore: captions supplied per request by the
user, a new question set on every call, or a process that restarts often. None of those
ever warm the cache, and each pays the text tower every time. If your questions are fixed
and your images are not, you are in the regime the design is for.

Measured by the library's author on other hardware, and kept because nothing else records
them. The two question rows are read as captions-cached: 32 captions through a text tower
cannot come in under a 13.58 ms image encode.

| What | Measured |
|---|---|
| encode one image (`siglip2-base-224`, one RTX 3050 laptop GPU) | 13.58 ms |
| answer 1 question against that embedding, captions cached | 0.193 ms |
| answer 16 questions against that embedding, captions cached | 1.001 ms |
| encode one image, CPU, batch of 4, `siglip2-base-224` | 138 ms |
| encode one image, CPU, batch of 4, `siglip2-so400m-384` | 1731 ms |

The last two are the encoder-choice claim: a 12.5x spread between two rows of the registry
table, on the same CPU. Time the two or three encoders you are choosing between on your
own hardware; that is the only ratio that governs your bill.

## Failure modes on this page

| what you see | what it means |
|---|---|
| `unknown encoder 'x'. Available: ...` | Typo, or a name registered in a different process - including the `jevimage` CLI, which never imports your code. |
| `malformed ensemble name 'x'` | A leading or trailing `+`, a doubled `++`, a `+ +`, or a bare `+`. Spaces *around* names are stripped and are fine. |
| `'+' is reserved for ensembles` | `register()` name contains `+`. |
| `embeddings must be L2-normalised, got norm N` | Your `img()` did not normalise. Wrap the return in `jevimage.encoders.unit()`. |
| `encoder 'x' has no calibrated sigmoid` | Single-caption `noul` on an encoder without both `logit_scale` and `logit_bias` - including every ensemble. Use the `{"true": ..., "false": ...}` pair. |
| `head 'h' was trained on encoder 'a' but this instance runs 'b'` | Load `a`, or retrain `h` under `b`. No conversion exists. |
| `head 'h' records no encoder` | A head from before provenance existed, or a hand-built file. Retrain it. |
| `jevimage.load runs the encoder in this process, which needs torch` | API-only install. `pip install 'jevimage[local]'`, or use `jevimage.connect(url)`. |
| `No module named 'open_clip'` | `dfn5b-h-14-384` needs `pip install 'jevimage[openclip]'`, and its weights carry Apple's licence. |

---

Next: [Question types](questions.md) for what each encoder choice changes about `noul`,
[Training a head](training.md) for beating prompts without swapping models, and
[Serving](serving.md) for running one for other people. Source:
[`jevimage/encoders.py`](https://github.com/Per0x1de-1337/jevimage/blob/main/jevimage/encoders.py).

---

[← Training a head](training.md) · [Docs index](index.md) · [Serving →](serving.md)
