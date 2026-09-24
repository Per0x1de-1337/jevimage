---
title: "Troubleshooting"
description: "Every error you can hit, what causes it and how to fix it, organised by symptom."
---

Every error jevimage raises is a sentence, not a code. The sentence names the thing that
was wrong and usually names the fix, so the fastest route is to read it before reading
this page. This page is for when the sentence is right but the *reason* is not obvious.

It is organised by symptom (what you were doing when it happened), not by exception
class. Within each section: the exact message, why it exists, and what to do.

**About the examples.** Every message below was produced by running the code shown. Most
were produced against a deterministic toy encoder (a 16-dimensional hash, no weights, no
download) rather than SigLIP, because the error paths are in the framework, not the
model. Where the encoder itself matters (calibration, embedding width), that is said
explicitly and a real encoder was used. No number on this page is invented.

Two facts explain a large share of reported errors:

- `pip install jevimage` installs **no torch**. `jevimage.connect(url)` works fully;
  `jevimage.load()` does not. See [Install](/docs/install).
- A trained head belongs to **one encoder**. It records which one and refuses to run
  under another.

---

## Contents

- [Install and import](#install-and-import)
- [Loading an encoder](#loading-an-encoder)
- [The first run takes minutes](#the-first-run-takes-minutes)
- [Questions the engine refuses](#questions-the-engine-refuses)
- [`noul` specifically](#noul-specifically)
- [Temperature](#temperature)
- [Trained heads](#trained-heads)
- [Training refuses your data](#training-refuses-your-data)
- [Out of memory while training](#out-of-memory-while-training)
- [`nan` in an answer](#nan-in-an-answer)
- [Reading images](#reading-images)
- [Talking to a server](#talking-to-a-server)
- [Reading an `Answer`](#reading-an-answer)
- [HTTP status codes at a glance](#http-status-codes-at-a-glance)

---

## Install and import

### `jevimage.load runs the encoder in this process, which needs torch and transformers`

```text
ImportError: jevimage.load runs the encoder in this process, which needs torch and
transformers: pip install 'jevimage[local]'
To use a server instead - no torch required - call jevimage.connect(url).
(underlying import error: No module named 'torch')
```

**Why.** `pip install jevimage` is deliberately a few hundred kilobytes: pillow and the
standard library. `jevimage/__init__.py` resolves `load`, `Jev`, `Encoder`, `register`,
`available` and `DEFAULT` through a module-level `__getattr__`, so importing the package
never imports torch. Touching `load` is the moment the heavy half is needed, and that is
where the ImportError is raised, with both fixes named.

**Fix.** Pick the one that matches your situation.

```bash
pip install 'jevimage[local]'          # run the encoder here
```

```python
import jevimage
jev = jevimage.connect("http://your-host:8000")   # someone else runs it
```

`Remote` has the same method names, takes the same arguments and returns the same
`Answer` objects, including `train()`, which uploads the images and fits the head on the
server. The only missing method is `ask_embedding`, because its whole purpose is to skip
the network. One argument is local-only: `Jev.train` and `Jev.compare` take `template=`
to set the zero-shot baseline prompt (`"a photo of {}"` by default), and the wire format
carries no equivalent.

Confirming an API-only install is healthy, in a venv with no torch at all:

```python
import jevimage
jevimage.available()
jevimage.connect("http://127.0.0.1:8123").health()
```

```text
['clip-vit-l-14', 'dfn5b-h-14-384', 'metaclip2-huge', 'siglip2-base-224',
 'siglip2-base-384', 'siglip2-giant-384', 'siglip2-so400m-384']
{'encoder': 'toy', 'dim': 16, 'device': 'cpu',
 'primitives': ['choice', 'score', 'noul', 'head'], 'heads': 0,
 'auth': 'open', 'version': '0.1.0'}
```

`available()` works without torch on purpose: seeing which encoders exist should not
require the stack that runs them.

### `jev: this command runs the encoder locally, which needs torch`

```console
$ jev ask photo.png --choice red="a red square" blue="a blue square"
jev: this command runs the encoder locally, which needs torch:
       pip install 'jevimage[local]'
     or point it at a server that already has one:
       jev ask --url http://host:8000   (also $JEV_URL)
exit 1
```

**Why.** The CLI catches `ImportError`, checks for `torch`/`transformers` in the message
and prints both fixes, rather than letting `No module named 'torch'` escape; that
message does not tell you which of the two fixes you want. Exit status is 1, with no
traceback.

**Fix.** Either install the extra, or add `--url` / `$JEV_URL`:

```bash
jev ask photo.png --url http://127.0.0.1:8123 --choice red="a red square" blue="a blue square"
export JEV_URL=http://127.0.0.1:8123 JEV_API_KEY=s3cret   # same thing, once
```

Which commands can go remote: `ask`, `train`, `heads`, `rm`. `jev encoders` needs
nothing and always works. `jev serve` is local.

**`jev heads` with no `--url` also needs torch.** Not because of the files: the local
branch of `cmd_heads` imports `jevimage.training` for its one formatting helper, and
that module imports torch at module scope. So it fails with the message above even when
the head directory is empty or absent. `jev heads --url ...` never takes that branch, and
is the path that exists for API-only installs:

```console
$ jev heads --url http://127.0.0.1:8123
no heads in http://127.0.0.1:8123. Train one: jev train NAME FOLDER
```

### `jev serve needs uvicorn`

```console
$ jev serve
jev: jev serve needs uvicorn: pip install 'jevimage[serve]'
exit 1
```

**Fix.** `pip install 'jevimage[serve]'` (fastapi + uvicorn). See
[Serving](/docs/serving).

### `No module named 'open_clip'`

Only `dfn5b-h-14-384` needs it. `pip install 'jevimage[openclip]'`. Those are Apple's
DFN5B weights under Apple's own licence; read it before you redistribute anything built
on them. Every other registry entry runs on `transformers` alone.

---

## Loading an encoder

### `unknown encoder 'siglip-base'`

```python
jevimage.load("siglip-base")
```

```text
KeyError: "unknown encoder 'siglip-base'. Available: clip-vit-l-14, dfn5b-h-14-384,
metaclip2-huge, siglip2-base-224, siglip2-base-384, siglip2-giant-384,
siglip2-so400m-384. Register your own with jevimage.register(name, factory) - from
Python, in the process that will serve it; the `jev` command only sees these built-in
names, because it never imports your code."
```

**Why.** Registry names are exact. They are also stable identifiers: a trained head
stores the name of the encoder that fitted it, so an entry is never renamed, only added.

**Fix.** Copy a name from `jevimage.available()` or `jev encoders`. If it is your own
encoder, note the second half of that message: `register()` mutates a dict in *your*
process. The `jev` command never imports your code, so a name you registered in a script
is invisible to the CLI. To serve a custom encoder, build the app yourself:

```python
import jevimage, uvicorn
from jevimage.server import build_app

jev = jevimage.load(MyEncoder())          # or jevimage.load("mine") after register()
uvicorn.run(build_app(jev), host="127.0.0.1", port=8000)
```

### Registry and ensemble errors

| Message | Cause | Fix |
| --- | --- | --- |
| `'+' is reserved for ensembles, so a name cannot contain it` | `register("a+b", ...)` | Rename without `+` |
| `factory must be callable and take no arguments` | `register("x", MyEncoder())` — an instance, not a factory | `register("x", lambda: MyEncoder())` |
| `malformed ensemble name 'a++b': join two or more encoder names with single '+'` | An empty member: a leading `+`, a trailing `+`, a doubled `++`, or `+` alone | Name every member, one `+` between each: `load("a+b")` |
| `unknown encoder 'toyy'` from an ensemble | One member name is a typo; members are resolved individually | Fix that member |

Every empty-member spelling is refused, with the same sentence, before any member is
resolved, so a member name in the string is never even looked up:

```python
from jevimage import encoders

for bad in ("a++b", "+a", "a+", "a+ +b", "+"):
    try:
        encoders.load(bad)
    except ValueError as e:
        print(f"ValueError: {e}")
```

```text
ValueError: malformed ensemble name 'a++b': join two or more encoder names with single
'+', e.g. siglip2-base-224+dfn5b-h-14-384
ValueError: malformed ensemble name '+a': join two or more encoder names with single
'+', e.g. siglip2-base-224+dfn5b-h-14-384
ValueError: malformed ensemble name 'a+': join two or more encoder names with single
'+', e.g. siglip2-base-224+dfn5b-h-14-384
ValueError: malformed ensemble name 'a+ +b': join two or more encoder names with single
'+', e.g. siglip2-base-224+dfn5b-h-14-384
ValueError: malformed ensemble name '+': join two or more encoder names with single
'+', e.g. siglip2-base-224+dfn5b-h-14-384
```

In particular a doubled `+` is not a shorthand for a single one: `a++b` is refused, not
read as `a+b`. An encoder name is provenance (a head records the exact string), so a typo
that would have loaded under a *different* name than the one you wrote is refused rather
than accepted.

(`an ensemble needs at least 2 members` still exists, but only for
`jevimage.encoders.Ensemble([one])` built by hand; `load()` never reaches it.)

An ensemble's members each load fully, so a typo in the *second* member is only reported
after the first one has downloaded. Check names with `jev encoders` first.

### Custom encoders: `'Half' object has no attribute 'img'`

```python
class Half:
    name, dim = "half", 8
jevimage.load(Half()).ask("photo.png", {"q": {"type": "choice", "criteria": {"a": "x", "b": "y"}}})
```

```text
AttributeError: 'Half' object has no attribute 'img'
```

Subclassing `jevimage.Encoder` and forgetting the same method gives the base class's
stub instead:

```text
NotImplementedError:
```

**Why.** `load()` accepts any object as-is when it is not a string, which is how a custom
encoder gets in without registering it, and the contract is checked by use, not by an
`isinstance` gate.

**Fix.** An encoder needs exactly four things:

```python
name              # str, recorded in every head you train under it
dim               # int, embedding width
img(list[PIL])    # -> (n, dim) fp32, L2-normalised
txt(list[str])    # -> (n, dim) fp32, L2-normalised
```

`logit_scale` and `logit_bias` are optional; see [`noul` specifically](#noul-specifically).

### `embeddings must be L2-normalised`

```python
class RawEnc(Toy):
    name = "rawenc"
    def img(self, ims): return torch.ones(len(ims), self.dim) * 3

jevimage.load(RawEnc()).ask("photo.png", {"scene": {"type": "choice", "criteria": {"a": "x", "b": "y"}}})
```

```text
ValueError: embeddings must be L2-normalised, got norm 12; wrap your encoder's output
in jevimage.encoders.unit()
```

**Why.** Every logit in this framework is a cosine, and a dot product is only a cosine if
both vectors are unit-norm. A vector 12 times too long multiplies every logit by 12,
which saturates the softmax to exactly `1.0` and `0.0`, the one output a
probability-first API must never produce by accident. The check tolerates 1e-2, so
ordinary float error passes.

**Fix.** `jevimage.encoders.unit()` does the job on anything a model returns: a tensor, a
tuple, or a `transformers` output object with `image_embeds` / `text_embeds` /
`pooler_output` / `last_hidden_state`. It also mean-pools a 3-D token-level output and
casts to fp32.

```python
from jevimage.encoders import unit

def img(self, images):
    return unit(self.model.get_image_features(**self.proc(images=images, return_tensors="pt")))
```

The same error, reported as `norm 0`, means an all-zero embedding, usually an encoder
that silently failed rather than one that forgot to normalise.

---

## The first run takes minutes

`jevimage.load()` with no arguments loads `siglip2-base-224`. Nothing is bundled with the
package; weights come from the Hugging Face hub on first use and are cached by
`huggingface_hub` afterwards. The first call downloads hundreds of megabytes (several
gigabytes for `siglip2-giant-384`) with no jevimage-specific progress output, so a
process that looks hung on `load()` is usually downloading.

Loading `siglip2-base-224` from an already-populated cache, measured here:

```python
import time, jevimage
t = time.time(); jev = jevimage.load("siglip2-base-224"); print(f"{time.time()-t:.1f}s")
```

```text
9.7s      # 16-thread laptop CPU (i7-12650H), weights already in ~/.cache/huggingface
24.7s     # the same call, same box, under heavy load
```

Both are *cached* load time, with no download at all. The same call swings by 2-3x with
what else the machine is doing. Budget accordingly; this is not a per-request cost, but
it is a real startup cost.

**What to do about it.**

- Warm the cache before you need it (a Dockerfile layer, a CI step, an init container) so
  the first user request is not the first download.
- `HF_HOME` / `HF_HUB_CACHE` control where the cache lives. Point them at a volume that
  survives container restarts, or you re-download on every deploy.
- `HF_HUB_OFFLINE=1` makes a missing cache fail immediately instead of hanging on a slow
  network. Useful for proving the cache is warm.
- Set `JEV_ENCODER` so every process agrees which weights to warm.
- Instances are cached per name inside the process, so calling `load("siglip2-base-224")`
  twice loads the weights once.
- A server pays this once at startup. That is the strongest argument for
  `connect()` over `load()` in short-lived processes: a script that runs for two seconds
  should not spend another five to twenty-five loading a model.

`jevimage.connect(url)` downloads nothing, ever.

---

## Questions the engine refuses

A malformed question is not free: `ask()` encodes the image first and plans the questions
afterwards, so a bad question in a batch of 64 fails the whole call *after* paying for the
encode. (The CLI's own flag parsing is the exception, which happens before any weights
load.) Types and shapes are in [Question types](/docs/questions); this is the failure list.

| Message | Cause | Fix |
| --- | --- | --- |
| `a question must be an object, got str` | `{"q": "is it red"}` — a bare string where a question object belongs | `{"q": {"type": "choice", "criteria": {...}}}` |
| `unknown question type 'yesno'; expected one of: choice, score, noul, head` | A type that does not exist | Use one of the four |
| `choice needs a criteria object of 2..255 options, e.g. {"indoor": "an indoor scene", "outdoor": "the outdoors"}` | Fewer than 2 options, more than 255, or `criteria` is not a dict/list | Give at least two options |
| `choice criteria repeats 'a cat'; each option needs its own caption` | A bare list `["a cat", "a cat"]`; the list is keyed by caption, so duplicates would collapse | Make each caption distinct, or use the `{label: caption}` form |
| `score needs an ordered criteria array of 2..10 levels, low end first, e.g. ["empty", "busy", "packed"]` | `criteria` is a dict, or has fewer than 2 / more than 10 levels | Use an ordered list, lowest first |
| `prompt 'a photo' has no '{}' placeholder, so every option would get the same caption; write e.g. 'a photo of {}'` | A custom `prompt` template with no `{}` | Put `{}` where the option goes |
| `at most 64 questions per request` | More than 64 entries in one `ask()` | Split the call; the image is re-encoded, so cache the embedding with `embed()` + `ask_embedding()` |
| `two questions share the id 'a'; ids must be unique` | A list of questions where two carry the same `"id"` | Rename one; the dict of answers would otherwise return fewer answers than you asked |
| `questions must be a {id: question} object or a list of questions` | `ask(img, 7)` | Pass a dict or a list |

Note the asymmetry between `choice` and `score`: `choice` takes a **dict** (`{label:
caption}`) or a bare list, `score` takes an **ordered list** only. That is the single
most common shape mix-up. A dict passed to `score` gets the "ordered criteria array"
message.

A `choice` with 255 options is legal and cheap, because the captions are encoded once and
cached per string, so two questions sharing an option pay for it once.

---

## `noul` specifically

`noul` is the yes/no probability, and it has two forms. The difference is the source of
almost every `noul` error.

```python
# the pair form - works on EVERY encoder, better calibrated, prefer it
{"type": "noul", "criteria": {"true": "the lights are on", "false": "the lights are off"}}

# the single-caption form - needs an encoder with BOTH logit_scale and logit_bias
{"type": "noul", "instructions": "the lights are on"}
```

### `encoder 'toy' has no calibrated sigmoid`

```python
jev.ask("photo.png", {"q": {"type": "noul", "instructions": "it is dark"}})
```

```text
ValueError: encoder 'toy' has no calibrated sigmoid, so a single-caption noul cannot
produce a probability. Give a {"true": ..., "false": ...} criteria pair instead, which
works on every encoder.
```

**Why.** The single-caption form turns one cosine into P(yes) by `sigmoid(cos * scale +
bias)`. Those two constants are learned during pairwise training; without them there is no
honest mapping from a cosine to a probability, and inventing one would return a number
that looks like a probability and is not. So it raises instead.

**Who has them.**

| Encoder | `logit_scale` | `logit_bias` | Single-caption `noul` |
| --- | --- | --- | --- |
| SigLIP / SigLIP2 (`siglip2-*`) | yes | yes | works |
| CLIP-family (`clip-vit-l-14`, `metaclip2-huge`) | yes | **no** | raises |
| `dfn5b-h-14-384` (open_clip) | yes | **no** — set to `None` explicitly | raises |
| Any `Ensemble` (`a+b`) | the mean of its members', when every member has one | **no** | raises |
| A custom encoder that declares neither | no | no | raises |

Measured on the real default encoder:

```python
jev = jevimage.load("siglip2-base-224")
jev.encoder.logit_scale, jev.encoder.logit_bias, jev.encoder.dim
jev.ask("ds/red/0.png", {"q": {"type": "noul", "instructions": "a red square"}})
```

```text
logit_scale 112.0 logit_bias -16.75 dim 768
{'q': <noul 0.169>}
```

An ensemble refuses even when both members are SigLIP, and the reason is the bias alone.
The concatenated cosine is the *mean* of the members' cosines, so the mean of their
scales is the scale that reads it, and `Ensemble.logit_scale` is exactly that whenever
every member publishes one. The default temperature is derived from it. The bias does not
average: it was fitted against each member's own negative set, and there is no pair to
calibrate across a concatenation. So it stays `None`, and the single-caption form stays
refused:

```python
e = jevimage.encoders.load("toycal+toycal2")     # two toys, calibrated 100.0 and 120.0
print(f"ensemble: {e.name} dim {e.dim} logit_scale {e.logit_scale} logit_bias {e.logit_bias}")
jevimage.load(e).ask(img, {"q": {"type": "noul", "instructions": "a red square"}})
```

```text
ensemble: toycal+toycal2 dim 32 logit_scale 110.0 logit_bias None
ValueError: encoder 'toycal+toycal2' has no calibrated sigmoid, so a single-caption noul
cannot produce a probability. Give a {"true": ..., "false": ...} criteria pair instead,
which works on every encoder.
```

An ensemble of *uncalibrated* members still reports `logit_scale None`, because a mean of
nothing is nothing:

```text
ensemble: toy+toy2 dim 32 logit_scale None logit_bias None
```

**Fix.** Use the pair form. It is the better answer anyway. Two captions that mean
opposite things is a real two-way decision and reads as a contrast, needing no
calibration:

```python
{"type": "noul", "criteria": {"true": "it is dark", "false": "it is brightly lit"}}
```

Before writing single-caption `noul` against a server you do not run, check
`connect(url).health()` for the `encoder` field; there is no encoder object on the
client side to inspect.

### `noul criteria has an empty 'false' caption`

```python
{"type": "noul", "criteria": {"true": "outdoors", "false": ""}}
```

```text
ValueError: noul criteria has an empty 'false' caption; both sides of a pair need text
```

**Why.** A half-written pair would otherwise fall through to the single-caption form and
silently discard the side you did fill in, a wrong answer with no error.

**Fix.** Write both sides, or drop `criteria` entirely and use `instructions`.

### `noul needs instructions, or a {"true": ..., "false": ...} criteria pair`

`{"type": "noul"}` with neither. Supply one of the two.

---

## Temperature

```python
jev.ask(img, questions, temperature=0)
jev.ask(img, questions, temperature=-1)
jev.ask(img, questions, temperature=float("inf"))
jev.ask(img, questions, temperature="0.1")
jev.ask(img, questions, temperature=True)
```

```text
ValueError: temperature must be greater than 0, got 0
ValueError: temperature must be greater than 0, got -1
ValueError: temperature must be finite, got inf; an infinite temperature makes every option equally likely
ValueError: temperature must be a number greater than 0, got '0.1'
ValueError: temperature must be a number greater than 0, got True
```

**Why each case is refused, and not merely clamped:**

- **0** divides every logit by zero: `nan` or `inf`, no usable answer.
- **negative** inverts every softmax. The *least* similar option wins and the answer
  still reports high confidence. This is the dangerous one: it produces a perfectly
  well-formed, confidently wrong answer that nothing downstream can detect.
- **inf** divides every logit to zero, so every option comes back equally likely with
  confidence 0. A well-formed answer carrying no signal, which is worse than an error
  because it looks fine.
- **a string** is rejected rather than coerced; `"0.1"` from an unparsed config file
  should say so.
- **`True`** is an `int` in Python and would silently mean `temperature=1`. Booleans are
  excluded explicitly.

**Fix.** Pass a positive finite float, or leave it out. The default is the right one for
your encoder: `1 / logit_scale` when the encoder publishes one (so `1/112 ≈ 0.0089` for
`siglip2-base-224`), and `0.01` otherwise, the CLIP-family convention. A raw cosine
lives in roughly `[0, 0.3]` and softmaxes to nearly uniform, which is why the scale is
needed at all.

`Remote.ask` runs its own version of this check before sending, so `0`, a negative, a
string and `True` are all local `ValueError`s and never become a round trip, though its
wording is the `must be a number greater than 0` one in every case. It does not test for
`inf`, which travels and comes back as the engine's own message in a 422.

---

## Trained heads

A head is a small matrix that lives in one encoder's embedding space. Most head errors
are that identity being enforced.

### `unknown head 'nope'; trained heads: colours`

```python
jev.ask(img, {"q": {"type": "head", "head": "nope"}})
```

```text
ValueError: unknown head 'nope'; trained heads: colours
```

The message lists what is there (`none trained yet` when the directory is empty). Check
with `jev heads`, and check you are pointed at the right directory: `heads_dir=`, else
`$JEV_HEADS_DIR`, else `$JEV_HOME/heads`, else `~/.jev/heads`.

A head file that exists but *fails to load* gets its own message, not this one. See
[a head file that will not load](#a-head-file-that-will-not-load).

### `head 'colours' was trained on encoder 'toy' but this instance runs 'toy-calibrated'`

```python
other = jevimage.load(ToyCalibrated(), heads_dir="heads")   # a different encoder
other.ask(img, {"q": {"type": "head", "head": "colours"}})
```

```text
ValueError: head 'colours' was trained on encoder 'toy' but this instance runs
'toy-calibrated'. Load 'toy', or retrain the head here.
```

`compare()` enforces the same rule, for the same reason:

```text
ValueError: head 'colours' was trained on encoder 'toy' but this instance runs
'toy-calibrated'. Load 'toy', or retrain the head here.
```

**Why.** The head's weights are a map from *that* encoder's 768 (or 16, or 1152)
dimensions to your classes. Another encoder's embedding of the same image is a different
vector, so the matmul produces plausible-looking probabilities that mean nothing.
Whenever the widths happen to match, this would be silently wrong, which is why the
encoder **name** is checked and not just the shape.

**Fix.** Load the encoder named in the message, or retrain:

```python
jev = jevimage.load(Toy())                  # the encoder the head names
head = jev.train("colours", "ds/")          # or refit under the new one
```

(`Toy` is a class you hold, so it is loaded as an instance. A registry name - `jevimage.load("siglip2-base-224")` - is the other spelling; `jevimage.load("toy")` is neither, and raises `unknown encoder 'toy'`.)

This is also why changing `$JEV_ENCODER` on a server invalidates every head it holds.
Retrain them, or run a second server.

### `head 'colours' expects 16-dim embeddings, this encoder produces 32`

The names matched but the widths did not. Two different builds were sharing one `name`.
Names are identities here; give the new one its own name and retrain.

### `head 'nopro' records no encoder, so it cannot be checked`

```text
ValueError: head 'nopro' records no encoder, so it cannot be checked against this one;
retrain it
```

A head file predating the provenance field, or one hand-edited. An unchecked head runs
silently wrong wherever the widths happen to match, so it is refused rather than
assumed. Retrain it.

### `head 'damaged' lists 3 classes but its weights have 2 columns`

```text
ValueError: head 'damaged' lists 3 classes but its weights have 2 columns; the file is
damaged, retrain it
```

The class list and the weight matrix disagree. Zipping them would silently drop a class
or index out of range. The file is corrupt; retrain.

### `invalid head name '../evil'`

```text
ValueError: invalid head name '../evil': use 1-64 characters from [A-Za-z0-9._-],
starting with a letter or digit
```

Head names become filenames. They are **checked rather than sanitised** on purpose: a
silently rewritten name means the `ask()` that follows cannot find what `train()` said it
saved. The client validates the name before it builds a URL from it, so this is a local
`ValueError` even against a server, and you never see a confusing 404 for it.

### A head file that will not load

Drop a file that is not a valid head into the heads directory and the store records it as
broken rather than crashing:

```python
jev = jevimage.load(Toy(), heads_dir="heads")     # heads/junk.pt is not a torch file
sorted(jev.heads)
jev.heads.broken
```

```text
['colours']
{'junk': 'UnpicklingError: Weights only load failed. In PyTorch 2.6, we changed the
default value of the `weights_only` argument in `torch.load` from `False` to `True`.
Re-running `torch.'}
```

`jev heads` surfaces it on **stderr**, so it does not vanish from the listing:

```console
$ jev heads
junk.pt is present but unreadable: UnpicklingError: Weights only load failed. ...
head     encoder  classes   n  held out     created
colours      toy        2  12     50.0%  2026-09-23
```

Reaching for it, directly or through `ask()`, gives the same accurate message. The
store counts a file it could not read as *present*, so neither route claims the head is
unknown while it is sitting there on disk:

```python
jev.heads["junk"]
jev.ask(img, {"q": {"type": "head", "head": "junk"}})
```

```text
KeyError: 'junk.pt is present in heads but unreadable: UnpicklingError: Weights only
load failed. In PyTorch 2.6, we changed the default value of the `weights_only`
argument in `torch.load` from `False` to `True`. Re-running `torch.. Delete it, or
retrain the head.'
```

A `ValueError: unknown head ...` therefore means what it says: no such file. Compare:

```text
ValueError: unknown head 'nope'; trained heads: colours
```

**Fix.** Read the message: it names the file, the directory and the loader's own reason.
`jev heads` (stderr), `jev heads --json` (the `broken` key) and `jev.heads.broken` list
every such file at once. The usual causes:

- The file was written by a newer torch, or is not a torch file at all.
- A `.pt` from somewhere else. Heads load with `weights_only=True`, so a pickle
  containing anything beyond tensors and plain metadata is refused, deliberately: a head
  file from a colleague cannot execute code on the way in.
- A truncated copy. `save()` writes to `.pt.tmp` and `os.replace`s it, so jevimage's own
  writes are atomic; a truncated file came from something else copying it.

Retrain, or delete the file.

### `compare()` returns `None`

Not an error. A head keeps the embeddings it was fitted on so the trained-vs-zero-shot
comparison stays answerable later. A head saved without them cannot be compared, and
`compare()` says `None` rather than inventing a baseline. The server says `409` with the
same reason, and `Remote.compare()` translates that back to `None` so both sides behave
identically:

```text
{"detail":"head 'noemb' was saved without its embeddings, so it cannot be compared;
retrain it to restore this"}   [HTTP 409]
```

Retrain to restore it.

---

## Training refuses your data

| Message | Cause | Fix |
| --- | --- | --- |
| `need at least 2 distinct labels` | One class | A classifier needs something to discriminate against |
| `every class needs at least 2 examples; too few for: red` | A class with one image | Add examples. Two is the floor, not a target — `accuracy` from a 2-example class is a coin toss |
| `labels ['red', 'red '] differ only by surrounding whitespace, so they would print identically; make them distinct` | A stray space in a folder name or label | Rename. They would print identically in every table, so one class would appear to score half of what it should for no visible reason |
| `at most 100 classes` | More than `MAX_CLASSES` | Group them, or fit several heads |
| `at most 4000 examples; got 4001` | More than `MAX_EXAMPLES` | `max_per_class=`, or several heads |
| `no labelled images under 'ds/red'. Expected one sub-folder per label, each holding .jpeg/.jpg/.png/.webp files.` | Images sit loose in the folder, or in an unsupported format | One sub-folder per label. Loose files are ignored because they have no label |
| `'ds/red/0.png' is not a directory` | A file passed where a folder was expected | Pass the parent |
| `a list of examples must hold (image, label) pairs` | A 3-tuple, or a flat list | `[(image, label), ...]` |
| `train() takes a folder path, a {label: [images]} mapping, or a list of (image, label) pairs` | Something else entirely | One of the three shapes |

The suffix list is short on purpose: `.jpg`, `.jpeg`, `.png`, `.webp`. An animated GIF,
an SVG or a RAW file is not something to silently take the first frame of. Convert them
first if you need them.

**"Training did not help."** That is a result, not an error. The CLI says so:

```text
Training did not beat the prompts here - try more examples per class, or a bigger encoder.
```

The comparison exists because "will training help?" has no general answer. Read
`head.comparison["per_class"]`; training often helps some classes and hurts others, and
the overall number hides that.

**`accuracy` is `None`.** k-fold uses `k = min(5, thinnest class)`, and refuses to
report at `k < 2`, because a number computed on a fold that was missing a class is worse
than no number. The two-per-class floor above means `train()` never lands there, so a
`None` you see came from a head record that carries no `cv_accuracy` at all, one built or
edited by hand. `jev heads` prints `-` for it, not `0.0%`.

---

## Out of memory while training

There is no `OutOfMemoryError` with a helpful message here; you get CUDA's or the OOM
killer's.

`Jev.train()` embeds **every image in one call** to the encoder. Measured with an encoder
that counts its own batch sizes:

```python
seen = []
class Watch(Toy):
    name = "watch"
    def img(self, ims):
        seen.append(len(ims)); return super().img(ims)

j = jevimage.load(Watch(), heads_dir="heads")
j.train("w", "ds")                        # 12 images
j.train("w2", "ds", max_per_class=2)
```

```text
img() batch sizes, train on 12 images: [12]
img() batch sizes, max_per_class=2: [4]
```

One call, every image. For a 224px encoder the preprocessed pixel tensor alone is
`n × 3 × 224 × 224 × 4` bytes, about 600 KB per image, so 4000 images is ~2.4 GB of
pixels before the model has done anything. At 384px it is roughly three times that.

**Fixes, cheapest first.**

```python
jev.train("street", "ds/", max_per_class=200)          # cap the images
```

```python
# embed in chunks yourself, then fit on the embeddings
import torch, jevimage
from jevimage import as_examples
from jevimage import training                            # module, not in __all__

images, labels = as_examples("ds/")
embs = torch.cat([jev.embed(images[i:i + 32]) for i in range(0, len(images), 32)])
head = training.train(embs, labels)
head["encoder"] = jev.encoder.name
jev.heads.save("street", head)
print(jev.head("street"))
print(jev.ask("ds/red/0.png", {"q": {"type": "head", "head": "street"}}))
```

```text
<Head 'street' 2 classes, 12 examples, 50.0% held out>
{'q': <head 'red' p=0.986>}
```

That second form bypasses `Jev.train`, so remember `head["encoder"]`: without it the
head fails the provenance check with `records no encoder`.

Other levers: `JEV_DEVICE=cpu` trades speed for system RAM, which is usually far larger
than GPU RAM; a 224px encoder over a 384px one roughly thirds the pixel memory; and
training on a server (`connect(url).train(...)`) moves the problem to a machine sized for
it, though the images then cross the wire, bounded by the server's own per-image and
per-request limits.

OOM at *query* time is a different thing: `ask()` embeds one image. If that OOMs, the
encoder is too large for the device, not the batch.

---

## `nan` in an answer

You will not get a `nan` back. You get this instead:

```python
jev.ask_embedding(torch.full((1, 16), float("nan")), {"scene": {"type": "choice", "criteria": {"a": "x", "b": "y"}}})
```

```text
ValueError: scene.probabilities.a is nan: the image could not be scored (its
embedding is not finite)
```

The path is named, so you know which question and which field. It works the same for the
other types:

```text
ValueError: busy.score is nan: the image could not be scored (its embedding is not finite)
ValueError: h.probabilities.blue is nan: the image could not be scored (its embedding is not finite)
```

**Why it is an error and not a value.** `json.dumps` turns `nan` into invalid JSON
(literal `NaN`, which strict parsers reject), and a caller thresholding a probability
turns it into an unexplained crash three layers away. Refusing at the source, with the
field named, turns that into something you can act on.

**What causes it.** Not temperature, which is validated separately and cannot reach
here. In practice it is the encoder emitting `nan`:

```python
class NanEnc(Toy):
    name = "nanenc"
    def img(self, ims): return torch.full((len(ims), self.dim), float("nan"))

jevimage.load(NanEnc()).ask("photo.png", {"scene": {"type": "choice", "criteria": {"a": "x", "b": "y"}}})
```

```text
ValueError: scene.probabilities.a is nan: the image could not be scored (its
embedding is not finite)
```

Real causes, in rough order of likelihood:

- **fp16 overflow.** A model forced to fp16 on a GPU can produce `inf` in the forward
  pass, which becomes `nan` after normalisation. `unit()` casts to fp32 and the
  framework's own dtype choice is bf16 on CUDA and fp32 on CPU for exactly this reason:
  bf16 has too few mantissa bits for a stable softmax, and a probability that flips with
  the dtype is not a probability.
- **A custom encoder dividing by a zero norm.** `unit()` clamps at `1e-12`; hand-rolled
  normalisation often does not.
- **A degenerate image.** A zero-byte or entirely blank input reaching a model that
  produces a zero vector.

Note the ordering of the two guards: an all-zero embedding is caught earlier by the
L2-norm check (`got norm 0`), because `0` compares cleanly against `1`. A `nan` norm
compares `False` against every threshold, slips past that check, and is caught here.
Neither guard catches both cases, which is why both exist.

---

## Reading images

```text
FileNotFoundError: no such image file: 'nope.png'. Pass a path that exists, raw image
bytes, a PIL.Image, or a 'data:image/...;base64,...' string.

TypeError: cannot read an image from int; pass a path, bytes, a PIL.Image, or a data URL
```

**Why the first is not a guess.** A plain string that is neither an existing path nor a
data URL raises rather than being tried as bare base64. A typo'd filename should say so,
not fail three layers down inside a decoder.

**What `to_image` accepts:** `Path` or `str` (an existing file), `bytes` (encoded image
bytes), a `PIL.Image`, or a `data:image/...;base64,...` string. Anything else is a
`TypeError`.

A JPEG, PNG or WebP travels to a server as the original bytes; anything else is
re-encoded to PNG first. Portrait phone photos are `exif_transpose`d before encoding, on
both the local and the server path, so an image with `Orientation=6` is not scored
sideways, and local and remote answers stay identical.

`embed() needs at least one image` means you passed `[]`.

### `ask() takes one image, got a list of 2`

```python
jev.ask([img, img], questions)
```

```text
TypeError: ask() takes one image, got a list of 2. Loop over them, or call embed() once
and ask_embedding() per row.
```

**Why.** `ask()` answers about one image. A list used to be encoded in full and then
answered for its first element only, paying the whole cost and discarding the rest
silently, so it is refused instead.

**Both sides refuse the same way.** `Remote.ask` raises the same `TypeError`, with the
same first sentence, before anything goes over the wire; it is not a server-side shape
error and not a different exception. The suggested fix differs, because `ask_embedding`
is local-only: over HTTP the sentence ends `Loop over them, one ask() each.`

**Fix.** Loop, or, locally where `ask_embedding` exists, encode once and read the rows:

```python
embs = jev.embed([img, img])                   # one batched pass; embed() takes a list
[jev.ask_embedding(e, questions)["q"] for e in embs]
```

```text
[<choice 'a' p=0.978>, <choice 'a' p=0.978>]
```

On a `Remote` there is no `ask_embedding`, because that method's whole purpose is to skip
the network, so loop over `ask()` there.

---

## Talking to a server

`JevError` is the client's exception. It subclasses `ValueError` deliberately, so an
existing `except ValueError` keeps working when a `load()` line becomes a `connect()`
line. It carries `.status` (the HTTP code) and `.detail` (the server's own sentence). The
detail is the useful part:

```python
from jevimage import JevError

try:
    jev.ask(img, questions)
except JevError as e:
    print(e.status, e.detail)
```

### `url '127.0.0.1:8123' has no scheme`

```text
ValueError: url '127.0.0.1:8123' has no scheme; use 'http://127.0.0.1:8123'
```

Raised by `connect()` before any request. urllib's own message for this is `unknown url
type: 127.0.0.1`, which points at the port and sends you looking in the wrong place.
Prefix the scheme.

### Connection refused, or unknown host

```text
JevError: cannot reach jev at http://127.0.0.1:9 ([Errno 111] Connection refused).
Start one with `jev serve`, or pass the URL it is actually listening on.

JevError: cannot reach jev at http://no-such-host.invalid:8000 ([Errno -2] Name or
service not known). Start one with `jev serve`, or pass the URL it is actually
listening on.
```

`.status` is `None`: nothing answered, so there is no HTTP code - and `.detail` is
`None` too, for the same reason, so print the exception itself (`print(e.status,
e.detail or e)`) if you want the sentence in that case. The OS reason in parentheses is
the diagnostic: `Connection refused` means wrong port or the process is
down; `Name or service not known` means DNS. Check `curl http://host:port/v1/health`;
health is open even on a keyed server precisely so that check always works.

### `did not answer within 60.0s`

```text
JevError: GET http://127.0.0.1:42513/v1/health did not answer within 2s; raise timeout=
if the server is loading a model
```

The default timeout is 60 s. The commonest legitimate cause is exactly what the message
says: a server still loading weights, seconds to tens of seconds from a warm cache here,
minutes cold. Raise it for the first call, or wait for `/v1/health` to answer before
sending work:

```python
jev = jevimage.connect(url, timeout=300)
```

Long timeouts are also right for a large `train()` upload, which sends every image in one
request. From the CLI that is `--timeout SECONDS` on `jev ask` and `jev train`, or
`$JEV_TIMEOUT`, which covers all four remote commands, with the same 60 s default:

```bash
jev train street dataset/ --url http://127.0.0.1:8123 --timeout 600
export JEV_TIMEOUT=600          # same thing, once
```

### `401`: `send 'Authorization: Bearer <key>'`

```text
JevError: send 'Authorization: Bearer <key>' with a key this server was started with
(JEV_API_KEYS) [HTTP 401 from GET http://127.0.0.1:8124/v1/heads]
```

Missing key, wrong key, or a malformed header (`Authorization: secret` without `Bearer `
is a 401, not a 400).

```python
jevimage.connect(url, api_key="s3cret")
```

```bash
export JEV_API_KEY=s3cret      # what the CLI reads, alongside $JEV_URL
```

A server started without keys is **open**. `health()["auth"]` says `open` or `bearer`,
and that endpoint needs no key, so you can always check which mode you are in. On the
server side, keys come from `JEV_API_KEYS` (comma- or space-separated) or `jev serve
--api-key K1 K2`. A non-ASCII key is a plain 401, not a 500.

### `413`: too large, four different ways

```text
at most 8 images per request; got 9
images[0] is 13MB of base64; the limit is 12MB per image
images[0] is 8000x8000; the limit is 50MP per image
images[0] is too large to decode: Image size (196000000 pixels) exceeds limit of
178956970 pixels, could be decompression bomb DOS attack.
```

**Why four.** Compressed size is no bound on decoded size: a solid-colour 8000×8000 PNG
is 263 KB of base64 and ~190MB of pixels once decoded, and the same trick scales as far
as you like. So the byte length, the pixel count from the header, and PIL's own
decompression-bomb guard are all checked, each before the expensive step it protects.

The last one quotes PIL and stops there. Which guard fired and at what number is already
in PIL's own sentence (`178956970` above is PIL's ceiling, not the 50MP one), so adding
"the limit is 50MP per image" after it would name a limit that was not the one refused.
The third message is the 50MP guard, and it says so.

**The image-count cap is not your problem for `embed()`**: the client chunks at 8
automatically, so `jev.embed([...20 images...])` works and matches `Jev.embed` exactly:

```python
len(jev.embed(["ds/red/0.png"] * 20))
```

```text
20
```

You only hit `at most 8 images per request` by building the HTTP request yourself.
Training is exempt: `POST /v1/heads` is a bulk upload bounded by `MAX_EXAMPLES` (4000),
not by 8, and it counts in its own noun:

```text
at most 4000 examples per request; got 4001
```

**Fixes.** Resize before uploading; a 224px or 384px encoder gains nothing from a 50 MP
photo. If you genuinely need larger, `MAX_IMAGES`, `MAX_B64` and `MAX_PIXELS` are
module-level constants in `jevimage/server.py`, so raising them means running a patched
server, not passing a flag.

### `415`: unsupported format

```text
images[0] is GIF; this server reads JPEG, PNG, WEBP
```

Checked before any `convert()`, because that is what drops `.format`. Convert first. An
unrecognisable format reports `an unknown format`.

### `422`: malformed request

```text
images[0] is not a base64-encoded image; send the raw bytes base64'd, or a
'data:image/png;base64,...' URL
images[0] is a truncated or corrupt PNG
send 'image' (one base64 string) or 'images' (a list)
send 'image' or 'images', not both: 'image' returns one answer object, 'images' returns
a list
```

"Truncated or corrupt" is separate from "not an image" because PIL decodes lazily: the
header parsed fine and the pixels did not. Usually a partial upload or a half-written
file.

**`image` and `images` together are refused, not resolved.** They do not merely select
different pixels, they select different response shapes (`image` answers with one object
under `answers`, `images` answers with a list), so picking one silently would change what
the caller gets back. Send exactly one, on `/v1/ask` and on `/v1/embed` alike. A client
built on `jevimage.connect()` never sends both.

On `POST /v1/heads` the per-item prefix is `examples[i]`, matching the field the item
came from:

```text
examples[1] is not a base64-encoded image; send the raw bytes base64'd, or a
'data:image/png;base64,...' URL
```

The core's own `ValueError`s also arrive as 422, with the message intact. That is the
design, not a leak. An unknown head over HTTP reads exactly as it does locally:

```text
JevError: unknown head 'nope'; trained heads: colours, docref_colours2
[HTTP 422 from POST http://127.0.0.1:8123/v1/ask]
```

A 422 whose detail is a **list** of field reports came from pydantic, not from jevimage;
your JSON body has the wrong shape:

```text
{"detail": [{"type": "missing", "loc": ["body", "questions"], "msg": "Field required",
"input": {"image": "x"}}]}
{"detail": [{"type": "less_than_equal", "loc": ["body", "epochs"], "msg": "Input should
be less than or equal to 5000", "input": 99999, "ctx": {"le": 5000}}]}
```

`loc` names the offending field. `epochs` is capped at 1..5000 because it is a loop count
a caller can set: 5000 epochs on 4000 examples is about 70 seconds on the 16-thread
laptop CPU these numbers were measured on, and 5 million is an outage.

### `404`: head, or route

```text
JevError: no head named 'nope' (have: colours, docref_colours2)
[HTTP 404 from GET http://127.0.0.1:8231/v1/heads/nope]
```

The same miss on a local `Jev` (`jev.head("nope")`) names the directory it searched:
`KeyError: "no head named 'nope' in /srv/jev/heads (have: ...)"`, because that is what
an operator at a REPL needs. The server strips any filesystem path out of a 404 body on
the way out, so a caller who can guess a head name cannot map the server's disk. Nothing
to configure, and nothing to put behind a proxy for this.

A head file that exists but will not load is also a 404 here, carrying the store's own
reason rather than "unknown head":

```text
JevError: junk.pt is present but unreadable: UnpicklingError: Weights only load failed.
... Delete it, or retrain the head. [HTTP 404 from POST http://127.0.0.1:8231/v1/ask]
```

`Remote.head()` propagates the 404; `Remote.delete_head()` translates it to `False`,
matching `Jev.delete_head`. (`GET /v1/heads` names no head, so it has no 404 of this
kind; an empty list is the answer.) The CLI turns that into an error with a non-zero
exit:

```console
$ jev rm nope --url http://127.0.0.1:8123
jev: no head named 'nope' on http://127.0.0.1:8123
exit 1
```

A bare `Not Found` with no jevimage wording is a routing problem; check the path and
that you did not drop `/v1`.

### `409`

Only from `GET /v1/heads/{name}/compare`, for a head saved without its embeddings. See
[`compare()` returns `None`](#compare-returns-none).

---

## Reading an `Answer`

`Answer` is a `dict` subclass with attribute access, so it serialises to JSON unchanged.
Asking for a field the answer type does not have says which fields it does have:

```python
a = jev.ask(img, {"scene": {"type": "choice", "criteria": {"a": "x", "b": "y"}}})["scene"]
a.noul
```

```text
AttributeError: 'choice' answers have no 'noul'; this one has: type, choice,
probabilities, confidence
```

Printing an answer gives a short repr, not the data:

```text
<choice 'a' p=0.978>
```

For the full shape, serialise it:

```python
json.dumps(a)
```

```text
{"type": "choice", "choice": "a", "probabilities": {"a": 0.977548, "b": 0.022453},
 "confidence": 0.955095}
```

Fields by type: `choice` and `head` carry `choice` / `probabilities` / `confidence`
(`head` adds `head`, `trained_on`, `accuracy`); `score` carries `score` /
`probabilities` / `legend` / `confidence`; `noul` carries `noul` / `basis`. Full shapes
are in [Question types](/docs/questions).

---

## HTTP status codes at a glance

| Code | Meaning here | Where to look |
| --- | --- | --- |
| `401` | Missing or wrong bearer key | [401](#401-send-authorization-bearer-key) |
| `404` | No such head, a head file that will not load, or no such route | [404](#404-head-or-route) |
| `409` | Head has no stored embeddings to compare | [409](#409) |
| `413` | Too many images, too many bytes, or too many pixels | [413](#413-too-large-four-different-ways) |
| `415` | Not JPEG, PNG or WebP | [415](#415-unsupported-format) |
| `422` | Malformed question, bad body, or a core `ValueError` verbatim | [422](#422-malformed-request) |
| none | Nothing answered — wrong URL, dead server, DNS, timeout | [Connection refused, or unknown host](#connection-refused-or-unknown-host) |

---

## Still stuck

- `jev encoders`: works with no torch and no network; confirms the package imports.
- `curl http://host:port/v1/health`: open even on a keyed server; tells you the
  encoder, its `dim`, the device, the head count and the auth mode.
- `jev heads`: read **stderr** too, which is where unreadable head files are reported.
- `jev.heads.broken`: the same, from Python.
- `jev ask ... --json`: the raw answer shape, when the formatted output is hiding
  something.
- Every module ships a runnable self-check: `python3 -m jevimage.client`,
  `python3 -m jevimage.server`, `python3 -m jevimage.training`. If one of those fails,
  the problem is the install, not your code. All three stand up a real encoder and a
  real server, so they need `pip install 'jevimage[serve]'`; on an API-only install they
  exit with `No module named 'torch'`, which is the install working as intended. The
  self-check that fits that install is `jev encoders`.

Related: [Install](/docs/install) · [Choosing an encoder](/docs/encoders) ·
[Question types](/docs/questions) · [Serving](/docs/serving) ·
[HTTP reference](/docs/reference-http) · [Python reference](/docs/reference-python)

---

[← Cookbook](/docs/cookbook) · [Docs index](/docs/index)
