---
title: "Python API"
description: "Every exported name, every method of Jev and Remote, what each returns and what it raises."
---

Every public name in `jevimage`, with its exact signature, what it returns, what it
raises, and a one-line example. Nothing here is aspirational: every snippet below was
run, and its output pasted back, shortened only where a long absolute path or a
repeated float would obscure the point, never altered.

For the ideas behind the API (why answers are probabilities, why a head is linear),
start at [index.md](index.md). For the same surface over HTTP see
[reference-http.md](reference-http.md), and for the shell see
[reference-cli.md](reference-cli.md).

**About the numbers on this page.** Real encoder weights are a multi-gigabyte download,
so every example runs against a deterministic 16-dimensional *toy* encoder
(`name = "toy"`, `dim = 16`, no `logit_scale`/`logit_bias`) and the 12-image red/blue
`ds/` dataset that [`examples/make_fixtures.py`](https://github.com/Per0x1de-1337/jevimage/blob/main/examples/make_fixtures.py) writes. The shapes, types, keys and error messages are exactly what you get from
SigLIP2; the probability values are the toy's and mean nothing about images. Where a
number would be misleading if you read it as a model result, it is called out.

---

## The public surface

```python
import jevimage
jevimage.__all__
```

```
['load', 'connect', 'register', 'available', 'Jev', 'Head', 'Answer',
 'Encoder', 'JevError', 'to_image', 'read_folder', 'as_examples',
 'confidence', 'IMAGE_SUFFIXES', 'DEFAULT', '__version__']
```

| Name | Kind | Needs torch? |
| --- | --- | --- |
| [`load`](#load) | function → `Jev` | yes |
| [`connect`](#connect) | function → `Remote` | no |
| [`register`](#register) | function | no |
| [`available`](#available) | function → `list[str]` | no |
| [`Jev`](#jev) | class | yes |
| [`Head`](#head) | class | no |
| [`Answer`](#answer) | class (`dict` subclass) | no |
| [`Encoder`](#encoder) | class (the interface) | no |
| [`JevError`](#jeverror) | exception (`ValueError` subclass) | no |
| [`to_image`](#to_image) | function → `PIL.Image` | no |
| [`read_folder`](#read_folder) | function → `dict` | no |
| [`as_examples`](#as_examples) | function → `(list, list)` | no |
| [`confidence`](#confidence) | function → `float` | no |
| [`IMAGE_SUFFIXES`](#image_suffixes) | `set[str]` | no |
| [`DEFAULT`](#default) | `str` | no |
| [`__version__`](#__version__) | `str` | no |

`load`, `Jev`, `Encoder`, `register`, `available`, `DEFAULT` and `JevError` are resolved
lazily by a module `__getattr__`, so `import jevimage` never imports torch. Only `load`
and `Jev` *need* it; the rest are lazy purely to keep import time down, and they work on
an API-only install.

The object `connect()` returns is `jevimage.client.Remote`. It is deliberately **not**
exported; you get one from `connect()`, you do not construct one by name.

### On an API-only install

`pip install jevimage` brings pillow and the standard library, nothing else. Everything
above works except `load`/`Jev`, which say so:

```python
import jevimage
jevimage.available()        # -> ['clip-vit-l-14', 'dfn5b-h-14-384', 'metaclip2-huge',
                            #     'siglip2-base-224', 'siglip2-base-384',
                            #     'siglip2-giant-384', 'siglip2-so400m-384']
jevimage.load()
```

```
ImportError: jevimage.load runs the encoder in this process, which needs torch and
transformers: pip install 'jevimage[local]'
To use a server instead - no torch required - call jevimage.connect(url).
(underlying import error: No module named 'torch')
```

See [install.md](install.md) for the four install paths.

---

## Entry points

### `load`

```python
jevimage.load(encoder: str | Encoder | None = None, *, heads_dir=None,
              temperature=None, batch=False) -> Jev
```

Loads an encoder into this process and returns a [`Jev`](#jev).

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `encoder` | `str`, `Encoder`, or `None` | `None` | A registry name, `'a+b'` for an ensemble, or any object with `.name`/`.dim`/`.img()`/`.txt()` used as-is. `None` reads `$JEVIMAGE_ENCODER`, else [`DEFAULT`](#default). |
| `heads_dir` | path-like or `None` | `None` | Where trained heads live. `None` means `$JEVIMAGE_HEADS_DIR`, else `$JEVIMAGE_HOME/heads`, else `~/.jev/heads`. |
| `temperature` | `float` or `None` | `None` | Softmax divisor for `choice`/`score`/`noul` pair. `None` uses `1 / encoder.logit_scale` when the encoder publishes one, else `0.01`. Must be a real number between `1e-6` and `1e6`. |
| `batch` | `bool` | `False` | Hold single-image `embed()` calls for a few milliseconds so concurrent ones ride one forward pass. Worth it in a server, pure overhead in a script. |

Raises `ImportError` if torch/transformers are missing, `KeyError` for an unknown
registry name, and `ValueError` for a non-positive or non-finite `temperature`.
Instances are cached per registry name, so calling `load("siglip2-base-224")` twice
loads the weights once.

```python
jev = jevimage.load(Toy(), heads_dir="…/heads")
repr(jev)        # -> "<Jev encoder='toy' dim=16 heads=0>"
```

Note the exception type for a bad name is `KeyError`, not `ValueError`:

```python
jevimage.load("nope")
```

```
KeyError: "unknown encoder 'nope'. Available: clip-vit-l-14, dfn5b-h-14-384,
metaclip2-huge, siglip2-base-224, siglip2-base-384, siglip2-giant-384,
siglip2-so400m-384. Register your own with jevimage.register(name, factory) - from
Python, in the process that will serve it; the `jevimage` command only sees these built-in
names, because it never imports your code."
```

`jevimage.Jev(...)` takes the identical arguments; `load()` is the name to use.

### `connect`

```python
jevimage.connect(url: str, *, api_key: str | None = None, timeout: float = 60.0,
                 max_edge: int | None = 384) -> Remote
```

Returns a [`Remote`](#remote) pointed at a `jevimage serve` instance. Downloads nothing,
imports no torch.

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `url` | `str` | none | Base URL including scheme. Trailing slashes are stripped. |
| `api_key` | `str` or `None` | `None` | Sent as `Authorization: Bearer <key>` on every `/v1` call. `None` for an open server. Refused at construction with a `ValueError` if it has surrounding whitespace (the usual `$(cat key.txt)` newline) or characters that cannot go in an HTTP header - no server key can have either, since `$JEVIMAGE_API_KEYS` is split on whitespace. Redirects are refused, not followed, so the key never travels to a host you did not name - use `https` and a URL you control. |
| `timeout` | `float` | `60.0` | Seconds per request. Raise it if the server may still be loading a model. |
| `max_edge` | `int` or `None` | `384` | Downscale an image to this longest edge before uploading it. No encoder in the registry reads more than 384px, so anything larger is upload time spent on pixels the server throws away: a 2560px PNG goes from 3.1 MB on the wire to 26 KB. Size matters more than it looks on a fresh connection: TCP starts with a ~14KB window, so each doubling of the body costs another round trip. Measured at 190ms RTT, 79KB took 940ms and 26KB took 660ms. Images already within the limit are sent byte for byte, unmodified, and so is an oversized one whose downscaled re-encode would not actually be smaller (a small flat-colour PNG, say). A `data:` URL is passed through exactly as given, since it is already the bytes that go on the wire. `None` disables it. |

Raises `ValueError` immediately if `url` has no scheme. That is the only validation done
at construction time, because nothing else can be checked without a round trip:

```python
jevimage.connect("127.0.0.1:8123")
# ValueError: url '127.0.0.1:8123' has no scheme; use 'http://127.0.0.1:8123'
```

```python
jev = jevimage.connect("http://127.0.0.1:8123")
repr(jev)        # -> "<Remote 'http://127.0.0.1:8123'>"
```

A keyed client reprs as `<Remote 'http://…' keyed>`. The key itself is never printed.

`Remote` keeps one connection open and reuses it. A TCP handshake costs a full round
trip, which against a server on another continent is around 200ms, an order of magnitude
more than the model spends answering, so paying it once per call would dominate every
measurement. Calls are serialised on a lock, so one instance is safe to share between
threads; for genuine upload concurrency, give each thread its own. A connection the
server has since closed is reopened once, silently, for a `GET` or a `DELETE`. A
`POST` (`ask`, `embed`, `train`) is not replayed - the failure can surface after the
server has already done the work - so it raises `JevError` and you decide whether
repeating it is safe. Release it with `close()`, or use
the client as a context manager:

```python
with jevimage.connect("http://127.0.0.1:8123") as jev:
    for path in paths:
        print(jev.ask(path, questions)["q"])
```

---

## `Jev`

The local object: a loaded encoder plus its head directory. Thread-safe for reads;
`train` and `delete_head` write files.

Attributes: `.encoder` (the encoder object), `.heads` (a `HeadStore`, a real `dict` of
name → raw record *including* the weights), `.engine` (the readout engine).

Methods: [`ask`](#jevask), [`ask_embedding`](#jevask_embedding), [`embed`](#jevembed),
[`train`](#jevtrain), [`compare`](#jevcompare), [`head`](#jevhead),
[`list_heads`](#jevlist_heads), [`delete_head`](#jevdelete_head).

### `Jev.ask`

```python
ask(image, questions, *, temperature=None) -> dict[str, Answer]
```

Encode `image` once, then answer every question against that one embedding.

| Parameter | Type | Meaning |
| --- | --- | --- |
| `image` | anything [`to_image`](#to_image) accepts | Path, `str`, `bytes`, `PIL.Image`, or a `data:` URL. EXIF orientation is applied before encoding. |
| `questions` | `dict[str, dict]` or `list[dict]` | A mapping of id → question, or a list; list entries get `q1`, `q2`, … unless they carry an `"id"` key. |
| `temperature` | `float` or `None` | Overrides the instance temperature for this call only. Must be a real number between `1e-6` and `1e6`. |

Returns `{question id: Answer}` in the order the questions were given. Raises
`ValueError` for a malformed question, an unknown head, a bad temperature, duplicate
ids, or a non-finite result; [`KeyError`](#jevhead) if a `head` question names a head
file that is present but will not load; `TypeError` if `questions` is neither a dict nor
a list, or if `image` is a list rather than one image. `Jev` and `Remote` refuse it with
the same first sentence; the advice differs, because `ask_embedding` is local-only:

```python
jev.ask(["ds/red/0.png", "ds/blue/0.png"], q)
# TypeError: ask() takes one image, got a list of 2. Loop over them, or call embed()
#            once and ask_embedding() per row.

remote.ask(["ds/red/0.png", "ds/blue/0.png"], q)
# TypeError: ask() takes one image, got a list of 2. Loop over them, one ask() each.
```

```python
a = jev.ask("ds/red/0.png", {
    "colour":  {"type": "choice", "criteria": {"red": "a red square",
                                               "blue": "a blue square"}},
    "busy":    {"type": "score",  "criteria": ["empty", "half full", "packed"]},
    "outside": {"type": "noul",   "criteria": {"true": "outdoors",
                                               "false": "indoors"}},
})
a["colour"]     # -> <choice 'red' p=1.000>
a["busy"]       # -> <score 1.28/2>
a["outside"]    # -> <noul 0.046>
```

Question shapes are covered in full in [questions.md](questions.md). The four `type`
values are `choice`, `score`, `noul` and `head`; every type also accepts an optional
`"prompt"` template that must contain `{}`.

### `Jev.ask_embedding`

```python
ask_embedding(embedding, questions, *, temperature=None) -> dict[str, Answer]
```

The same, against an embedding you already hold. **Local only.** There is no remote
equivalent, because the entire point is skipping the network hop that `Remote` *is*.
Use it to ask more questions later about an image you embedded earlier.

`embedding` is a `(dim,)` or `(n, dim)` fp32 tensor; only the first row is read.
It must be L2-normalised; the engine checks and refuses otherwise.

```python
e = jev.embed("ds/red/0.png")           # (1, 16)
jev.ask_embedding(e,    q)["c"].choice  # -> 'red'
jev.ask_embedding(e[0], q)["c"].choice  # -> 'red'   (1-D is accepted too)
```

### `Jev.embed`

```python
embed(images) -> torch.Tensor
```

One image or a list of them → an `(n, dim)` fp32 L2-normalised tensor. This is the
whole cost of a request; everything downstream is a matmul, so cache what this returns
if you will ask about the same image again.

```python
jev.embed("ds/red/0.png")                    # -> Tensor (1, 16) torch.float32
jev.embed(["ds/red/0.png", "ds/blue/0.png"]) # -> Tensor (2, 16)
jev.embed([])
# ValueError: embed() needs at least one image
```

A single image always comes back as `(1, dim)`, not `(dim,)`. There is no cap on the
list length locally.

### `Jev.train`

```python
train(name, data, *, epochs=500, max_per_class=None, template=None) -> Head
```

Fit a linear head on the frozen encoder and save it to `heads_dir`.

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | `str` | none | 1–64 chars from `[A-Za-z0-9._-]`, starting alphanumeric. Becomes a filename, so it is validated, never sanitised. |
| `data` | folder path, `{label: [images]}`, or `[(image, label)]` | none | See [`as_examples`](#as_examples). |
| `epochs` | `int` | `500` | Adam steps for each fit (k CV fits plus the final one). `Remote.train` bounds this to 1..5000 and refuses anything else locally, because the server does (5 million epochs on 4000 examples is an outage); a local `Jev` takes any int. |
| `max_per_class` | `int` or `None` | `None` | Take at most this many examples per label, in all three `data` shapes - see [`as_examples`](#as_examples). A positive int, or `None` for no cap; anything else is refused by name. |
| `template` | `str` or `None` | `None` | The zero-shot prompt template used for the **comparison only**, not for training. Default `"a photo of {}"`. |

Returns a [`Head`](#head) whose `.comparison` is already filled in. Raises `ValueError`
for fewer than 2 distinct labels, more than 100 classes, any class with fewer than 2
examples, labels differing only by surrounding whitespace, more than 4000 examples, or
an invalid name.

```python
head = jev.train("colours", "ds/")
repr(head)          # -> "<Head 'colours' 2 classes, 12 examples, 50.0% held out>"
head.classes        # -> ['blue', 'red']
head.counts         # -> {'blue': 6, 'red': 6}
head.accuracy       # -> 0.5
head.per_class      # -> {'blue': 0.5, 'red': 0.5}
```

`accuracy` is **held-out**, from stratified k-fold with `k = min(5, thinnest class)`,
never the fit. It is `None` only when no fold was possible, which means a class with
fewer than 2 examples, something `train()` refuses outright, so a head trained here
always carries a number. (50.0% here is the toy encoder hashing pixel bytes on twelve
squares; it is not a claim about SigLIP.)

Failure modes worth knowing:

```python
jev.train("one",  {"a": [img, img]})
# ValueError: need at least 2 distinct labels
jev.train("thin", {"a": [img_a], "b": [img_b]})
# ValueError: every class needs at least 2 examples; too few for: a, b
```

### `Jev.compare`

```python
compare(name, *, template=None) -> dict | None
```

Did training actually help, per class, against zero-shot prompts, on the same images.
Returns `None` for a head saved without its embeddings. Raises `KeyError` if no such
head, `ValueError` if the head was fitted under a different encoder.

```python
jev.compare("colours")
```

```json
{
  "overall": {"zero_shot": 0.75, "trained": 0.5, "delta": -0.25},
  "per_class": [
    {"label": "blue", "n": 6, "zero_shot": 0.5, "trained": 0.5},
    {"label": "red",  "n": 6, "zero_shot": 1.0, "trained": 0.5}
  ],
  "basis": "trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example"
}
```

The two sides are not scored the same way, and `basis` says so in every response. That
asymmetry favours zero-shot, which is the right bias for a "should I have bothered"
question.

`template` changes the prompts the baseline is built from, which moves the zero-shot
side. On the toy data, `"a close-up photo of {}"` moves it from `0.75` to `0.5` and
turns the verdict from `delta: -0.25` into `delta: 0.0`. A zero-shot baseline is only
as good as the prompt you compare against.

### `Jev.head`

```python
head(name) -> Head
```

One saved head's metadata. **`.comparison` is always `None` here.** Computing it means
re-scoring the stored embeddings, so it is done by `train()` and `compare()`, not by
this lookup.

```python
jev.head("colours")             # -> <Head 'colours' 2 classes, 12 examples, 50.0% held out>
jev.head("colours").comparison  # -> None
jev.head("nope")
# KeyError: "no head named 'nope' in …/heads (have: colours)"
```

A head file that exists but will not load is reported as unreadable rather than as
missing. "unknown head" about a file sitting right there is a lie:

```python
jev.head("broken")
# KeyError: 'broken.pt is present in …/heads but unreadable: UnpicklingError: Weights
#            only load failed. …'
```

`list_heads()` omits such a file entirely; `jev.heads.broken` is the `{name: why}` map
if you need to see them.

### `Jev.list_heads`

```python
list_heads() -> list[dict]
```

Every saved head as a plain dict, sorted by name, re-reading the directory first so a
head another process trained a moment ago is visible. Weights are never included.

```python
jev.list_heads()
```

```json
[{"name": "colours", "encoder": "toy", "classes": ["blue", "red"], "n": 12,
  "counts": {"blue": 6, "red": 6}, "cv_accuracy": 0.5,
  "per_class": {"blue": 0.5, "red": 0.5}, "folds": 5,
  "created": 1790180568.554609, "dim": 16, "bytes": 136}]
```

`created` is a Unix timestamp; `bytes` is the size of the weight matrices in fp32.

### `Jev.delete_head`

```python
delete_head(name) -> bool
```

`True` if a file was removed, `False` if there was nothing to remove. Raises
`ValueError` for an invalid name. Deleting twice is not an error:

```python
jev.delete_head("colours")   # -> True
jev.delete_head("colours")   # -> False
```

---

## `Remote`

What [`connect()`](#connect) returns. Same method names, arguments and return types as
`Jev`, so moving a script from a local encoder to a server is a one-line change.
It keeps one connection open. Calls are serialised on a lock, so one instance is
safe to share between threads; for genuine upload concurrency, give each thread its
own, and release the connection with `close()` or the context manager.

Attributes: `.url` (normalised, no trailing slash), `.api_key`, `.timeout`.

Methods: [`health`](#remotehealth), `ask`, `embed`, `train`, `compare`, `head`,
`list_heads`, `delete_head`.

### What differs from `Jev`

| | `Jev` | `Remote` |
| --- | --- | --- |
| `ask_embedding` | yes | **absent** - its purpose is skipping the network |
| `health` | **absent** - there is no server to report on | yes |
| `embed` return | `torch.Tensor` | `torch.Tensor` if torch is installed, else `list[list[float]]` |
| `embed` batching | uncapped | chunked into requests of 8, transparently |
| `train`/`compare` | take `template=` | do not (the server picks the default template) |
| `train(epochs=)` | any int | must be 1..5000, refused locally before any HTTP |
| a lost connection | n/a | reopened once for `GET`/`DELETE`; a `POST` is never replayed |
| unknown head in `head()`/`compare()` | `KeyError` | [`JevError`](#jeverror), `.status == 404` |
| unknown head in `ask()` | `ValueError` | `JevError`, `.status == 422` |
| unreadable head file in `ask()` | `KeyError` | `JevError`, `.status == 404` |
| bad head name | `ValueError` | `ValueError`, raised locally before any HTTP |
| introspection | `.encoder`, `.heads`, `.engine` | `.health()` |

Everything else matches, including `Answer` objects out of `ask`, a `Head` out of
`train`, `None` out of `compare` for a head with no stored embeddings, and `False` out
of `delete_head` for a head that was not there.

### `Remote.health`

```python
health() -> dict
```

What the server is running. Never requires the API key, so a load balancer without the
key does not report a live process as dead.

```python
jev.health()
```

```json
{"encoder": "toy", "dim": 16, "device": "cpu",
 "primitives": ["choice", "score", "noul", "head"],
 "heads": 2, "auth": "open", "version": "0.1.20"}
```

This is how you find out remotely which encoder your heads will be trained under. It
does **not** tell you whether single-caption `noul` will work; that needs the server's
`logit_scale`/`logit_bias`, which `/v1/health` does not report. Ask a single-caption
`noul` and read the error, or use the `{"true": …, "false": …}` pair form, which works
on every encoder.

### The rest, verified against a running server

```python
jev = jevimage.connect("http://127.0.0.1:8123")

jev.ask(img, {"colour": {"type": "choice",
                         "criteria": {"red": "a red square",
                                      "blue": "a blue square"}}})
# -> {'colour': <choice 'red' p=1.000>}          (values are Answer instances)

jev.embed(img)                    # -> Tensor (1, 16) torch.float32   (with torch)
                                  # -> list of 1 list of 16 floats    (API-only)

h = jev.train("docref_colours", "ds/")
repr(h)                           # -> "<Head 'docref_colours' 2 classes, 12 examples, 50.0% held out>"
h.encoder                         # -> 'toy'
h.comparison["overall"]           # -> {'zero_shot': 0.75, 'trained': 0.5, 'delta': -0.25}

jev.head("docref_colours")        # -> <Head 'docref_colours' 2 classes, 12 examples, 50.0% held out>
[x["name"] for x in jev.list_heads()]
                                  # -> ['colours', 'docref_colours', 'docs_colour']
                                  #    (a server's heads are shared by every client)
jev.compare("docref_colours")["overall"]
                                  # -> {'zero_shot': 0.75, 'trained': 0.5, 'delta': -0.25}
jev.ask(img, {"c": {"type": "head", "head": "docref_colours"}})
                                  # -> {'c': <head 'red' p=0.986>}
jev.delete_head("docref_colours") # -> True
jev.delete_head("docref_colours") # -> False
```

`train()` uploads every image once; the head and its weights stay on the server, and
what comes back is metadata. A head name is validated on this side before a URL is
built out of it:

```python
jev.head("../etc")
# ValueError: invalid head name '../etc': use 1-64 characters from [A-Za-z0-9._-],
#             starting with a letter or digit
```

---

## `Answer`

A `dict` subclass, so it serialises to JSON unchanged, with attribute access because
`a.choice` reads better than `a["choice"]`. `repr()` is a one-line summary; the full
shape is `json.dumps(answer)`.

```python
a["colour"]            # -> <choice 'red' p=1.000>
json.dumps(a["colour"])
# {"type": "choice", "choice": "red",
#  "probabilities": {"red": 1.0, "blue": 0.0}, "confidence": 0.999999}
```

A missing key raises `AttributeError` naming what the answer *does* have:

```python
a["outside"].choice
# AttributeError: 'noul' answers have no 'choice'; this one has: type, noul, basis
```

### Shapes per question type

| `type` | Keys | Types |
| --- | --- | --- |
| `choice` | `type`, `choice`, `probabilities`, `confidence` | `str`, `str` (winning label), `dict[label, float]` summing to 1, `float` 0–1 |
| `score` | `type`, `score`, `probabilities`, `legend`, `confidence` | `str`, `float` (expected level index, `0 … len-1`), `dict[str(level index), float]`, `dict[str(level index), str]` (the rubric text), `float` |
| `noul` | `type`, `noul`, `basis` | `str`, `float` 0–1, `str`: `"contrast pair"` or `"sigmoid readout"` |
| `head` | `type`, `head`, `choice`, `probabilities`, `confidence`, `trained_on`, `accuracy` | `str`, `str` (head name), `str`, `dict[class, float]`, `float`, `int` (examples fitted), `float` or `None` (held-out) |

Probabilities are rounded to 6 decimals, `score` to 4. Real examples of each:

```json
{"type": "score", "score": 1.2774,
 "probabilities": {"0": 0.007146, "1": 0.708312, "2": 0.284542},
 "legend": {"0": "empty", "1": "half full", "2": "packed"},
 "confidence": 0.562468}
```

```json
{"type": "noul", "noul": 0.045724, "basis": "contrast pair"}
```

```json
{"type": "head", "head": "colours", "choice": "red",
 "probabilities": {"blue": 0.014136, "red": 0.985864}, "confidence": 0.971727,
 "trained_on": 12, "accuracy": 0.5}
```

`basis` tells you which `noul` form ran. The `{"true": …, "false": …}` pair gives
`"contrast pair"` and works on every encoder. A single caption gives `"sigmoid readout"`
and needs an encoder publishing **both** `logit_scale` and `logit_bias`. SigLIP does;
CLIP-family models and every `Ensemble` do not, and say so rather than returning a
number that looks like a probability and is not one:

```python
jev.ask(img, {"n": {"type": "noul", "instructions": "outdoors"}})
# ValueError: encoder 'toy' has no calibrated sigmoid, so a single-caption noul cannot
#             produce a probability. Give a {"true": ..., "false": ...} criteria pair
#             instead, which works on every encoder.
```

---

## `Head`

The metadata of a trained head. The same object comes back from a local `train()` and
a remote one, because a head *is* metadata plus numbers; the weights stay wherever the
encoder is, and never cross the wire.

| Member | Type | Meaning |
| --- | --- | --- |
| `.name` | `str` | As given to `train()`. |
| `.classes` | `list[str]` | Sorted class labels; the column order of the weight matrix. |
| `.n` | `int` | Examples fitted. |
| `.counts` | `dict[str, int]` | Examples per class. `{}` if absent. |
| `.accuracy` | `float` or `None` | Held-out k-fold accuracy, `k = min(5, thinnest class)`. `None` when no fold was possible. Read it next to `n` and `counts`. |
| `.per_class` | `dict[str, float]` | Held-out accuracy per class. `{}` if absent. |
| `.encoder` | `str` or `None` | The encoder that fitted it. A head refuses to run under any other. |
| `.comparison` | `dict` or `None` | The zero-shot comparison, when one was computed. |
| `.to_dict()` | `dict` | The full public record. |

`to_dict()` carries four fields with no property of their own: `folds`, `created`,
`dim`, `bytes`.

```python
head.to_dict()
```

```json
{"name": "colours", "encoder": "toy", "classes": ["blue", "red"], "n": 12,
 "counts": {"blue": 6, "red": 6}, "cv_accuracy": 0.5,
 "per_class": {"blue": 0.5, "red": 0.5}, "folds": 5,
 "created": 1790180568.554609, "dim": 16, "bytes": 136}
```

Note the key is `cv_accuracy` in the dict and `.accuracy` on the object. `to_dict()` is
also exactly what `list_heads()` returns per head and what the HTTP API serves.

A head records its encoder and is checked, not trusted:

```python
other_encoder_jev.ask(img, {"q": {"type": "head", "head": "x"}})
# ValueError: head 'x' was trained on encoder 'toy' but this instance runs
#             'toy-calibrated'. Load 'toy', or retrain the head here.
```

The same check guards `compare()`. It compares the encoder *name*, not just the width,
because two encoders of equal dimension would otherwise produce confident nonsense.

---

## Encoders

### `Encoder`

The interface. Subclass it, or duck-type the four required members and pass the object
straight to [`load()`](#load); registration is optional.

| Member | Required | Type | Meaning |
| --- | --- | --- | --- |
| `.name` | yes | `str` | Provenance. A head records it and refuses to run under another. Stable identifier: rename it and you invalidate every head. |
| `.dim` | yes | `int` | Embedding width. |
| `.img(images)` | yes | `list[PIL.Image]` → `(n, dim)` fp32 | Must be L2-normalised. |
| `.txt(texts)` | yes | `list[str]` → `(n, dim)` fp32 | Must be L2-normalised. |
| `.logit_scale` | no | `float` or `None` | Multiplies the cosine before the sigmoid. Enables single-caption `noul` together with `logit_bias`. Absent is read as `None`. |
| `.logit_bias` | no | `float` or `None` | Shifts it. |

The normalisation requirement is enforced, not assumed. A rescaled embedding would
saturate the softmax to exactly 1.0 and 0.0:

```python
# an encoder whose img() returns vectors of norm 3
jev.ask(img, {"c": {"type": "choice", "criteria": {"a": "x", "b": "y"}}})
# ValueError: embeddings must be L2-normalised, got norm 3; wrap your encoder's output
#             in jevimage.encoders.unit()
```

`jevimage.encoders.unit()` does that for you: it pulls the tensor out of whatever a
model returned, mean-pools token-level output, and normalises in fp32.

### `register`

```python
jevimage.register(name: str, factory) -> None
```

Add an encoder to the registry. `factory` is called with no arguments the first time
`load(name)` asks for it, so the weights load lazily. Re-registering a name replaces
any cached instance.

Raises `ValueError` if `name` contains `'+'` (reserved for ensembles) and `TypeError`
if `factory` is not callable.

```python
jevimage.register("mine", lambda: Mine())
jevimage.available()
# -> ['clip-vit-l-14', 'dfn5b-h-14-384', 'metaclip2-huge', 'mine', 'siglip2-base-224',
#     'siglip2-base-384', 'siglip2-giant-384', 'siglip2-so400m-384']
repr(jevimage.load("mine").encoder)   # -> '<Mine mine dim=8>'
```

Registration lives in *your* process. The `jevimage` CLI never imports your code, so a
registered encoder is invisible to it; see [encoders.md](encoders.md).

### `available`

```python
jevimage.available() -> list[str]
```

Registry names, sorted. Ensembles are formed on demand from `'a+b'` and are not listed.
Needs no torch, so it works on an API-only install.

```python
jevimage.available()
# -> ['clip-vit-l-14', 'dfn5b-h-14-384', 'metaclip2-huge', 'siglip2-base-224',
#     'siglip2-base-384', 'siglip2-giant-384', 'siglip2-so400m-384']
```

### `DEFAULT`

```python
jevimage.DEFAULT     # -> 'siglip2-base-224'
```

The encoder `load()` uses when given neither an argument nor `$JEVIMAGE_ENCODER`.

---

## Image and data helpers

All four are pure pillow/stdlib and available on every install.

### `to_image`

```python
jevimage.to_image(src) -> PIL.Image
```

Accepts the four things people have, and refuses anything ambiguous:

| `src` | Handling |
| --- | --- |
| `PIL.Image` | used as-is |
| `bytes` / `bytearray` / `memoryview` | opened from the encoded bytes |
| `pathlib.Path` | exactly as the `str` rows below - the two are interchangeable |
| `str` starting `data:` (any case) | base64 payload after the comma is decoded |
| `str` starting `data:` that will not decode | `ValueError` naming the data URL |
| `str` that is an existing path | opened |
| `str` that is a directory | `IsADirectoryError` |
| `str` that is not | `FileNotFoundError` (including a name too long to be one) |
| anything else | `TypeError` |

```python
import base64

jevimage.to_image("ds/red/0.png").size                     # -> (64, 64)
jevimage.to_image(open("ds/red/0.png", "rb").read()).size  # -> (64, 64)

raw = open("ds/red/0.png", "rb").read()
url = "data:image/png;base64," + base64.b64encode(raw).decode()
url[:30]                        # -> 'data:image/png;base64,iVBORw0K'
jevimage.to_image(url).size     # -> (64, 64)

jevimage.to_image("nope.jpg")
# FileNotFoundError: no such image file: 'nope.jpg'. Pass a path that exists, raw image
#                    bytes, a PIL.Image, or a 'data:image/...;base64,...' string.
jevimage.to_image(5)
# TypeError: cannot read an image from int; pass a path, bytes, a PIL.Image, or a data URL
```

A plain string that is neither a path nor a data URL is an error rather than a guess at
base64, so a typo'd filename says so instead of failing three layers down. This
function does not restrict the format. `IMAGE_SUFFIXES` governs folder scanning, and
the HTTP server enforces its own format allowlist.

### `read_folder`

```python
jevimage.read_folder(folder, max_per_class: int | None = None) -> dict[str, list[Path]]
```

One sub-folder per label, the layout every image dataset already uses. Recurses into
each label folder, keeps only [`IMAGE_SUFFIXES`](#image_suffixes), sorts both labels and
files. Loose files directly in `folder` are ignored, because they have no label.

```python
jevimage.read_folder("ds/", max_per_class=2)
# -> {'blue': [PosixPath('ds/blue/0.png'), PosixPath('ds/blue/1.png')],
#     'red':  [PosixPath('ds/red/0.png'),  PosixPath('ds/red/1.png')]}

jevimage.read_folder("ds/red/0.png")
# NotADirectoryError: 'ds/red/0.png' is not a directory
jevimage.read_folder("docref/")
# ValueError: no labelled images under 'docref/'. Expected one sub-folder per label,
#             each holding .jpeg/.jpg/.png/.webp files.
```

### `as_examples`

```python
jevimage.as_examples(data, max_per_class=None) -> tuple[list, list[str]]
```

Normalises the three shapes `train()` accepts into parallel `(images, labels)` lists.
Labels are always stringified.

```python
jevimage.as_examples({"red": ["ds/red/0.png"]})
# -> (['ds/red/0.png'], ['red'])
jevimage.as_examples([("ds/red/0.png", "red"), ("ds/blue/0.png", "blue")])
# -> (['ds/red/0.png', 'ds/blue/0.png'], ['red', 'blue'])
jevimage.as_examples([("ds/red/0.png", "red"), ("ds/red/1.png", "red"),
                      ("ds/blue/0.png", "blue")], max_per_class=1)
# -> (['ds/red/0.png', 'ds/blue/0.png'], ['red', 'blue'])

jevimage.as_examples(42)
# TypeError: train() takes a folder path, a {label: [images]} mapping, or a list of
#            (image, label) pairs
```

A `str`/`Path` is passed to [`read_folder`](#read_folder) first. `max_per_class`
applies to all three shapes: for a list of `(image, label)` pairs it counts per label and
keeps the first N of each. Images are not opened here; that happens at embed time.

### `confidence`

```python
jevimage.confidence(p) -> float
```

How peaked a distribution is, rescaled so uniform is `0.0` and certain is `1.0`:
`(max(p) - 1/k) / (1 - 1/k)`. `p` is any sequence of probabilities; `k = 1` returns
`1.0`.

```python
jevimage.confidence([0.5, 0.5])   # -> 0.0
jevimage.confidence([0.9, 0.1])   # -> 0.8
jevimage.confidence([1.0])        # -> 1.0
```

Raw max-probability is not comparable across questions. 0.5 is lukewarm over two
options and emphatic over fifty. This rescaling removes that dependence, so one
threshold works for every question you ask. `choice`, `score` and `head` answers carry
it already; this function is for distributions you built yourself.

### `IMAGE_SUFFIXES`

```python
sorted(jevimage.IMAGE_SUFFIXES)   # -> ['.jpeg', '.jpg', '.png', '.webp']
```

The extensions `read_folder` picks up (sorted here because it is a `set` and its repr
order changes between runs). Deliberately short: an animated GIF, an SVG or a RAW file is
not a thing to silently take the first frame of. It is a plain `set`, so you can add to
it, at your own risk, since the encoders and the HTTP server have their own format
expectations.

### `__version__`

```python
jevimage.__version__   # -> '0.1.20'
```

Also reported by `Remote.health()["version"]`, which is how you check that a server and
a client agree.

---

## `JevError`

```python
class JevError(ValueError):
    status: int | None      # HTTP status, or None if the request never got that far
    detail: str | None      # the server's own message, verbatim
```

Raised by every `Remote` method for a request the server refused or could not reach.
It subclasses `ValueError` on purpose: a local `Jev` raises `ValueError` for these same
refusals, so an existing `except ValueError` keeps working when `load()` becomes
`connect()`.

```python
try:
    jev.ask(img, {"c": {"type": "head", "head": "nope"}})
except jevimage.JevError as e:
    print(e)          # unknown head 'nope'; trained heads: colours, docs_colour
                      #   [HTTP 422 from POST http://127.0.0.1:8123/v1/ask]
    print(e.status)   # 422
    print(e.detail)   # unknown head 'nope'; trained heads: colours, docs_colour
    isinstance(e, ValueError)   # True
```

`str(e)` adds the status, method and URL for a log line; `.detail` is the part a human
needs. Typical statuses:

| `.status` | When |
| --- | --- |
| `401` | missing or wrong API key |
| `404` | no such head, from `head()` and `compare()`; also a head file that is present but will not load, which reaches `ask()` too. `delete_head()` turns a missing head into `False` instead. |
| `409` | head has no stored embeddings. `compare()` turns it into `None`, so you never see a 409. |
| `413` / `415` / `422` | image too large / wrong format / malformed request or unknown head |
| `None` | the request never reached a server |

The `None` case is the connection failure, and it names the port and what to run:

```python
jevimage.connect("http://127.0.0.1:9").list_heads()
# JevError: cannot reach a jevimage server at http://127.0.0.1:9 ([Errno 111] Connection refused).
#           Start one with `jevimage serve`, or pass the URL it is actually listening on.
```

A timeout raises `JevError` too, suggesting `timeout=`. Two statuses are deliberately
swallowed so `Remote` matches `Jev`: a `404` from `delete_head()` becomes `False`, and a
`409` from `compare()` becomes `None`.

---

## Limits and errors

These ceilings are enforced by the engine. They are module constants, not part of the
public API, but hitting one produces a message you may need to recognise.

| Limit | Value | Where |
| --- | --- | --- |
| options in one `choice` | 255 | `jevimage.core.MAX_OPTIONS` |
| levels in one `score` | 10 | `jevimage.core.MAX_LEVELS` |
| questions per `ask()` | 64 | `jevimage.core.MAX_QUESTIONS` |
| cached text embeddings | 50 000 | `jevimage.core.MAX_CACHED_TEXTS` |
| training examples | 4000 | `jevimage.training.MAX_EXAMPLES` |
| training classes | 100 | `jevimage.training.MAX_CLASSES` |
| images per HTTP request | 8 | `jevimage.client.MAX_IMAGES` (chunked transparently) |

Every message below is verbatim. All of them are `ValueError` unless marked otherwise.

| Trigger | Message |
| --- | --- |
| unknown `type` | `unknown question type 'colour'; expected one of: choice, score, noul, head` |
| `choice` with \<2 options | `choice needs a criteria object of 2..255 options, e.g. {"indoor": …}` |
| `choice` list with a repeat | `choice criteria repeats 'x'; each option needs its own caption` |
| `score` with \<2 levels | `score needs an ordered criteria array of 2..10 levels, low end first, …` |
| `head` naming nothing saved | `unknown head 'nope'; trained heads: none trained yet` |
| `prompt` without `{}` | `prompt 'a photo' has no '{}' placeholder, so every option would get the same caption; write e.g. 'a photo of {}'` |
| `temperature=0` | `temperature must be greater than 0, got 0` |
| `temperature=inf` | `temperature must be finite, got inf; an infinite temperature makes every option equally likely` |
| `noul` pair with one side blank | `noul criteria has an empty 'false' caption; both sides of a pair need text` |
| `noul` single caption, uncalibrated encoder | `encoder 'toy' has no calibrated sigmoid, …` |
| duplicate ids in a list | `two questions share the id 'a'; ids must be unique` |
| `questions` neither dict nor list | **`TypeError`**: `questions must be a {id: question} object or a list of questions` |
| >64 questions | `at most 64 questions per request` |
| non-unit embeddings | `embeddings must be L2-normalised, got norm 3; …` |
| `embed([])` | `embed() needs at least one image` |
| bad head name | `invalid head name '../etc': use 1-64 characters from [A-Za-z0-9._-], starting with a letter or digit` |
| head under another encoder | `head 'x' was trained on encoder 'toy' but this instance runs 'toy-calibrated'. …` |
| missing head, locally | **`KeyError`**: `no head named 'nope' in …/heads (have: none)` |
| head file present but unreadable, locally | **`KeyError`**: `junk.pt is present in …/heads but unreadable: UnpicklingError: …. Delete it, or retrain the head.` |
| unknown encoder name | **`KeyError`**: `unknown encoder 'nope'. Available: …` |
| `load()` without torch | **`ImportError`**: `jevimage.load runs the encoder in this process, …` |

`Answer` attribute misses raise `AttributeError`; `to_image` raises `FileNotFoundError`,
`IsADirectoryError`, `ValueError` (a malformed `data:` URL) or `TypeError`; `read_folder`
raises `NotADirectoryError`.

---

## Temperature

The divisor applied to the logits before the softmax. It affects `choice`, `score`, the
`noul` pair form and `head`; it does **not** affect the `noul` sigmoid readout, which is
calibrated by the encoder's own scale and bias.

Resolution differs between the two cases, because they are not the same kind of logit:

| Question type | `temperature=` given | not given |
| --- | --- | --- |
| `choice`, `score`, `noul` pair | that value | the instance temperature, i.e. `load(temperature=…)`, else `1 / encoder.logit_scale`, else `0.01` |
| `head` | that value | `1.0` - a trained head's logits are already on a sensible scale, so the instance temperature is **not** applied |
| `noul` single caption | ignored | ignored |

```python
q = {"c": {"type": "choice", "criteria": {"red": "a red square",
                                          "blue": "a blue square"}}}
jev.ask(img, q)["c"]["probabilities"]                    # -> {'red': 1.0, 'blue': 0.0}
jev.ask(img, q, temperature=1.0)["c"]["probabilities"]   # -> {'red': 0.537485, 'blue': 0.462515}
jev.ask(img, q, temperature=100.0)["c"]["probabilities"] # -> {'red': 0.500376, 'blue': 0.499624}
```

A `head` question reacts to the same argument, from a base of `1.0`:

```python
hq = {"c": {"type": "head", "head": "colours"}}           # the head trained above
jev.ask(img, hq)["c"]["probabilities"]                    # -> {'blue': 0.014136, 'red': 0.985864}
jev.ask(img, hq, temperature=0.5)["c"]["probabilities"]   # -> {'blue': 0.000206, 'red': 0.999794}
jev.ask(img, hq, temperature=5.0)["c"]["probabilities"]   # -> {'blue': 0.299652, 'red': 0.700348}
```

Higher flattens, lower sharpens. The default for an uncalibrated encoder is `0.01`,
which is the CLIP-family convention (a raw cosine lives in roughly `[0, 0.3]` and
softmaxes to nearly uniform without it). A calibrated encoder's own `logit_scale` is the
temperature it was trained to be read at, so overriding it costs you the calibration
that makes the number mean something. Change it only if you have measured the result.

---

[← CLI reference](reference-cli.md) · [Docs index](index.md) · [HTTP reference →](reference-http.md)
