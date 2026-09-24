---
title: "Training a head"
description: "Fit a linear head on the frozen encoder in seconds, and read whether it actually beat the prompts."
---

A *head* is a linear classifier fitted on top of a frozen encoder, over labels you chose
and images you own. You train one when prompts are not good enough, because your classes
have no natural caption (`batch_2024_reject` is not a phrase the encoder was trained on),
or because the encoder can see the difference but your wording cannot name it.

Training is a `train()` call, it runs on your labelled folder, and what comes back tells
you, per class and on held-out data, whether it was worth doing. Sometimes it was not.
This page shows both outcomes, because a page that only shows the win teaches you to
trust a number you have not checked.

```python
head = jev.train("lighting", "shots/")      # shots/daylight/*.png, shots/flash/*.png, ...
jev.ask("photo.png", {"lit": {"type": "head", "head": "lighting"}})
```

> **About the numbers on this page.** Every block below was run, and its output pasted
> back unedited. They use a stand-in encoder (a few lines, no download) which embeds an
> image as its mean colour and a caption as the colour word it mentions. That is enough
> to show every mechanism honestly: the shapes, the fold arithmetic, the comparison, the
> errors and the files are exactly what you get with SigLIP. What is *not* transferable
> is how good the numbers are. A real encoder sees content; this one only sees colour.
> No SigLIP accuracy is quoted anywhere on this page, because none was measured. The
> stand-in is at the bottom if you want to reproduce these outputs.

## Why a linear probe, and not fine-tuning

Fine-tuning updates the encoder. That means backprop through the whole model, a GPU,
thousands of labelled images to avoid destroying what the model already knew, and hours
of somebody's attention. Then you own a new set of weights: gigabytes to store, to ship,
to version, to load in every worker.

It also breaks everything else. The zero-shot prompts you were using were calibrated
against *that* embedding space. Move the space and every `choice`, every `score`, every
other head you fitted is measuring something subtly different, and nothing tells you.

A probe leaves the encoder exactly as it was. Only two tensors are fitted:

```text
W   (dim x K)   float32
b   (K,)        float32
```

The consequences are the whole reason the library works this way:

- **It is cheap to fit.** Multinomial logistic regression on a few thousand rows.
  No GPU required.
- **It is free at query time.** `emb @ W + b` on the embedding the request already
  computed. A head answers in the same request as your `choice` questions, off the same
  single encode. Ten heads cost ten matmuls, not ten forward passes.
- **It is small enough to treat as data.** The weights of a 5-class head on a 768-dim
  encoder are 15,380 bytes. You can commit that, or `COPY` it into an image.
- **Nothing else changes.** Your prompts still mean what they meant. Other heads still
  work. You can throw the head away and be exactly where you started.

The price is real and worth stating: a linear layer can only use what the encoder
already represents. If the encoder cannot tell your two classes apart at all, no linear
boundary over its output will either, and you will see that as a held-out accuracy near
chance. The fix then is a stronger encoder, not more epochs. See [Choosing an
encoder](/docs/encoders).

## The folder layout

One sub-folder per label. The folder name *is* the label:

```text
shots/
  daylight/00.png  01.png  ...
  flash/00.png     01.png  ...
  nightshot/00.png 01.png  ...
```

That folder is 36 flat squares - 12 warm, 12 near-white, 12 dark blue - written by
[`examples/make_fixtures.py`](https://github.com/jevimage/jevimage/blob/main/examples/make_fixtures.py) along with the `colours/`
folder used further down. The stand-in encoder at the bottom of this page sees an image's
mean colour and nothing else, so those exact squares are what every number below is a
number of; run the script in the directory you run these blocks from.

What `read_folder()` does with that, exactly:

- Every sub-directory becomes a class. Sub-directories are searched recursively, so
  `shots/daylight/2024/x.png` is a `daylight` image.
- Files with a suffix outside `jevimage.IMAGE_SUFFIXES` (`.jpg`, `.jpeg`, `.png`,
  `.webp`) are skipped silently. An animated GIF or a RAW file is not something to
  quietly take one frame of.
- **Loose files directly in `shots/` are ignored**, because they have no label. A folder
  of 500 unsorted images plus three labelled sub-folders trains on the three sub-folders
  and never mentions the 500.
- A sub-directory containing no usable image is not a class, it is absent.
- Files are sorted by name, and `max_per_class=N` keeps the **first N of that sorted
  list**, a deterministic truncation rather than a random sample. If your filenames encode a
  date, you get the oldest N.

## Train one

```python
import jevimage
from tint import Tint          # the stand-in encoder; normally jevimage.load()

jev = jevimage.load(Tint(), heads_dir="heads2")
head = jev.train("lighting", "shots/")

print(head)
print(head.classes, head.n, head.counts)
print(head.accuracy, head.per_class)
```

```text
<Head 'lighting' 3 classes, 36 examples, 97.2% held out>
['daylight', 'flash', 'nightshot'] 36 {'daylight': 12, 'flash': 12, 'nightshot': 12}
0.9722222222222222 {'daylight': 0.9167, 'flash': 1.0, 'nightshot': 1.0}
```

That call encoded 36 images once, fitted six linear models (five folds plus the final
head on everything), wrote `heads2/lighting.pt`, and scored the result against zero-shot
prompts. The head is now usable by name, from this process or any other pointed at the
same directory.

Ask with it like any other question. Heads live in the same `ask()` call as everything
else, against the same single encode:

```python
a = jev.ask("shots/daylight/11.png", {"lit": {"type": "head", "head": "lighting"}})["lit"]
print(a, a.probabilities, a.confidence)
```

```text
<head 'daylight' p=0.573> {'daylight': 0.573317, 'flash': 0.391864, 'nightshot': 0.034819} 0.359975
```

(`shots/daylight/00.png` is the one the held-out number above is missing: it is the
dimmest, most neutral of the twelve and the head reads it as `flash` at p=0.452. One
image in 36, which is what 97.2% is made of.)

A `head` answer carries `choice`, `probabilities`, `confidence`, and two fields the other
question types do not have: `trained_on` (how many examples fitted it) and `accuracy`
(the held-out number below). They are there so a downstream service can see how much to
trust the distribution without going to look the head up.

One caveat on temperature: head logits are divided by the `temperature=` you pass to
`ask()`, and by **1.0** when you pass nothing. They do not use the instance temperature
that `choice` and `score` questions use, because a head's logits are fitted, not cosines,
and the encoder's calibration has nothing to say about them.

## The other two input shapes

`train()` takes a folder, a mapping, or a list of pairs. All three end up in the same
place; use whichever you already have.

```python
jev.train("lighting", "shots/")                                  # folder

jev.train("lighting", {"daylight": [img1, img2, ...],            # {label: [images]}
                       "flash":    ["a.png", b"\x89PNG...", ...]})

jev.train("lighting", [(img1, "daylight"), (img2, "flash")])     # (image, label) pairs
```

An "image" is anything [`to_image()`](/docs/reference-python) accepts: a path, a `Path`,
raw encoded bytes, a `PIL.Image`, or a `data:image/...;base64,...` string. Labels are
passed through `str()`, so integer labels become `"0"`, `"1"`.

`max_per_class=N` must be a positive integer (or `None`), and applies to all three
forms: it keeps the first N *per label* in the
order you gave them: for a folder, the sorted filename order above; for a mapping or a
pair list, the order of the list. It is applied before anything is encoded, so it is also
how you keep a first experiment cheap:

```python
small = jev.train("lighting_small", "shots/", max_per_class=4)
print(small, small.counts)
```

```text
<Head 'lighting_small' 3 classes, 12 examples, 100.0% held out> {'daylight': 4, 'flash': 4, 'nightshot': 4}
```

Twelve examples instead of 36, and `k` drops from 5 to 4 with the thinnest class. The
held-out accuracy goes *up*, to 100%, which is the warning and not the reward: the four
images kept per class are the first four in sorted order, and on this fixture they are
the ones furthest from the boundary, so the hard one the full head misses is no longer in
the data. With 4 examples per class each fold holds out one image, so the number can only
ever read 0, 0.25, 0.5, 0.75 or 1.0 per class. Small `n` makes both the model and the
estimate of it worse, and the estimate gets coarse first.

## What `accuracy` means

`head.accuracy` is **stratified k-fold cross-validation accuracy**. It is never the
accuracy on the images the head was fitted to. A linear head scored on its own training
rows will report something close to 100% on almost any data, including data with no signal
in it at all.

The arithmetic, from `jevimage/training.py`:

- `k = min(5, size of the thinnest class)`. Three classes of 12 gives `k = 5`; add one
  class with 3 examples and every fold count drops to 3.
- Folds are stratified (each class is split across the folds separately, so no fold is
  missing a class) and the shuffle is seeded per class index, so retraining the same
  data reports the same number rather than a different one each time.
- For each fold: fit on the other `k-1` folds, predict the held-out fold. The overall
  accuracy is every held-out prediction pooled; `per_class` is the fraction of each
  class's examples that were correct when held out (recall, not precision).
- Then one final fit on **all** the examples. That last fit is the head you ship. The
  accuracy describes a model trained on `(k-1)/k` of the data, which is a slightly
  pessimistic estimate of the one you actually got, which is the direction you want an
  estimate to be wrong in.

`head.to_dict()["folds"]` records the `k` that was used. It is one of the four record
fields with no attribute of its own, so there is no `head.folds`. Read the accuracy next
to `counts`:

```text
{'daylight': 12, 'flash': 12, 'nightshot': 12}   -> k=5, each fold holds out ~7 images
{'daylight': 12, 'flash': 12, 'green': 2}        -> k=2, each fold holds out 1 of the 2
```

With a class of 2, that class's "accuracy" is one of `0.0`, `0.5`, `1.0`. It is a coin
toss with two flips. It is still better than the fit accuracy, and it is still nearly
meaningless, which is why `counts` sits next to it in every printout the library
produces.

### How much data

There is no threshold the library enforces beyond "at least 2 per class", and no honest
universal number. What there is:

- Below ~5 per class, `k` drops with your thinnest class and the accuracy estimate is
  too noisy to compare against anything.
- The real signal is the comparison in the next section. If a class is below zero-shot
  with 10 examples, more examples of it is the cheapest thing to try.
- The ceiling is `MAX_EXAMPLES = 4000` in one `train()` call.

### Retraining is deterministic

```python
again = jev.train("lighting", "shots/")      # again, same data
print("same accuracy:", again.accuracy == head.accuracy,
      "| same per class:", again.per_class == head.per_class)
```

```text
same accuracy: True | same per class: True
```

Same folds, same fit, same number. A changed accuracy after a retrain means the data
changed.

## Did training actually help?

`train()` answers this for you and `compare()` re-answers it later. Both score the
trained head against what the same encoder would have said from prompts alone:
`"a photo of {class}"`, with underscores in class names turned into spaces.

```python
import json
print(json.dumps(head.comparison, indent=2))
```

```json
{
  "overall": {
    "zero_shot": 0.3333,
    "trained": 0.9722222222222222,
    "delta": 0.6389
  },
  "per_class": [
    {"label": "daylight",  "n": 12, "zero_shot": 1.0, "trained": 0.9167},
    {"label": "flash",     "n": 12, "zero_shot": 0.0, "trained": 1.0},
    {"label": "nightshot", "n": 12, "zero_shot": 0.0, "trained": 1.0}
  ],
  "basis": "trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example"
}
```

Read `per_class`, not `overall`. The overall row here says +64 points, which sounds like
a triumph and hides what happened: the prompts `"a photo of daylight"`,
`"a photo of flash"` and `"a photo of nightshot"` mean nothing to this encoder, so
zero-shot assigned every image to the same class and scored 100% on that class and 0% on
the other two. Overall accuracy averages that into one number. The per-class rows show
the shape of it.

The two sides are not scored the same way, and `basis` says so in the payload: the
trained side is held out, while the zero-shot side has nothing to hold out (there was no
fitting) and so is scored on every example. That asymmetry favours zero-shot slightly,
which is the right way round: the baseline should not be handicapped.

`compare()` works on any saved head, any time, because the head file keeps the
embeddings it was fitted on:

```python
jev.compare("lighting")                              # same shape as head.comparison
jev.compare("lighting", template="a {} photograph")  # try better prompts, local only
```

The `template=` argument is the useful one: if you think the baseline was unfair, change
the prompts and re-score without retraining anything. (`Remote.compare(name)` takes no
template; the server always uses the default.)

### A case where training makes a class worse

Same encoder, a different folder (`colours/`, also written by
[`examples/make_fixtures.py`](https://github.com/jevimage/jevimage/blob/main/examples/make_fixtures.py)): 12 `red`, 12 `blue`, and a
`green` class with exactly two images, one yellow-green and one cyan-green. The class
name is a word the encoder understands, so zero-shot handles it perfectly. The head has
one green example to learn from per fold - and the two greens are far enough apart that
the one it saw does not predict the one it is scored on.

```console
$ jev train colour colours/
'colour': 26 examples, 3 classes, encoder tint

class   n  zero-shot  trained    delta
blue   12     100.0%   100.0%     0.0%
green   2     100.0%     0.0%  -100.0%
red    12     100.0%   100.0%     0.0%
ALL    26     100.0%    92.3%    -7.7%

trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example.
Training did not beat the prompts here - try more examples per class, or a bigger encoder.
```

Nothing failed. The head fitted, saved and works. It is worse than three prompts, and the
honest move is to delete it and keep the prompts, or to go and label more green images,
which is what the per-class row is telling you to do.

This is the normal case for classes that already have good names. Training pays where
your labels are *not* describable: internal categories, part numbers, pass/fail criteria,
anything where `"a photo of {label}"` is a sentence nobody would write.

> The `jev train` output above was produced with the stand-in encoder registered under
> the name `tint` by a `sitecustomize.py`, which is the only way the CLI ever sees a
> custom encoder (see [reference-cli.md](/docs/reference-cli#custom-encoders-and-the-cli));
> the real command line is `jev train colour colours/` with whatever encoder is your
> default.

The table is `head.comparison` and nothing else; the closing line appears whenever
`overall.delta <= 0`. A `-` in any column means "not measured", never zero.

### The same thing measured on a real encoder

Every number on this page came from the stand-in encoder. These are the library author's
measurements on a 7-class, 350-image facial expression set under a real SigLIP, kept
here because they are the only record of what this looks like at realistic scale. They
were not re-run for these docs, and they are one dataset, not a benchmark.

- Overall, training was worth about +30 points over the zero-shot prompts.
- `happy` went from 6% zero-shot to 92% trained. `surprise` barely moved. `disgust` went
  from 82% down to 22%. The prompt `"a photo of disgust"` happened to be an excellent
  detector on that set, and the head, forced to carve seven regions out of one embedding
  space, gave that ground away. The overall row would have told you to ship it and said
  nothing about `disgust`.
- Fitting took 8.6 seconds end to end, including encoding all 350 images once. The head
  was 21 KB of weights in a 549 KB file, the difference being the stored embeddings
  described under [What is in the file](#what-is-in-the-file-and-how-big-it-is).
- Passing `template="a photo of a person who is {}"` moved the zero-shot column from
  22.3% to 26.9%. The trained column did not move, because the template is the baseline's
  prompt and has nothing to do with the head. If your labels are not English captions,
  the default `"a photo of {}"` understates what prompts alone could do, and the delta you
  read is flattering.

## The `Head` object

`train()`, `head()` and the remote versions of both return a `jevimage.Head`. It is
metadata, not weights: the same object either side of the wire, because weights stay
wherever the encoder is.

| Attribute | What it is |
|---|---|
| `head.name` | The name you trained it under, which is also its filename stem. |
| `head.classes` | Class names, **sorted**, in the column order of `W`. |
| `head.n` | Total examples fitted. |
| `head.counts` | `{label: count}` — read this next to `accuracy`. |
| `head.accuracy` | Held-out k-fold accuracy — `cv_accuracy` in the dict. `None` only for a head file that never recorded one; every head this version writes has one, because `train()` refuses a class with fewer than 2 examples and so `k` is always at least 2. |
| `head.per_class` | `{label: held-out accuracy}` for that class. |
| `head.encoder` | The encoder name that fitted it. Checked at query time. |
| `head.comparison` | The zero-shot comparison, or `None` when it was not computed. |
| `head.to_dict()` | The full public record. |

`to_dict()` carries four fields with no attribute of their own:

```json
{
  "name": "lighting",
  "encoder": "tint",
  "classes": ["daylight", "flash", "nightshot"],
  "n": 36,
  "counts": {"daylight": 12, "flash": 12, "nightshot": 12},
  "cv_accuracy": 0.9722222222222222,
  "per_class": {"daylight": 0.9167, "flash": 1.0, "nightshot": 1.0},
  "folds": 5,
  "created": 1790181634.5223048,
  "dim": 3,
  "bytes": 48
}
```

`folds` is the `k` used, `created` is a Unix timestamp, `dim` is the encoder width the
head expects, and `bytes` is the size of `W` and `b` in memory, **not** the size of the
file on disk, which is larger (see below). `accuracy` is `cv_accuracy` under its shorter
name; the dict uses the longer one because that is what the HTTP API returns.

Note that `to_dict()` never contains `W`. Weights do not cross the wire and are not part
of the public record. Locally they are still reachable as `jev.heads["lighting"]["W"]` if
you want them.

## Where heads are stored

One file per head, in a directory you can point anywhere:

```text
$JEV_HEADS_DIR          if set — wins outright
else $JEV_HOME/heads    if JEV_HOME is set
else ~/.jev/heads
```

`jevimage.load(heads_dir=...)`, `jev serve --heads-dir ...` and `jevimage.Jev(...)`
override it for one instance. `jev heads` prints the directory it read:

```console
$ JEV_HEADS_DIR=heads2 jev heads
head            encoder  classes   n  held out     created
lighting           tint        3  36     97.2%  2026-09-23
lighting_small     tint        3  12    100.0%  2026-09-23

heads2
```

The mechanics decide what you can safely do to the directory:

- **One file per head, named `<name>.pt`.** So a head can be copied, committed or deleted
  on its own, and two processes training different heads never contend.

  ```console
  $ ls -l heads2
  total 8
  -rw-rw-r-- 1 peroxide peroxide 2693 Sep 23 22:13 lighting.pt
  -rw-rw-r-- 1 peroxide peroxide 2555 Sep 23 22:15 lighting_small.pt
  ```

- **Writes are atomic.** The file is written to `<name>.pt.tmp` and `os.replace`d into
  position. A crash mid-training cannot leave a torn head behind.
- **Reads are `weights_only=True`.** A head is tensors plus plain metadata, so a file a
  colleague sent you cannot execute code on the way in.
- **The directory is re-read when it changes.** A head trained by another process, or
  copied in by hand, is visible without a restart. Dropping a file into a running
  server's heads directory is enough:

```console
$ curl -s localhost:8177/v1/heads | python3 -c "import json,sys;print([h['name'] for h in json.load(sys.stdin)])"
['lighting']
$ cp heads/colour.pt srvheads/
$ curl -s localhost:8177/v1/heads | python3 -c "import json,sys;print([h['name'] for h in json.load(sys.stdin)])"
['colour', 'lighting']
```

- **A file that will not load is reported, not hidden.** The table lists the heads that
  loaded; the reason the other one did not is printed beside it, because "no head named
  x" about a file sitting right there is a lie:

```console
$ jev heads
junk.pt is present but unreadable: UnpicklingError: Weights only load failed. In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.
head    encoder  classes   n  held out     created
colour     tint        3  26     92.3%  2026-09-23
```

(The first line goes to stderr. `jev heads --json` prints no warning at all; it carries
the same information in its `"broken"` object, so a script reading only stdout can see
it.)

### What is in the file, and how big it is

A head file holds `W`, `b`, the metadata above, **and the fp16 embeddings of every
training image** with their label indices. The embeddings are what makes `compare()`
answerable months later, and they dominate the size. Measured on a head fitted to 1000
examples of a 768-dim encoder with 5 classes, on synthetic embeddings, since the file
layout is what is being measured and not a model:

```text
file: 1555754 bytes;  public()['bytes'] = 15380
```

15 KB of weights, 1.5 MB of stored embeddings. Two consequences:

- Budget by `n * dim * 2` bytes, not by the weight size, when you are shipping heads
  around or holding many of them in a server.
- **The embeddings of your training images travel with the head.** They are not the
  images and they cannot be turned back into them, but they are a derived representation
  of your data. If that matters for the images you trained on, treat the head file with
  the same care as the folder.

A head stripped of its embeddings still answers questions; it just cannot be compared any
more. `compare()` returns `None` for it locally, and the HTTP route answers `409` with a
message saying to retrain.

### Backing one up, or sending it to someone

It is a file. Copy it.

```console
$ cp ~/.jev/heads/lighting.pt ./heads-backup/          # backup
$ scp ~/.jev/heads/lighting.pt colleague:~/.jev/heads/ # hand it over
```

The receiving side needs one thing to be true: **the same encoder**, by name. The head
records which encoder fitted it and refuses to run under any other one. Send the encoder
name along with the file.

In a container, put the file where the process will look and set nothing else:

```dockerfile
FROM python:3.12-slim
RUN pip install 'jevimage[serve]'
COPY heads/lighting.pt /srv/heads/
ENV JEV_HEADS_DIR=/srv/heads
ENV JEV_ENCODER=siglip2-base-224
CMD ["jev", "serve", "--host", "0.0.0.0"]
```

Nothing has to be registered, imported or migrated; the store picks up whatever `.pt`
files are in the directory at startup and whatever appears later.

## Training through a server

`connect()` mirrors `load()`, and that includes training. The encoder lives on the
server, so an install with **no torch at all** can fit a head:

```python
import jevimage

jev = jevimage.connect("http://127.0.0.1:8177", timeout=600)
head = jev.train("lighting", "shots/")

print(head)
print(head.encoder, head.counts)
print(head.comparison["overall"])
print([h["name"] for h in jev.list_heads()])
```

```text
<Head 'lighting' 3 classes, 36 examples, 97.2% held out>
tint {'daylight': 12, 'flash': 12, 'nightshot': 12}
{'zero_shot': 0.3333, 'trained': 0.9722222222222222, 'delta': 0.6389}
['lighting']
```

That is the same `Head`, with the same numbers, as the local run at the top of this page,
from an interpreter where `import torch` fails. The differences to know about:

- **Your images are uploaded**, one `data:` URL per example, in a single POST. A JPEG, PNG
  or WebP goes up byte-for-byte; anything else is re-encoded to PNG first. A thousand photos
  is a thousand photos of upload.
- **`timeout=` matters.** The default is 60 seconds for the whole request, which covers
  the upload, the encode of every image and `k+1` model fits. Training a real set will
  exceed it; pass a larger `timeout` to `connect()`. The error when you do not is
  explicit, and it names the fix.
- **The head lives on the server**, in *its* heads directory, under whatever encoder it
  runs. `jev.health()` tells you which encoder that is before you spend the upload.
- **`max_per_class=` is applied on your side**, before anything is uploaded.
- **The name is validated before the upload**, as it is locally before anything is
  encoded, so a typo costs nothing.
- The comparison comes back with the response; `jev.compare(name)` re-runs it later.

From the shell it is the same command with `--url`, and the client's 60-second default
applies there too. This 36-image set goes through it in about a second, but a real
dataset, or a server still loading its weights, runs past 60 seconds, and then the
command fails:

```console
$ jev train lighting_cli shots/ --url http://127.0.0.1:8278
jev: POST http://127.0.0.1:8278/v1/heads did not answer within 60.0s; raise timeout= if the server is loading a model
$ echo $?
1
```

(Produced against a listener that accepts the connection and never answers, which is what
a too-slow server looks like from here.) The knob the message asks for is on the command
too: `jev train --timeout SECONDS` (also `$JEV_TIMEOUT`, and the same flag on `jev ask`)
is `connect(url, timeout=...)` from the shell, so a big folder trains either way. The
other commands are quick and need none of it:

```console
$ jev heads --url http://127.0.0.1:8177
head      encoder  classes   n  held out     created
colour       tint        3  26     92.3%  2026-09-23
lighting     tint        3  36     97.2%  2026-09-23

http://127.0.0.1:8177
```

`--url` also reads `$JEV_URL`, with `$JEV_API_KEY` for a server started with keys.

See [Serving](/docs/serving) for running the server end, and
[HTTP reference](/docs/reference-http) for the raw `POST /v1/heads` shape.

## Limits

Read from the source, not guessed. `jevimage/training.py`:

| Limit | Value | What hits it |
|---|---|---|
| `MAX_EXAMPLES` | 4000 | `ValueError: at most 4000 examples; got 4001`, raised **before** anything is encoded. |
| `MAX_CLASSES` | 100 | `ValueError: at most 100 classes`. |
| `FOLDS` | 5 | The ceiling on `k`; the thinnest class lowers it. |
| minimum per class | 2 | Below it, `train()` refuses and names the classes. |
| minimum classes | 2 | One label is not a classifier. |

From `jevimage/store.py`. Head names become filenames, so they are checked rather than
sanitised: `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`. 1 to 64 characters, starting with a
letter or digit. A silently rewritten name would mean `ask()` could not find what
`train()` said it saved.

From `jevimage/server.py`, for training over HTTP: `epochs` is bounded to `1..5000` by the
request model (5000 epochs on 4000 examples is seconds, 5 million is an outage); each
uploaded image must be under 12 MB of base64 and 50 megapixels and be JPEG, PNG or WebP.
The `MAX_IMAGES = 8` cap on `/v1/ask` and `/v1/embed` deliberately does **not** apply to
`/v1/heads`; training is a bulk upload, bounded by `MAX_EXAMPLES` instead.

`epochs=` on the local `train()` has no upper bound. It is a loop count; 500 is the
default and there is rarely a reason to move it. The floor is shared with the server:
fewer than 1 epoch is refused, because a head fitted in zero steps is all zeros and
would still be saved, listed and answered from.

## Retraining, and what invalidates a head

Training under a name that already exists replaces it. The write is atomic, so a reader
sees either the old head or the new one, never a half-written file. There is no version
history, so if you want the old head, copy the file before you retrain.

A head stops being usable when:

| Cause | What you see | Fix |
|---|---|---|
| Different encoder | `head 'colour' was trained on encoder 'tint' but this instance runs 'tint-v2'. Load 'tint', or retrain the head here.` | Load the named encoder, or retrain. |
| Encoder width changed | `head 'x' expects 768-dim embeddings, this encoder produces 1152` | Retrain. |
| Damaged file | `head 'x' lists 3 classes but its weights have 2 columns; the file is damaged, retrain it` | Retrain. |
| No encoder recorded | `head 'x' records no encoder, so it cannot be checked against this one; retrain it` | Retrain. A head with no provenance is refused rather than run unchecked. |
| Unreadable file | listed by `jev heads` with the loader's reason, and `ask()` raises `KeyError: x.pt is present in <dir> but unreadable: <reason>. Delete it, or retrain the head.` | Replace the file. |

The check is on the encoder **name**, not on its weights. If you `register()` a
different model under a name you have used before, or an ensemble's members change while
its name does not, an old head will load and answer confidently in an embedding space it
was never fitted to. Nothing can detect that. Change the name when you change the model.

Changing your labels (renaming a class, adding one, merging two) is a retrain. The head
has one column per class, fixed at fit time.

## Failure modes

Each of these was produced by running it.

```text
ValueError: need at least 2 distinct labels
```
One class. A classifier over one class has nothing to decide.

```text
ValueError: every class needs at least 2 examples; too few for: blue
```
Cross-validation needs to hold something out. The message names every thin class, not
just the first.

```text
ValueError: labels ['red', 'red '] differ only by surrounding whitespace, so they would print identically; make them distinct
```
Two sub-folders whose names differ by a trailing space, or a trailing space in a mapping
key. Without this check you would see one class scoring half of what it should, in a
table where the two rows look identical.

```text
ValueError: no labelled images under 'colours/red'. Expected one sub-folder per label, each holding .jpeg/.jpg/.png/.webp files.
```
You pointed at a class folder instead of the dataset root, or the images are a format
that is not read.

```text
ValueError: invalid head name 'bad name!': use 1-64 characters from [A-Za-z0-9._-], starting with a letter or digit
```
Raised before anything is encoded, on both sides: `train()` checks the name first, and
`Remote.train()` checks it before it uploads anything. A typo'd name costs one call, not
a thousand images of encoding.

```text
JevError: POST http://127.0.0.1:8177/v1/heads did not answer within 60.0s; raise timeout= if the server is loading a model
```
Remote training that outran the client's default timeout. Pass `timeout=` to `connect()`,
or `--timeout` to `jev train`.

And the failure that raises nothing at all: a head that fits, saves, works, and is worse
than the prompts it replaced. That one only shows up in the comparison, which is why it
is printed at the end of every `jev train`.

## The stand-in encoder used on this page

For reproducing the outputs above without downloading any weights. It is a colour meter
with a text side, sufficient to exercise every code path on this page.

```python
# tint.py
import torch, jevimage

WORDS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1),
         "yellow": (1, 1, 0), "cyan": (0, 1, 1), "magenta": (1, 0, 1),
         "white": (1, 1, 1), "black": (.2, .2, .2)}


class Tint(jevimage.Encoder):
    name, dim = "tint", 3

    def img(self, ims):
        v = torch.tensor([[sum(c) / len(c) for c in zip(*im.getdata())] for im in ims])
        return torch.nn.functional.normalize(v, dim=-1)

    def txt(self, ts):
        v = torch.tensor([next((WORDS[w] for w in t.lower().split() if w in WORDS),
                               (1., 1., 1.)) for t in ts])
        return torch.nn.functional.normalize(v.float(), dim=-1)


jevimage.register("tint", Tint)     # usable from Python as jevimage.load("tint");
                                   # `jev --encoder tint` only sees it if a
                                   # sitecustomize.py runs this at startup
                                   # (see reference-cli.md#custom-encoders-and-the-cli)
```

---

Next: [Question types](/docs/questions) for the other three things you can ask ·
[Choosing an encoder](/docs/encoders) when a head cannot beat the prompts ·
[Serving](/docs/serving) to train through a server you run ·
[Python reference](/docs/reference-python) for exact signatures.

---

[← Question types](/docs/questions) · [Docs index](/docs/index) · [Choosing an encoder →](/docs/encoders)
