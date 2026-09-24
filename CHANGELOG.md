# Changelog

## Unreleased

Local SDK fixes - input that was accepted and then failed somewhere else, and three
things that were wrong rather than unclear:

- **`batch=True` no longer fails a whole batch for one bad image.** One caller's
  truncated upload answered every other caller who rode the same forward pass with an
  error about a file they never sent - on a server, up to 15 healthy requests at a time.
  A failing batch is now re-run one item at a time, so each caller gets its own result
  or its own error.
- **A head retrained by another process is picked up by `head()` and `ask()`,** not only
  by `list_heads()`. The freshness check ran on a cache MISS, so once a name was cached
  every later read served the pre-retrain weights and class names, with no error.
  Deleting a head propagates the same way.
- **`HeadStore` is thread-safe for reads again.** Two threads syncing at once both
  removed the same vanished head, and the loser raised `KeyError` out of `name in store`
  - the check that gates every `head` question.
- Two processes saving the same head name no longer collide: the temp file is per
  writer, so the loser is overwritten instead of failing with a `FileNotFoundError`
  about a `.pt.tmp` it never named. Retraining a broken head clears it from
  `heads.broken`.
- `load()` refuses to rename an encoder instance that is already cached under another
  registry name, instead of silently renaming it and invalidating every head trained
  under the first name. `register()` also drops cached ensembles built from that name.
- `Ensemble` normalises each member before concatenating, as `docs/encoders.md` always
  said it did. Without it a member whose output is not unit-norm outvoted the others,
  and the engine's L2 check could not see it.
- SigLIP entries truncate a caption to the tokeniser's 64 tokens instead of failing with
  a transformers error about `max_position_embeddings`.
- A `prompt` or `template` is validated by trial substitution, so `'{} and {}'`,
  `'a photo of {x}'` and `'{.__class__}'` are refused by name instead of escaping as a
  raw `IndexError`/`KeyError`, or captioning every option `"<class 'str'>"`. `train()`
  and `compare()` check `template=` before any work, so a bad one no longer raises
  *after* the head is on disk.
- `ask_embedding()` checks what it was handed: a numpy array, a list, a stray batch
  dimension or the wrong width are named, not left to fail inside the engine.
- `max_per_class` means one thing in all three `data` shapes: a positive int, or `None`.
  `0` used to mean "no limit", `-1` dropped different examples in each shape, and a
  folder was truncated twice - which could delete a whole class.
- `to_image()` treats `Path` exactly like `str`, truncates what it quotes back, reports a
  malformed `data:` URL as a `ValueError` naming the URL, and accepts `DATA:`.
  `read_folder` sorts by filename, not by full path, which is what the docs promised
  once sub-directories are involved.
- `temperature` accepts any real number (numpy scalars included) in `1e-6 .. 1e6`, and
  says so. Outside that range fp32 underflowed and the reported winner could flip. An
  encoder's `logit_scale` goes through the same guard, so a negative one is an error
  rather than a confident answer on the wrong option.
- The singular `instruction` is read again as a spelling of `instructions`, which is
  what `docs/questions.md` documents. (0.1.0 removed it as "undocumented"; the docs row
  says otherwise, so the docs win.)
- `jevimage[local]` requires `transformers>=4.56`: `metaclip2-huge` is `model_type
  metaclip_2`, which older versions do not recognise, so the old floor allowed an
  install that could not load a listed encoder.
- `dfn5b-h-14-384` names the extra it needs when `open_clip` is missing.

Client fixes, all in `jevimage.connect()`:

- `max_edge=` is now actually used. It was stored and never passed to the upload, so
  every image was downscaled to 384px whatever you asked for, and `None` did nothing.
- An oversized image is exif-transposed before it is re-encoded for upload. A portrait
  phone photo used to reach the server sideways, with its Orientation tag dropped by the
  re-encode, so remote answers disagreed with local ones.
- A `POST` (`ask`, `embed`, `train`) is no longer silently replayed when a kept
  connection dies. The failure can surface after the server has done the work, so a
  `train()` could run twice. `GET` and `DELETE` still reconnect once.
- `api_key` is validated in `connect()`: a trailing newline or a non-ASCII character
  used to blow up inside `http.client`, with the key itself in the traceback.
- `Remote.ask` gives `Jev.ask`'s wording for a bad temperature, `Remote.train` refuses
  an out-of-range `epochs` (the server's 1..5000) with a sentence instead of a pydantic
  field dump, a 2xx that is not JSON is a `JevError` naming the URL, and a redirect
  leads with where it pointed instead of the proxy's HTML page.

Docs:

- **`examples/make_fixtures.py` ships the images the docs read.** Every page's
  transcripts were run over `ds/`, `shots/` and `colours/` folders that were never in
  the repo, so no block on any page ran as written, and the stand-in encoders key off
  raw pixels, so no number could be reproduced either. The script writes all three, and
  every printed number on quickstart, questions, training, cookbook, encoders, the three
  references, index and README was re-run against them and re-pasted.
- Blocks that could not run as printed now run: `import json` in the quickstart's first
  block, `ToyCalibrated` in the questions prelude, `import base64` and `import tempfile`
  where the reference and encoders pages use them, a `uvicorn` line for the mounted-app
  transcript in serving, and `jevimage.load(Toy())` where troubleshooting said
  `jevimage.load("toy")`, which is not a registry name.
- `jevimage.encoders` (and any other submodule) resolves after a plain `import
  jevimage`, so `troubleshooting.md`'s ensemble-name example and the advice in the
  engine's own "wrap your encoder's output in `jevimage.encoders.unit()`" both work.
- Claims corrected against measurement: warm `load()` of `siglip2-base-224` is 8-10 s on
  the 16-thread laptop CPU these docs are measured on, not 4.8/5.0 s; 5000 epochs on
  4000 examples is about 70 s, not "seconds"; 16 concurrent single-image embeds coalesce
  into 2 forward passes on the toy encoder rather than staying 16; several keys may be
  configured so callers can migrate, but withdrawing one is a restart, not zero
  downtime; and N questions are only near-free once their captions are cached.

## 0.1.0

First release.

- `choice`, `score`, `noul` and `head` questions against one image encode.
- `jevimage.load()` for in-process use, `jevimage.connect(url)` for a server. Same
  methods, same return shapes.
- `jev` CLI: `ask`, `train`, `heads`, `rm`, `encoders`, `serve`. The first four take
  `--url`.
- Linear heads trained on a labelled folder, with a per-class comparison against the
  zero-shot prompts.
- Seven built-in encoders plus `'+'` ensembles, and `register()` for your own.

Two notes for anyone who tracked the pre-release source:

- `plan()` no longer accepts the singular `instruction` key as an alias for
  `instructions`. It was undocumented; the singular spelling now raises.
- `Encoder` has no `preprocess()` / `img_from_px()` pair. Nothing called it, and the
  cross-request batching it was meant to enable was never wired up. `Batcher` batches
  whole images through `img()`.
