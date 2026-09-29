---
title: "HTTP API"
description: "Every /v1 route with its request and response shape, status codes and a real curl."
---

Every route under `/v1` except `/v1/health` is one method on `jevimage.Jev`, with the same
arguments and the same return shape (a local `Jev` has no server to report on, so it has
no `health()`). `POST /v1/ask` is `ask()`, `POST /v1/embed` is `embed()`,
`/v1/heads` is `train()` / `list_heads()` / `delete_head()`. There are no HTTP-only
concepts to learn, and `jevimage.connect(url)` is a client that already speaks all of it;
see [Python API reference](reference-python.md).

Everything is JSON in, JSON out. Images arrive as base64: either bare, or as a
`data:image/png;base64,...` URL (the server strips the prefix and does not check that it
matches the actual bytes; the format is decided by the decoded content).

Start a server with `jevimage serve`; see [Serving](serving.md) for deployment. A quick one:

```bash
pip install 'jevimage[serve]'
jevimage serve --port 8731 --api-key s3cret
```

## The server these examples run against

Every request and response below was executed against a real `jevimage.server` app. The
encoder is a deterministic 16-dimension toy (a hash projected through a fixed random
matrix), not SigLIP, so no weights had to be downloaded. **The numbers in the answers are
therefore meaningless as classification results.** The shapes, field names, status codes
and error messages are the real thing. A real encoder produces 768-dim or larger
embeddings and answers that mean something; nothing else about the wire changes.

Three servers were used:

| Base URL | Encoder | Auth |
|---|---|---|
| `http://127.0.0.1:8123` | `toy` (no `logit_scale`/`logit_bias`) | open |
| `http://127.0.0.1:8126` | `toy` | `Authorization: Bearer s3cret` |
| `http://127.0.0.1:8127` | `toy-calibrated` (`logit_scale=100`, `logit_bias=-10`) | `Authorization: Bearer s3cret` |

Images in the examples are the flat 64x64 PNGs of the `ds/` fixture, written by
[`examples/make_fixtures.py`](https://github.com/Per0x1de-1337/jevimage/blob/main/examples/make_fixtures.py); a flat square compresses to
a couple of hundred bytes, which is why the base64 is short enough to paste.
The reusable recipe is:

```bash
IMG=$(base64 -w0 photo.png)
curl -s http://127.0.0.1:8731/v1/ask -H 'Content-Type: application/json' \
     -d "{\"image\":\"$IMG\",\"questions\":{...}}"
```

## Authentication

A single shared secret in the standard bearer form:

```
Authorization: Bearer <key>
```

The scheme token is matched case-insensitively, as RFC 7235 §2.1 requires, and any run of
whitespace separates it from the key: `bearer <key>` and `BEARER\t<key>` work too. The key
itself is compared exactly.

The server accepts any one of the keys it was started with. Keys come from
`jevimage serve --api-key K1 K2`, from `$JEVIMAGE_API_KEYS` (split on commas *or* whitespace), or
from `build_app(jev, api_keys=[...])`. With no keys configured, every route is open and
`GET /v1/health` reports `"auth": "open"`.

Things worth knowing before you put this on a network:

- **`GET /v1/health` is never authenticated**, even when keys are set. It is registered
  outside the authenticated router on purpose: a load balancer that needs a secret to ask
  whether a process is alive will eventually report that it is not. Health does disclose
  the encoder name, dimension, device, version and how many heads exist.
- Keys are compared with `hmac.compare_digest` against the raw header bytes, so a
  non-ASCII key is a 401 rather than a 500.
- There is no identity, no per-key scoping, no rate limiting and no audit trail. Any key
  can train, read and delete any head. If you need more than "one shared secret in front
  of a trusted network", put a proxy in front of it.

```bash
curl -s -w ' [%{http_code}]' http://127.0.0.1:8126/v1/heads
```
```json
{"detail":"send 'Authorization: Bearer <key>' with a key this server was started with (JEVIMAGE_API_KEYS)"} [401]
```

The same 401 comes back for a wrong key and for a header without the `Bearer ` prefix.
The server does not distinguish, because the caller's fix is identical.

```bash
curl -s -w ' [%{http_code}]' -H 'Authorization: Bearer s3cret' http://127.0.0.1:8126/v1/heads
```
```json
[] [200]
```

## Content limits

All of these are enforced in `jevimage/server.py` and `jevimage/core.py`; they are
constants, not configuration.

| Limit | Value | Applies to | Status when exceeded |
|---|---|---|---|
| Images per request | 8 | `/v1/ask`, `/v1/embed` | 413 |
| Examples per request | 4000 | `POST /v1/heads` | 413 |
| Base64 length per image | 12 MB | every image field | 413 |
| Decoded pixels per image | 50 MP | every image field | 413 |
| Decoded pixels per request | 100 MP | every image field | 413 |
| Accepted formats | JPEG, PNG, WebP | every image field | 415 |
| Questions per `ask` | 64 | `/v1/ask` | 422 |
| Options in one `choice` | 2–255 | `/v1/ask` | 422 |
| Levels in one `score` | 2–10 | `/v1/ask` | 422 |
| Classes in one head | 2–100 | `POST /v1/heads` | 422 |
| Head name | `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}` | `POST /v1/heads`, `DELETE /v1/heads/{name}` | 422 |
| `epochs` | 1–5000 | `POST /v1/heads` | 422 |

`GET /v1/heads/{name}` and `/compare` do not check the name: the store lookup misses
first, so an illegal name there is a 404, not a 422.

The base64 length is checked *before* decoding, and the pixel count is read from the
image header *before* the pixels are allocated. A solid-colour 12000x12000 PNG is well
under a megabyte on the wire and over a gigabyte in RAM, so compressed size is no bound
on decoded size.

Nothing caps the total request body: the byte limits are per image. Eight images near
the 12 MB ceiling is a request of roughly 96 MB that no check in the server rejects, so
cap it at your proxy if that matters.

Decoded pixels *are* capped across the request as well as per image: 100 MP in total,
which still admits eight 12 MP camera photos. A proxy body cap cannot bound this - eight
solid-colour 50 MP PNGs are 1.7 MB on the wire and 1.2 GB of decoded pixels, and every
image in a request is decoded before any of them is embedded.

## Errors

Every error the application raises is a JSON object with one key:

```json
{"detail": "<a sentence saying what to do about it>"}
```

Two things sit outside that. A 500 is Starlette's own plain-text `Internal Server Error`,
with no JSON at all - any unhandled exception takes that path, and this package installs
no 500 handler - so a client that parses every error body as JSON must guard the one case
it most needs to log. And a request the body schema rejects outright has `detail` as
FastAPI's list of field reports instead of a string:

```bash
curl -s -w ' [%{http_code}]' -X POST http://127.0.0.1:8123/v1/ask \
     -H 'Content-Type: application/json' -d "{\"image\":\"$IMG\"}"
```
```json
{"detail":[{"type":"missing","loc":["body","questions"],"msg":"Field required","input":{"image":"iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAK0lEQVR4nO3NQQEAAATAQOTRP4VQSvC7BdjldMdn9XoHAAAAAAAAAAAAhy1j2QFkVm5ZOwAAAABJRU5ErkJggg=="}}]} [422]
```

Note that the echoed `input` contains your whole image. Do not log error bodies verbatim.

The engine's own refusals are the useful ones and they reach you intact. A `ValueError`
anywhere in the core becomes a 422 with its message; a `KeyError` (a head that is missing,
or a head file that is present but will not load) becomes a 404. "head 'x' was trained on
encoder 'y'" is an answer, not a symptom, so it is not flattened into
"422 Unprocessable Entity".

The 404 handler also strips any filesystem path out of the message on its way to the
client. The store names the directory it searched (the right thing at a REPL, the wrong
thing to hand anyone who can guess a head name), so over HTTP that part is removed.

| Status | Cause |
|---|---|
| 200 | Success. Every route returns 200, including `POST /v1/heads`. |
| 400 | The request body was not decodable as UTF-8 text. Starlette answers before the route, with `{"detail":"There was an error parsing the body"}`. |
| 401 | Keys are configured and the `Authorization` header was missing, malformed or wrong. Never on `/v1/health`. |
| 404 | No head by that name (`GET`/`DELETE /v1/heads/{name}`, `/compare`), a head file that will not load (those routes and a `head` question on `/v1/ask`), or an unrouted path (`{"detail":"Not Found"}`). |
| 405 | Wrong method on a real path, e.g. `POST /v1/health` → `{"detail":"Method Not Allowed"}`. |
| 409 | `GET /v1/heads/{name}/compare` on a head saved without its embeddings. |
| 413 | Too many images, too many examples, base64 over 12 MB, over 50 MP in one image or 100 MP across the request, or a decode PIL's own decompression-bomb guard refused. |
| 415 | An image that decoded but is not JPEG, PNG or WebP. |
| 422 | Anything the engine refuses: a malformed question, an unknown head, a bad temperature, a bad head name, a training set with one class - plus request-schema failures. |
| 500 | An unhandled server-side exception, e.g. the encoder running out of memory. The body is the plain string `Internal Server Error`, not JSON. |

---

## GET /v1/health

What this process is running. Unauthenticated. Cheap: it syncs the head directory and
returns counts, it does not touch the model.

**Response**

| Field | Type | Meaning |
|---|---|---|
| `encoder` | string | Encoder name. A head trained here records this name and refuses to run under any other. |
| `dim` | int | Embedding width. |
| `device` | string | `cuda`, `cpu`, … - whatever the encoder reports, `"cpu"` if it reports nothing. |
| `primitives` | string[] | Always `["choice", "score", "noul", "head"]` for this version. |
| `heads` | int | Heads currently on disk, after a re-scan. |
| `auth` | string | `"bearer"` if keys are configured, `"open"` if not. |
| `version` | string | `jevimage.__version__`, which is also the OpenAPI document's version. |

`encoder` is the field to read before you do anything else: it tells you which heads will
work here, and whether a single-caption `noul` is possible (it needs an encoder with both
`logit_scale` and `logit_bias`; SigLIP has them, CLIP and every ensemble do not).

```bash
curl -s http://127.0.0.1:8123/v1/health
```
```json
{"encoder":"toy","dim":16,"device":"cpu","primitives":["choice", "score", "noul", "head"],"heads":0,"auth":"open","version":"0.1.19"}
```

**Status:** 200 only.

---

## POST /v1/ask

One image encode, every question read against that same embedding. This is the route the
package exists for: N questions cost barely more than one *once their captions are
cached*, because the encoder runs once and each question is then a matmul. A caption this
process has not encoded before costs a text-tower forward pass first, so the first call
with a fresh question set is not cheap; see
[encoders.md#timings](encoders.md#timings).

**Request**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `image` | string | one of | One base64 image, bare or as a `data:` URL. |
| `images` | string[] | one of | Up to 8 base64 images, answered independently. |
| `questions` | object \| array | yes | `{id: question}`, or a list whose entries get the ids `q1`, `q2`, … (or the entry's own `"id"`). Max 64. |
| `temperature` | number \| null | no | Softmax temperature override for this request. Must be a number between 1e-6 and 1e6. Omit it unless you know why; the default is the encoder's calibrated scale. |

Send `image` **or** `images`, not both. Sending neither is a 422: `{"detail":"send
'image' (one base64 string) or 'images' (a list)"}`. Sending both is also a 422, because
the two fields select different response shapes and there is no way to guess which one
you meant:

```bash
curl -s -w ' [%{http_code}]' -X POST http://127.0.0.1:8123/v1/ask \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$RED\",\"images\":[\"$BLUE\"],\"questions\":{\"c\":{\"type\":\"choice\",\"criteria\":{\"red\":\"a red square\",\"blue\":\"a blue square\"}}}}"
```
```json
{"detail":"send 'image' or 'images', not both: 'image' returns one answer object, 'images' returns a list"} [422]
```

Question bodies are documented in [Question types](questions.md). Summarised:

| `type` | Required fields | Answer fields |
|---|---|---|
| `choice` | `criteria`: `{label: caption}` (2–255), or a list of captions | `type`, `choice`, `probabilities`, `confidence` |
| `score` | `criteria`: ordered array of 2–10 level captions, low end first | `type`, `score`, `probabilities`, `legend`, `confidence` |
| `noul` | `criteria`: `{"true": ..., "false": ...}`, **or** `instructions` with a calibrated encoder | `type`, `noul`, `basis` |
| `head` | `head`: the name of a trained head | `type`, `head`, `choice`, `probabilities`, `confidence`, `trained_on`, `accuracy` |

All four also accept `instructions` and `prompt` (a caption template that must contain
`{}`).

**Response**

| Field | Type | Meaning |
|---|---|---|
| `answers` | object \| array | `{question id: answer}` when you sent `image`; a list of those objects, one per image and in order, when you sent `images`. |
| `timing.embed_ms` | number | Milliseconds spent encoding the image(s). |
| `timing.total_ms` | number | Milliseconds for the whole call, encoding included. |

`total_ms - embed_ms` is what the questions cost. On this toy encoder the difference is
noise; on a real one, `embed_ms` is nearly all of it, which is the whole argument for
asking everything you need in one request.

```bash
curl -s -X POST http://127.0.0.1:8123/v1/ask -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"questions\":{\"colour\":{\"type\":\"choice\",\"criteria\":{\"red\":\"a red square\",\"blue\":\"a blue square\"}}}}"
```
```json
{"answers":{"colour":{"type":"choice","choice":"red","probabilities":{"red":1.0,"blue":0.0},"confidence":0.999999}},"timing":{"embed_ms":1.3,"total_ms":1.7}}
```

Two questions at once, with a `score` and the portable form of `noul`:

```bash
curl -s -X POST http://127.0.0.1:8123/v1/ask -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"temperature\":0.05,\"questions\":{\"busy\":{\"type\":\"score\",\"criteria\":[\"empty\",\"half full\",\"packed\"]},\"outside\":{\"type\":\"noul\",\"criteria\":{\"true\":\"outdoors\",\"false\":\"indoors\"}}}}"
```
```json
{"answers":{"busy":{"type":"score","score":1.1946,"probabilities":{"0":0.178673,"1":0.448013,"2":0.373314},"legend":{"0":"empty","1":"half full","2":"packed"},"confidence":0.172019},"outside":{"type":"noul","noul":0.352592,"basis":"contrast pair"}},"timing":{"embed_ms":0.6,"total_ms":0.9}}
```

`score` is the expected level index, not a class: `1.1946` on a three-level rubric sits
between "half full" and "packed" in probability mass. `legend` tells you what each index
was.

A list of questions and a list of images. Note the answers become a list, and the
questions pick up generated ids:

```bash
curl -s -X POST http://127.0.0.1:8123/v1/ask -H 'Content-Type: application/json' \
  -d "{\"images\":[\"$RED\",\"$BLUE\"],\"questions\":[{\"type\":\"choice\",\"criteria\":{\"red\":\"a red square\",\"blue\":\"a blue square\"}}]}"
```
```json
{"answers":[{"q1":{"type":"choice","choice":"red","probabilities":{"red":1.0,"blue":0.0},"confidence":0.999999}},{"q1":{"type":"choice","choice":"red","probabilities":{"red":0.99957,"blue":0.00043},"confidence":0.999141}}],"timing":{"embed_ms":0.5,"total_ms":0.8}}
```

(The toy encoder gets the second image wrong. A hash is not a vision model.)

**Status codes**

| Status | Cause |
|---|---|
| 200 | Answered. |
| 401 | Keys configured, header missing or wrong. |
| 404 | A `head` question naming a head file that is present but will not load. |
| 413 | More than 8 images, an image over 12 MB of base64, over 50 MP, or a decode refused by PIL's own decompression-bomb guard. |
| 415 | An image that is not JPEG/PNG/WebP. |
| 422 | Neither `image` nor `images`, or both; a malformed or unknown question type; more than 64 questions; an unknown head; a head trained on a different encoder; a single-caption `noul` on an uncalibrated encoder; a temperature that is not a finite number > 0; undecodable or truncated image bytes; a body that fails the schema. |

The 422s worth recognising:

```json
{"detail":"unknown head 'nope'; trained heads: colours"}
{"detail":"at most 64 questions per request"}
{"detail":"temperature must be greater than 0, got 0.0"}
{"detail":"head 'legacy' was trained on encoder 'toy' but this instance runs 'toy-calibrated'. Load 'toy', or retrain the head here."}
{"detail":"encoder 'toy' has no calibrated sigmoid, so a single-caption noul cannot produce a probability. Give a {\"true\": ..., \"false\": ...} criteria pair instead, which works on every encoder."}
```

`{"type":"noul","instructions":"..."}` with no `criteria` needs an encoder that publishes
both `logit_scale` and `logit_bias`. Against a server whose `/v1/health` says `siglip2-*`
it works; against CLIP or any ensemble it is a 422, and the fix is the
`{"true": ..., "false": ...}` pair, which works everywhere:

```bash
curl -s -X POST http://127.0.0.1:8127/v1/ask -H 'Authorization: Bearer s3cret' \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"questions\":{\"outside\":{\"type\":\"noul\",\"instructions\":\"a photo taken outdoors\"}}}"
```
```json
{"answers":{"outside":{"type":"noul","noul":1.0,"basis":"sigmoid readout"}},"timing":{"embed_ms":0.5,"total_ms":1.3}}
```

That last call ran against a third toy server whose encoder declares `logit_scale=100`
and `logit_bias=-10`, purely to show that the route exists and what `basis` says when it
is taken. `1.0` is an artefact of a fake calibration; read nothing into the number.

---

## POST /v1/embed

The embedding, so you can cache it, store it in a vector index, or ask questions locally
later without re-uploading the image.

**Request**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `image` | string | one of | One base64 image. |
| `images` | string[] | one of | Up to 8. |

Same rule as `/v1/ask`: one field or the other, and sending both is the same 422. Unlike
`/v1/ask`, the response shape does not change; `embeddings` is always a list of rows.

**Response**

| Field | Type | Meaning |
|---|---|---|
| `embeddings` | number[][] | One unit-norm row per image, in request order. |
| `dim` | int | Row width. Matches `/v1/health`'s `dim`. |

Rows are L2-normalised float32 rendered as JSON numbers. They are only comparable to
other embeddings from the same encoder, so check `/v1/health` before you mix them into an
existing index.

```bash
curl -s -X POST http://127.0.0.1:8123/v1/embed -H 'Content-Type: application/json' \
     -d "{\"image\":\"$IMG\"}" | jq -c '{dim, n: (.embeddings|length), head: .embeddings[0][:3]}'
```
```json
{"dim":16,"n":1,"head":[-0.09282517433166504,0.45749732851982117,-0.1432816982269287]}
```

(Piped through `jq` only to keep the page readable; the raw response carries all 16
numbers.)

**Status codes:** 200; 401; 413 (more than 8 images, over 12 MB, over 50 MP, or PIL's own
decompression-bomb guard); 415
(wrong format); 422 (neither field, both fields, or bytes that are not an image).

The decode errors are worth recognising, because they point at different mistakes:

```bash
curl -s -w ' [%{http_code}]' -X POST http://127.0.0.1:8123/v1/embed \
     -H 'Content-Type: application/json' -d '{"image":"aGVsbG8="}'
```
```json
{"detail":"images[0] is not a base64-encoded image; send the raw bytes base64'd, or a 'data:image/png;base64,...' URL"} [422]
```

```json
{"detail":"images[0] is a truncated or corrupt PNG"} [422]
{"detail":"images[0] is GIF; this server reads JPEG, PNG, WEBP"} [415]
{"detail":"images[0] is 9000x9000; the limit is 50MP per image"} [413]
{"detail":"images[0] is 13MB of base64; the limit is 12MB per image"} [413]
{"detail":"at most 8 images per request; got 9"} [413]
{"detail":"images[0] is too large to decode: Image size (196000000 pixels) exceeds limit of 178956970 pixels, could be decompression bomb DOS attack."} [413]
```

`images[0]` is the index in the list you sent; when you sent `image`, it is always
`images[0]`.

---

## GET /v1/heads

Every head on this server, sorted by name. No pagination, no filtering. A head is tens
of kilobytes of metadata and servers hold tens of them, not thousands.

**Response:** an array of head objects (see below). Empty array if none.

```bash
curl -s http://127.0.0.1:8123/v1/heads
```
```json
[{"name":"colours","encoder":"toy","classes":["blue","red"],"n":12,"counts":{"blue":6,"red":6},"cv_accuracy":0.5,"per_class":{"blue":0.5,"red":0.5},"folds":5,"created":1790180587.8140433,"dim":16,"bytes":136},{"name":"docs_colour","encoder":"toy","classes":["blue","red"],"n":12,"counts":{"blue":6,"red":6},"cv_accuracy":0.5,"per_class":{"blue":0.5,"red":0.5},"folds":5,"created":1790180588.652512,"dim":16,"bytes":136}]
```

Heads are global to the server, not scoped to a caller. This list re-reads the head
directory on every call, so it also picks up heads another process wrote.

**Status codes:** 200; 401.

---

## POST /v1/heads

Train a head: upload labelled images, get back the fitted head's metadata plus the
comparison against zero-shot prompts. The encoder stays frozen; all that is fitted is a
linear readout over its embeddings, which is why this is seconds and tens of kilobytes.
See [Training a head](training.md).

**Request**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`. Becomes a filename. Training a name that exists overwrites it. |
| `examples` | array | yes | `[{"image": "<base64>", "label": "<class>"}]`. Up to 4000 entries, 2–100 distinct labels, at least 2 examples per label. |
| `epochs` | int | no | 1–5000, default 500. |

The 8-image cap does not apply here; `examples` is bounded by 4000 instead. Every image
is embedded in this request, so a few thousand examples is a long synchronous call and
the connection stays open for the whole fit. Raise your client timeout rather than your hopes.

**Response:** the head object, plus `comparison`.

| Field | Type | Meaning |
|---|---|---|
| `name` | string | As sent. |
| `encoder` | string | The encoder that fitted it. A head refuses to run under any other. |
| `classes` | string[] | Sorted distinct labels. Column order of the head's weights. |
| `n` | int | Examples used. |
| `counts` | object | `{label: count}`. |
| `cv_accuracy` | number \| null | **Held-out** accuracy: stratified k-fold with `k = min(5, thinnest class)`. Never the accuracy on the images it was fitted to. Never `null` for a head trained through this route: a class with fewer than 2 examples is refused as a 422 before any fold is cut, so `k` is always at least 2. `null` only shows up for a head record that reached the directory some other way. |
| `per_class` | object | `{label: held-out accuracy}`, same basis. |
| `folds` | int \| null | The `k` actually used: 2 to 5, being `min(5, thinnest class)`. |
| `created` | number | Unix timestamp, float seconds. |
| `dim` | int | Embedding width the head expects. |
| `bytes` | int | Size of the weights in memory (fp32 `W` plus `b`). |
| `comparison` | object | Trained vs zero-shot, same shape as `GET /v1/heads/{name}/compare`. |

The weights never cross the wire, in either direction. There is no route that returns
`W`, and no route that uploads one.

```bash
curl -s -X POST http://127.0.0.1:8123/v1/heads -H 'Content-Type: application/json' -d @train.json
```
```json
{"name":"docs_colour","encoder":"toy","classes":["blue","red"],"n":12,"counts":{"blue":6,"red":6},"cv_accuracy":0.5,"per_class":{"blue":0.5,"red":0.5},"folds":5,"created":1790180588.652512,"dim":16,"bytes":136,"comparison":{"overall":{"zero_shot":0.75,"trained":0.5,"delta":-0.25},"per_class":[{"label":"blue","n":6,"zero_shot":0.5,"trained":0.5},{"label":"red","n":6,"zero_shot":1.0,"trained":0.5}],"basis":"trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example"}}
```

(`train.json` held 12 examples, six per class, built by base64-ing each PNG into
`{"name": "docs_colour", "examples": [{"image": ..., "label": ...}], "epochs": 300}`.)

**Status codes**

| Status | Cause |
|---|---|
| 200 | Trained and saved. |
| 401 | Keys configured, header missing or wrong. |
| 413 | More than 4000 examples, an image over 12 MB / 50 MP, or a decode refused by PIL's own decompression-bomb guard. |
| 415 | An example image that is not JPEG/PNG/WebP. |
| 422 | Schema failure (`name` or `examples` missing, `epochs` out of range); fewer than 2 distinct labels; a label with fewer than 2 examples; more than 100 classes; two labels differing only by surrounding whitespace; an invalid head name. |

```json
{"detail":"need at least 2 distinct labels"}
{"detail":"every class needs at least 2 examples; too few for: red"}
{"detail":"invalid head name '../evil': use 1-64 characters from [A-Za-z0-9._-], starting with a letter or digit"}
{"detail":[{"type":"less_than_equal","loc":["body","epochs"],"msg":"Input should be less than or equal to 5000","input":99999,"ctx":{"le":5000}}]}
```

`name` is validated first, before a single example is decoded, so a typo costs you the
upload and nothing else:

```bash
curl -s -w ' [%{http_code}]' -X POST http://127.0.0.1:8123/v1/heads \
     -H 'Content-Type: application/json' -d @bad-name-4001-examples.json
```
```json
{"detail":"invalid head name 'bad name!': use 1-64 characters from [A-Za-z0-9._-], starting with a letter or digit"} [422]
```

That body came back in 0.01s from a request carrying 4001 examples, over the example
cap, and it never got as far as counting them.

Per-example failures are labelled by their index in `examples`, not `images`:

```json
{"detail":"examples[2] is GIF; this server reads JPEG, PNG, WEBP"} [415]
{"detail":"at most 4000 examples per request; got 4001"} [413]
```

---

## GET /v1/heads/\{name\}

One head's metadata. Same object as an entry in `GET /v1/heads`, without `comparison`.

```bash
curl -s http://127.0.0.1:8123/v1/heads/docs_colour
```
```json
{"name":"docs_colour","encoder":"toy","classes":["blue","red"],"n":12,"counts":{"blue":6,"red":6},"cv_accuracy":0.5,"per_class":{"blue":0.5,"red":0.5},"folds":5,"created":1790180588.652512,"dim":16,"bytes":136}
```

**Status codes:** 200; 401; 404 (no such head).

```bash
curl -s -w ' [%{http_code}]' http://127.0.0.1:8123/v1/heads/nope
```
```json
{"detail":"no head named 'nope' (have: colours, docs_colour)"} [404]
```

It lists the heads that do exist, which is usually what you want. It does not name the
head directory: the store's own `KeyError` says "... in /var/lib/jev/heads (have: ...)",
which is right at a REPL and wrong over a socket, so the server strips the path before
sending.

A second 404 shape exists: a `.pt` file that is present but will not load says so rather
than claiming the head does not exist.

```bash
printf 'not a torch file' > $JEVIMAGE_HEADS_DIR/junkdoc.pt
curl -s -w ' [%{http_code}]' http://127.0.0.1:8123/v1/heads/junkdoc
```
```json
{"detail":"junkdoc.pt is present but unreadable: UnpicklingError: Weights only load failed. In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.. Delete it, or retrain the head."} [404]
```

The same body comes back from `/compare` and from a `head` question on `POST /v1/ask`,
where it is a 404, not the 422 an unknown head gets. `DELETE` still removes the file.

---

## GET /v1/heads/\{name\}/compare

"Did training actually help?", answered per class on held-out data. Training a head does
not always beat well-written prompts; this route is why you do not have to guess.

**Response**

| Field | Type | Meaning |
|---|---|---|
| `overall.zero_shot` | number | Accuracy of `"a photo of {class}"` prompts, scored on every example. |
| `overall.trained` | number \| null | The head's k-fold held-out accuracy. `null` if no cross-validation ran. |
| `overall.delta` | number \| null | `trained - zero_shot`. `null` when `trained` is. |
| `per_class` | array | One row per class: `label`, `n`, `zero_shot`, `trained`. |
| `basis` | string | A sentence stating the asymmetry, so a screenshot of this cannot be misread. |

The two sides are scored differently on purpose: the trained side is cross-validated,
because a head scored on the examples it was fitted to wins by construction and tells you
nothing, while the zero-shot side has no training to hold out and so is scored on
everything.

```bash
curl -s http://127.0.0.1:8123/v1/heads/docs_colour/compare
```
```json
{"overall":{"zero_shot":0.75,"trained":0.5,"delta":-0.25},"per_class":[{"label":"blue","n":6,"zero_shot":0.5,"trained":0.5},{"label":"red","n":6,"zero_shot":1.0,"trained":0.5}],"basis":"trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example"}
```

Read the rows, not the total. A head can gain overall and still lose a class you care
about.

**Status codes**

| Status | Cause |
|---|---|
| 200 | Compared. |
| 401 | Keys configured, header missing or wrong. |
| 404 | No such head. |
| 409 | The head exists but was saved without the embeddings it was fitted on, so there is nothing to re-score. |
| 422 | The head was fitted by a different encoder, or expects a different embedding width, than this server runs. `compare()` runs the same check `ask()` does before it looks for the stored embeddings, so `GET /v1/heads/{name}` can be a 200 on a head whose `/compare` is a 422. |

```bash
curl -s -w ' [%{http_code}]' -H 'Authorization: Bearer s3cret' \
     http://127.0.0.1:8126/v1/heads/legacy/compare
```
```json
{"detail":"head 'legacy' was saved without its embeddings, so it cannot be compared; retrain it to restore this"} [409]
```

(That head was produced by saving a trained head with its stored embeddings stripped,
the shape of an older or hand-edited head file.) `jevimage.connect(...).compare()` turns
this 409 into `None`, matching the local API.

---

## DELETE /v1/heads/\{name\}

Remove a head, and its file, permanently. There is no undo and no versioning.

```bash
curl -s -X DELETE http://127.0.0.1:8123/v1/heads/docs_colour
```
```json
{"deleted":"docs_colour"}
```

**Status codes**

| Status | Cause |
|---|---|
| 200 | Deleted. Body is `{"deleted": "<name>"}`. |
| 401 | Keys configured, header missing or wrong. |
| 404 | `{"detail":"no head named 'nope'"}` - nothing was deleted. Not idempotent-friendly: a second delete of the same head is a 404. |
| 422 | The name is not a legal head name. |

```bash
curl -s -w ' [%{http_code}]' -X DELETE 'http://127.0.0.1:8123/v1/heads/bad%20name!'
```
```json
{"detail":"invalid head name 'bad name!': use 1-64 characters from [A-Za-z0-9._-], starting with a letter or digit"} [422]
```

As on `POST /v1/heads`, the name is checked before anything else happens. `GET` on a bad
name is a 404 rather than a 422, because the lookup fails before any filename is built
from it.

---

## OpenAPI

FastAPI serves the generated schema at `/openapi.json` and interactive docs at `/docs`
(and `/redoc`). Both are unauthenticated even when API keys are configured, because they
sit outside the `/v1` router.

```bash
curl -s http://127.0.0.1:8126/openapi.json | jq -c '{info, paths: (.paths|keys)}'
```
```json
{"info":{"title":"jevimage","version":"0.1.19"},"paths":["/v1/ask","/v1/embed","/v1/heads","/v1/heads/{name}","/v1/heads/{name}/compare","/v1/health"]}
```

The schema is generated from the request models, so it is exact about request bodies and
vague about responses; the handlers return plain dicts with no declared response model.
For response shapes, this page and `jevimage/server.py` are the reference. The key check
is a router dependency rather than a declared security scheme, so the schema carries
`authorization` as an ordinary optional header parameter on every `/v1` operation and
defines no `securitySchemes`. `/docs` gives you a text box rather than an Authorize
button, and will happily send the request without it and show you the 401.

---

## Failure modes worth knowing about

- **Heads are shared, mutable, global state.** Any caller with the key can overwrite or
  delete any head. Training a name that already exists replaces it with no warning.
- **Training blocks.** `POST /v1/heads` embeds every example inline. There is no job id,
  no progress and no way to cancel other than dropping the connection.
- **A head is tied to its encoder by name.** Point a server at the same head directory
  with a different `JEVIMAGE_ENCODER` and every `head` question becomes a 422, and so does
  `GET /v1/heads/{name}/compare` - though `GET /v1/heads` still lists the head and
  `GET /v1/heads/{name}` still returns it. The same embedding width does not mean the
  same embedding space.
- **A held-out number off two examples is a coin toss.** `k` is `min(5, thinnest class)`,
  so a class with two examples is scored on two folds and `cv_accuracy` means very little.
  Read it next to `counts`. (One example is not a thin number, it is a 422.)
- **No rate limiting, no request-size cap, no per-key identity.** The single shared key is
  the whole security model.

## The client that already speaks this

Every route has a method on `jevimage.connect(url)`, which is stdlib-only and needs
neither torch nor transformers:

| Route | `Remote` method |
|---|---|
| `GET /v1/health` | `health()` |
| `POST /v1/ask` | `ask(image, questions, temperature=None)` |
| `POST /v1/embed` | `embed(images)` - chunks past the 8-image cap for you |
| `POST /v1/heads` | `train(name, data, epochs=500, max_per_class=None)` |
| `GET /v1/heads` | `list_heads()` |
| `GET /v1/heads/{name}` | `head(name)` |
| `GET /v1/heads/{name}/compare` | `compare(name)` - `None` on 409 |
| `DELETE /v1/heads/{name}` | `delete_head(name)` - `False` on 404 |

Non-2xx responses become `jevimage.JevError`, which carries `.status` and `.detail`
verbatim. See [Python API reference](reference-python.md); for running the server,
[Serving](serving.md).

---

[← Python reference](reference-python.md) · [Docs index](index.md) · [Cookbook →](cookbook.md)
