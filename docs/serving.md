---
title: "Serving"
description: "jevimage serve, API keys, limits, deployment and what to monitor."
---

Running jevimage for other people. One process holds the encoder; everyone else installs
`pip install jevimage` (a few hundred kilobytes, no torch, no weights) and calls
`jevimage.connect(url)`.

Every route under `/v1` is one method on `jevimage.Jev`, so there is no second API to
learn: `/v1/ask` is `ask()`, `/v1/embed` is `embed()`, `/v1/heads` is
`train`/`list_heads`/`delete_head`. The wire format itself is in
[reference-http.md](reference-http.md); this page is about operating the thing.

> Every transcript below was run against a **deterministic stand-in encoder** (16-dim,
> no downloads), so the transcripts reproduce in a second. Statuses, error text,
> JSON shapes and commands are real. Probabilities and latencies are the toy's, not
> SigLIP's. A real encoder is slower and its probabilities are spread out rather than
> saturated.

## Install

```console
$ pip install 'jevimage[serve]'
```

`[serve]` implies `[local]`: a server runs the encoder, so it pulls torch, transformers,
fastapi and uvicorn. See [install.md](install.md) for what each extra costs.

## Starting it

### `jevimage serve`

```console
$ jevimage serve --help
usage: jevimage serve [-h] [--host HOST] [--port PORT] [--encoder ENCODER]
                      [--heads-dir HEADS_DIR] [--api-key KEY [KEY ...]]

options:
  -h, --help            show this help message and exit
  --host HOST
  --port PORT           default 8731; if it is taken, the next free port up to 8740 is
                        used
  --encoder ENCODER
  --heads-dir HEADS_DIR
  --api-key KEY [KEY ...]
                        require one of these keys; without it the API is open
```

Defaults: `--host 127.0.0.1`, `--port 8731`, the registry default encoder
(`siglip2-base-224`), heads in `$JEVIMAGE_HEADS_DIR`, else `$JEVIMAGE_HOME/heads`, else
`~/.jev/heads`, and **no key**.

8731 rather than 8000 because 8000 is the default of every other Python web server on
the machine, and losing that race after twelve seconds of loading the model is a bad
way to find out. If 8731 is taken too, `serve` walks up to 8740, prints which port it
took, and errors only when all ten are busy.

**The walk is a convenience for a laptop, not a deployment feature.** `--port` moves
where it starts but does not switch it off, so a pinned port that is busy still moves -
and an `upstream 127.0.0.1:8731` that quietly became 8732 on restart is a 502 your proxy
config cannot see. Behind a proxy, run uvicorn directly, as the systemd and Docker
examples below do: `uvicorn jevimage.server:app --port 8731` binds that port or fails,
which is what you want from a unit file.

The flags are not passed to the app object. They are written into the environment
(`JEVIMAGE_ENCODER`, `JEVIMAGE_HEADS_DIR`, `JEVIMAGE_API_KEYS`) and the app reads them itself, so the
same knobs work when a container or a process manager starts uvicorn directly:

```console
$ JEVIMAGE_ENCODER=siglip2-so400m-384 JEVIMAGE_HEADS_DIR=/var/lib/jev/heads \
  uvicorn jevimage.server:app --host 0.0.0.0 --port 8731
```

`jevimage serve` is a single uvicorn process with no `--workers`, no TLS flags and no
`--reload`. When you want any of those, run uvicorn yourself; the two paths build the
identical app.

**`jevimage serve` only sees the built-in encoder registry.** The CLI never imports your code,
so a registered custom encoder is invisible to it:

```console
$ jevimage serve --encoder nope
jevimage: unknown encoder 'nope'. Available: clip-vit-l-14, dfn5b-h-14-384, metaclip2-huge,
siglip2-base-224, siglip2-base-384, siglip2-giant-384, siglip2-so400m-384. Register your
own with jevimage.register(name, factory) - from Python, in the process that will serve
it; the `jevimage` command only sees these built-in names, because it never imports your code.
$ echo $?
1
```

### `build_app()`: your own encoder

```python
# serve_toy.py
"""Serve a custom encoder: build the app yourself and hand it a Jev."""
import uvicorn

import jevimage
from jevimage.server import build_app
from toy import Toy                      # your encoder class

jev = jevimage.load(Toy(), heads_dir="/tmp/jev-heads", batch=True)
app = build_app(jev, api_keys=["s3cret"])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8124)
```

```console
$ python serve_toy.py &
$ curl -s http://127.0.0.1:8124/v1/health
{"encoder":"toy","dim":16,"device":"cpu","primitives":["choice", "score", "noul", "head"],"heads":0,"auth":"bearer","version":"0.1.19"}
```

`build_app(jev=None, *, api_keys=None)`:

- `jev=None` calls `jevimage.load(batch=True)`, which reads `$JEVIMAGE_ENCODER` and
  `$JEVIMAGE_HEADS_DIR`. Pass a `Jev` to control the encoder, the heads directory, the
  temperature, or batching.
- `api_keys=None` reads `$JEVIMAGE_API_KEYS`. Pass a list to set them in code. Note that
  `api_keys=[]` means *open*, overriding the environment.

Keeping `app` at module level (rather than only inside `__main__`) is what lets you run
it under `uvicorn serve_toy:app --workers 2`.

`jevimage.server.app` is built lazily through a module `__getattr__`, so
`from jevimage.server import build_app` does **not** load a model; touching
`jevimage.server:app` does.

### Mounting it in your own FastAPI app

`build_app()` returns an ordinary `FastAPI` instance. Add routes to it, or mount it under
a prefix:

```python
# mount_toy.py
from fastapi import FastAPI
import jevimage
from jevimage.server import build_app
from toy import Toy

api = FastAPI()

@api.get("/")
def root():
    return {"service": "my app"}

api.mount("/jev", build_app(jevimage.load(Toy(), heads_dir="/tmp/jev-heads")))
```

```console
$ uvicorn mount_toy:api --host 127.0.0.1 --port 8735 &
$ curl -s http://127.0.0.1:8735/ ; echo
{"service":"my app"}
$ curl -s http://127.0.0.1:8735/jev/v1/health ; echo
{"encoder":"toy","dim":16,"device":"cpu","primitives":["choice", "score", "noul", "head"],"heads":0,"auth":"open","version":"0.1.19"}
```

(`mount_toy.py` has no `uvicorn.run` of its own - it is a module for a server to import,
which is why the ASGI path `mount_toy:api` is on the command line instead.)

A mount has its own auth: the sub-app checks the keys `build_app` was given, and the
outer app's route dependencies do not reach it. `FastAPI(dependencies=[...])` on the
outer app leaves `/jev/v1/health` open. Outer *middleware* does wrap the mount, so if you
mount an open app inside an authenticated one, put the check there, or give `build_app`
keys anyway.

## API keys

One header, one form:

```
Authorization: Bearer <key>
```

The scheme token is case-insensitive (RFC 7235 §2.1) and any run of whitespace separates
it from the key, so `bearer <key>` is accepted; the key itself is compared exactly.

Keys come from `$JEVIMAGE_API_KEYS`, split on commas **and** whitespace, so both spellings
work and a unit file that copies the CLI's comma form cannot produce a key with a space
in it:

```console
$ JEVIMAGE_API_KEYS="k1, k2 k3" python -c "import os; print(os.environ['JEVIMAGE_API_KEYS'].replace(',',' ').split())"
['k1', 'k2', 'k3']
```

Any one of them is accepted; comparison is `hmac.compare_digest` against each. A junk or
non-ASCII header is a 401, not a 500. Empty strings are dropped before the guard is built,
so `api_keys=[os.environ.get("JEVIMAGE_KEY", "")]` on a host where that variable is unset is
an *open* server that says `"auth": "open"`, not one that claims to be protected and is
not.

On the client side, the key goes to whatever the URL resolves to. `connect()` refuses
redirects rather than following them, because urllib copies the `Authorization` header
onto the redirect target (a different host, or a downgraded scheme, included), so one
302 from a hostile or compromised server would hand over the key. A 3xx surfaces as a
`JevError` naming the `Location`. That still leaves the transport: over plain `http://`,
as every example on this page does, the key is on the wire in clear. Terminate TLS in
front of `jevimage serve` for anything that leaves a trusted network.

`$IMG` in the transcripts below is one base64 image, made once:

```console
$ IMG=$(base64 -w0 ds/red/0.png)
```

```console
$ curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8124/v1/embed \
    -H 'Content-Type: application/json' -d "{\"image\":\"$IMG\"}"
401
$ curl -s -X POST http://127.0.0.1:8124/v1/embed -H 'Authorization: Bearer wrong' \
    -H 'Content-Type: application/json' -d "{\"image\":\"$IMG\"}"
{"detail":"send 'Authorization: Bearer <key>' with a key this server was started with (JEVIMAGE_API_KEYS)"}
$ curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8124/v1/embed \
    -H 'Authorization: s3cret' -H 'Content-Type: application/json' -d "{\"image\":\"$IMG\"}"
401
```

The key alone, without the `Bearer ` prefix, is rejected.

### What the key does not cover

| Path | Keyed? | Why |
|---|---|---|
| `/v1/health` | no | A load balancer that needs a secret to ask whether the process is alive will report that it is not. |
| `/docs`, `/redoc`, `/openapi.json` | no | FastAPI mounts them outside the `/v1` router, and jevimage does not disable them. |
| everything under `/v1` | yes | One dependency on the router. |

```console
$ for p in /docs /openapi.json /redoc; do printf "%-14s " $p; \
    curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8124$p; done
/docs          200
/openapi.json  200
/redoc         200
```

If a keyless schema page is not acceptable, build the app with `FastAPI(docs_url=None,
...)` (which means constructing it yourself rather than calling `build_app`), or block
those three paths at the proxy.

Auth is a route dependency, so the body is read and JSON-parsed **before** the key is
checked. A malformed body from an anonymous caller gets a 422, not a 401:

```console
$ curl -s -w " [%{http_code}]\n" -X POST http://127.0.0.1:8124/v1/embed \
    -H 'Content-Type: application/json' -d '{"image": '
{"detail":[{"type":"json_invalid","loc":["body",10],...}]} [422]
```

The practical consequence: a key does not protect you from a flood of large uploads. A
body-size cap at the proxy does.

### No key means open

Not "read-only open". Open:

```console
$ curl -s http://127.0.0.1:8731/v1/health ; echo
{"encoder":"toy",...,"auth":"open","version":"0.1.19"}
$ curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8731/v1/heads \
    -H 'Content-Type: application/json' --data-binary @train.json
200
$ curl -s -X DELETE http://127.0.0.1:8731/v1/heads/colour ; echo
{"deleted":"colour"}
```

Anyone who can reach the port can spend your GPU on `/v1/ask`, upload up to 4000 images
to `/v1/heads`, overwrite a head that shares a name with theirs, and delete any head you
have. **`--host 0.0.0.0` with no key is a public GPU with a public disk.** `127.0.0.1`
with no key, behind a proxy that does the authentication, is a reasonable setup; `0.0.0.0`
with no key is not.

Put the keys in a file that only the service user can read, not on the command line.
`jevimage serve --api-key hunter2` shows up in `ps` for every user on the box. The systemd
unit below uses `EnvironmentFile` for exactly this.

## The limits the server enforces

Read straight from `jevimage/server.py`:

| Constant | Value | Applies to |
|---|---|---|
| `MAX_IMAGES` | 8 | images per `/v1/ask` and `/v1/embed` |
| `MAX_B64` | 12 MB | base64 length of one image |
| `MAX_PIXELS` | 50 MP | width × height of one image |
| `FORMATS` | JPEG, PNG, WEBP | what it will decode |
| `training.MAX_EXAMPLES` | 4000 | examples per `/v1/heads` (POST) |
| `training.MAX_CLASSES` | 100 | distinct labels in one head |
| `core.MAX_QUESTIONS` | 64 | questions in one `ask` |
| `epochs` | 1..5000 | validated by pydantic |

Everything each refusal returns, verified against a running server:

| Status | Cause | Body |
|---|---|---|
| 401 | missing, malformed or wrong key | `send 'Authorization: Bearer <key>' with a key this server was started with (JEVIMAGE_API_KEYS)` |
| 404 | `GET` an absent head | `no head named 'nope' (have: none)`. The store names the directory it searched, and the server strips it back out on the way to the client |
| 404 | `DELETE` an absent head | `no head named 'nope'`. The route's own message, without the inventory |
| 404 | a head file that is present but will not load | `junkdoc.pt is present but unreadable: <the loader's first line>. Delete it, or retrain the head.` From `GET`, from `/compare`, and from a `head` question on `/v1/ask`; the full body is pasted in [reference-http.md](reference-http.md#get-v1headsname) |
| 409 | `compare` on a head stored without its embeddings | `head 'noemb' was saved without its embeddings, so it cannot be compared; retrain it to restore this` |
| 413 | too many images | `at most 8 images per request; got 9` |
| 413 | base64 too long | `images[0] is 13MB of base64; the limit is 12MB per image` |
| 413 | too many pixels | `images[0] is 8000x8000; the limit is 50MP per image` |
| 413 | PIL's own decompression-bomb guard | `images[0] is too large to decode: Image size (196000000 pixels) exceeds limit of 178956970 pixels, could be decompression bomb DOS attack.` Whichever guard fired names its own number, ours or PIL's larger one |
| 415 | format it does not read | `images[0] is GIF; this server reads JPEG, PNG, WEBP` |
| 422 | no `image` and no `images` | `send 'image' (one base64 string) or 'images' (a list)` |
| 422 | both `image` and `images` | `send 'image' or 'images', not both: 'image' returns one answer object, 'images' returns a list` |
| 422 | not base64, or not an image | `images[0] is not a base64-encoded image; send the raw bytes base64'd, or a 'data:image/png;base64,...' URL` |
| 422 | truncated upload | `images[0] is a truncated or corrupt PNG` |
| 422 | any `ValueError` from the core | `unknown head 'nope'; trained heads: none trained yet` |
| 422 | encoder/head mismatch | `head 'colour' was trained on encoder 'toy' but this instance runs 'toy-calibrated'. Load 'toy', or retrain the head here.` |
| 422 | pydantic field validation | `[{"type":"greater_than_equal","loc":["body","epochs"],...}]` |

The transcript behind that table:

```console
$ for f in gif big huge nine trunc; do printf "%-6s " $f; \
    curl -s -w " [%{http_code}]\n" -X POST http://127.0.0.1:8124/v1/embed \
      -H 'Authorization: Bearer s3cret' -H 'Content-Type: application/json' \
      --data-binary @$f.json; done
gif    {"detail":"images[0] is GIF; this server reads JPEG, PNG, WEBP"} [415]
big    {"detail":"images[0] is 8000x8000; the limit is 50MP per image"} [413]
huge   {"detail":"images[0] is 13MB of base64; the limit is 12MB per image"} [413]
nine   {"detail":"at most 8 images per request; got 9"} [413]
trunc  {"detail":"images[0] is a truncated or corrupt PNG"} [422]
```

Those bodies are `/v1/ask` and `/v1/embed`. `POST /v1/heads` counts and labels **examples**
instead, with its own larger cap:

```console
{"detail":"at most 4000 examples per request; got 4001"} [413]
{"detail":"examples[2] is GIF; this server reads JPEG, PNG, WEBP"} [415]
{"detail":"examples[1] is not a base64-encoded image; send the raw bytes base64'd, or a 'data:image/png;base64,...' URL"} [422]
```

- **The pixel cap is checked from the header, before the pixels are allocated.** A
  solid-colour 8000×8000 PNG is 263 KB of base64 and 190 MB of RAM once decoded; the
  413 above cost neither.
- **`422` is the core's own sentence, not a wrapper.** `ValueError` and `KeyError` from
  `jevimage` reach the caller as 422 and 404, because "head *x* was trained on
  encoder *y*" is the answer the caller needs, not a symptom to hide. The one thing the
  server edits out is a filesystem path: a 404 body does not disclose the heads
  directory, however the message read at the REPL.
- **There is no cap on the total body.** Only per-image size and per-request count. A
  training POST is 4000 images × up to 12 MB each by the server's own arithmetic. The
  ceiling has to come from the proxy.

## `batch=True` and the shared encode

`build_app()` with no `jev` uses `jevimage.load(batch=True)`, because a server is what the
batcher is for. Requests arriving within a few milliseconds of each other are collected
into one forward pass.

The mechanism, measured with a stand-in encoder that sleeps 10 ms per call to stand in
for a real forward pass (the toy's own encode is microseconds, so nothing ever queues):

```python
class Slow(Toy):
    passes = []
    def img(self, ims):
        Slow.passes.append(len(ims))
        time.sleep(0.010)
        return super().img(ims)

for batch in (False, True):
    jev = jevimage.load(Slow(), heads_dir=..., batch=batch)
    jev.embed("ds/red/0.png")                     # prime the worker thread
    Slow.passes.clear()
    ts = [threading.Thread(target=lambda: jev.embed("ds/red/0.png")) for _ in range(32)]
    for t in ts: t.start()
    for t in ts: t.join()
```

```text
batch=False 32 concurrent 1-image asks: 32 forward passes [1, 1, 1, ... 1], wall 38ms
batch=True  32 concurrent 1-image asks:  3 forward passes [3, 16, 13], wall 56ms
```

32 requests became 3 passes. **Only the collapse reproduces.** How many passes, and how
they are filled, moves from run to run and from machine to machine, and the wall times
say nothing at all: on the machine above the batched side is the *slower* of the two.
That is the honest limitation of this measurement: a `sleep` costs the same for 8 images
as for 1, so it models the coalescing but not the payoff. The payoff is real on a GPU,
where the marginal image in a batch is nearly free. `jevimage/batcher.py` records the
project's own measurements on an A10G. I did not reproduce them, and this machine has no
encoder to reproduce them with:

```text
batch  1   9.68 ms/img   103 img/s
batch  8   2.57 ms/img   388 img/s   3.8x
batch 16   2.05 ms/img   487 img/s   4.7x
```

Three properties worth knowing before you rely on it:

- **The wait is adaptive, and it is off when traffic is thin.** `max_wait` is 6 ms, but
  the batcher only waits when its recent batch size (an EMA) is at least 1.5. On an idle
  service a lone request goes straight through and pays nothing. Under load the batch
  fills before the timer. `max_batch` is 16, so one forward pass never carries more
  than 16 images, whatever `stats["max_seen"]` says you were offered.
- **Only single-image requests are batched.** `Jev.embed` routes through the batcher only
  when it is handed exactly one image. A `/v1/ask` with `"image"` is one image and gets
  batched; `/v1/embed` with `"images": [a, b]` goes straight to the encoder as its own
  pass:

  ```text
  16 concurrent 1-image embeds -> 2 forward passes    (they queue together and coalesce)
  one 2-image embed            -> [2]                 (one pass, batcher bypassed)
  ```

  That is the right default (a caller who sent a list already batched), but it means a
  client that loops `embed(one)` in 8 threads behaves differently from one that sends
  `embed([eight])`.
- **One bad batch does not kill the worker.** The exception is delivered to every caller
  in that batch and the thread carries on.

The other half of the economics is per-request: `/v1/ask` encodes the image once and reads
every question against that one vector. The response says so:

```console
$ curl -s -X POST http://127.0.0.1:8124/v1/ask -H 'Authorization: Bearer s3cret' \
    -H 'Content-Type: application/json' \
    -d "{\"image\":\"$IMG\",\"questions\":{\"colour\":{\"type\":\"choice\",\"criteria\":{\"red\":\"a red square\",\"blue\":\"a blue square\"}}}}"
{"answers":{"colour":{"type":"choice","choice":"red","probabilities":{"red":1.0,"blue":0.0},"confidence":0.999999}},"timing":{"embed_ms":11.1,"total_ms":12.0}}
```

`embed_ms` is the encode; `total_ms - embed_ms` is every question. Ten questions in one
request cost one encode; ten requests cost ten. Caption embeddings are cached per string
across requests (50 000 strings, then oldest-first eviction), so a stable question set
warms up once.

## Health checks

```console
$ curl -s http://127.0.0.1:8124/v1/health ; echo
{"encoder":"toy","dim":16,"device":"cpu","primitives":["choice", "score", "noul", "head"],"heads":0,"auth":"bearer","version":"0.1.19"}
```

- Never authenticated, on purpose.
- `encoder` and `dim` are what your heads will be trained under. `encoder` changing is
  the single most disruptive thing that can happen to a deployment (see below).
- `auth` is `"bearer"` or `"open"`, a one-request check that you did not ship an open
  server by accident.
- `device` is `getattr(encoder, "device", "cpu")`. `HFCLIP` and `OpenCLIP` set it; a
  custom encoder that does not will report `cpu` however it actually runs.
- `heads` re-reads the head directory on every call (`jev.heads.sync()`), which is a
  `glob` plus a `torch.load` of anything that changed. Once a second is nothing; once a
  millisecond is a filesystem benchmark.

**The process does not answer until the weights are loaded.** The app is built (and the
encoder loaded) while the app object is being imported, before uvicorn serves anything.
A single process refuses the connection until then; a `--workers N` run accepts it (the
supervisor owns the socket) and leaves it waiting for a worker. First boot also downloads
weights. Give the check a startup grace period generous enough for a cold model pull, or
you will restart-loop a box that is doing exactly what you asked.

## What to monitor

There is no `/metrics` route. What exists:

- `timing.embed_ms` and `timing.total_ms` on every `/v1/ask` response. Watch the p95 of
  `embed_ms`; that is the encoder and the queue. If `total_ms - embed_ms` grows instead,
  you are asking a lot of questions with a lot of new captions.
- Status-code rates, split the way the table above splits them. A rising 413/415 rate is
  a client sending the wrong thing; a rising 422 rate is usually one caller with a
  malformed question or a head that no longer matches the encoder.
- `uvicorn` access logs for latency and non-2xx.
- Process RSS and (on GPU) `nvidia-smi` memory. One model per worker process.

If you want counters, the app is yours to extend:

```python
# metrics_toy.py
jev = jevimage.load(Toy(), heads_dir="/tmp/jev-heads", batch=True)
app = build_app(jev)

@app.get("/metrics")
def metrics():
    return {"text_cache": jev.engine.stats,
            "batches": jev._batcher.stats if jev._batcher else None,
            "heads": len(jev.heads)}
```

```console
$ curl -s http://127.0.0.1:8733/metrics ; echo
{"text_cache":{"text_hits":2,"text_misses":2},"batches":{"calls":1,"batches":1,"items":1,"max_seen":1},"heads":1}
```

`text_hits`/`text_misses` is the caption cache; `items / batches` is your mean batch size,
which tells you whether batching is doing anything. `calls` counts what was submitted to
the batcher and `items` what a forward pass has come back for, so the two agree except
for whatever is in flight. Both of these are per process, so with
`--workers N` you are looking at one worker's view. `jev._batcher` is private; it is
stable enough to read for metrics, and nothing else here is.

## Reverse proxy and TLS

jevimage has no TLS and no CORS headers. A browser page on another origin cannot call it,
and `jevimage serve` cannot terminate HTTPS. Both belong to a proxy in front.

The proxy caps the whole body; the server does not.

```caddyfile
jev.example.com {
	# TLS is Caddy's job, not the app's: jevimage has no certificate handling.
	reverse_proxy 127.0.0.1:8731

	# The server caps images individually (12MB of base64, 8 per request) but puts no
	# ceiling on a whole body, and /v1/heads takes up to 4000 of them. This is that
	# ceiling.
	request_body {
		max_size 100MB
	}
}
```

```console
$ caddy validate --config Caddyfile --adapter caddyfile
...
Valid configuration
```

Verified end to end against a local copy of that config with `max_size 1MB`:

```console
$ curl -s -X POST http://127.0.0.1:8734/v1/ask -H 'Authorization: Bearer s3cret' \
    -H 'Content-Type: application/json' \
    -d "{\"image\":\"$IMG\",\"questions\":{\"c\":{\"type\":\"choice\",\"criteria\":{\"red\":\"a red square\",\"blue\":\"a blue square\"}}}}"
{"answers":{"c":{"type":"choice","choice":"red","probabilities":{"red":1.0,"blue":0.0},"confidence":0.999999}},"timing":{"embed_ms":0.8,"total_ms":1.1}}

$ curl -s -w " [%{http_code}]\n" -X POST http://127.0.0.1:8734/v1/embed \
    -H 'Authorization: Bearer s3cret' -H 'Content-Type: application/json' \
    --data-binary @huge.json        # 13 MB
 [502]
```

Note the status. Caddy 2.6.2 logs `HTTP 413: http: request body too large` and returns
**502** to the client, because it had already begun proxying when the limit tripped. If
your clients need a clean 413, check what your proxy version actually emits rather than
assuming. The nginx equivalent is `client_max_body_size 100m`, which does return 413.

Other proxy notes:

- If you need a browser to call this directly, add CORS at the proxy. Adding
  `CORSMiddleware` to the app is also possible, but then the policy lives in your code
  and the TLS lives in the proxy, which is one config too many.
- Uvicorn trusts `X-Forwarded-*` from `127.0.0.1` by default; if the proxy is on another
  host, pass `--forwarded-allow-ips` or every access log line will say the proxy's IP.
- `/v1/heads` (POST) is the slow route; it uploads a dataset and fits a head. Give the
  proxy a read timeout well above its default for that path, or the client will see a
  504 for a request the server completes anyway.
- Client-side, `jevimage.connect(url, timeout=60.0)` is the default. A 4000-image
  training upload will blow through 60 s: `connect(url, timeout=900)`.

## A worked deployment: systemd

Not containerised, one box, one GPU.

```ini
# /etc/systemd/system/jevimage.service
[Unit]
Description=jevimage HTTP API
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=jev
Group=jev
WorkingDirectory=/srv/jev

# The encoder and where heads are written. build_app() reads these itself, so there is
# no second place to keep them in sync.
Environment=JEVIMAGE_ENCODER=siglip2-base-224
Environment=JEVIMAGE_HEADS_DIR=/var/lib/jev/heads
Environment=HF_HOME=/var/lib/jev/hf
# JEVIMAGE_API_KEYS=key1,key2 - a separate file so the keys are not in the unit, and so
# rotating them is `systemctl restart`, not an edit of a world-readable unit.
EnvironmentFile=/etc/jev/keys.env

ExecStart=/srv/jev/venv/bin/uvicorn jevimage.server:app --host 127.0.0.1 --port 8731

# First boot downloads the weights; the socket does not open until that finishes.
TimeoutStartSec=900
Restart=on-failure
RestartSec=5

StateDirectory=jev
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/jev

[Install]
WantedBy=multi-user.target
```

```console
$ systemd-analyze verify ./jevimage.service
jevimage.service: Command /srv/jev/venv/bin/uvicorn is not executable: No such file or directory
$ # ... and with ExecStart pointed at a uvicorn that exists on this machine:
$ systemd-analyze verify ./check.service ; echo "exit=$?"
exit=0
```

Why these bits:

- `--host 127.0.0.1`. The proxy is the only thing that talks to it.
- `HF_HOME` inside `StateDirectory`; otherwise the cache lands in the service user's
  home, which `ProtectHome=yes` has just made inaccessible, and every restart
  re-downloads a few gigabytes.
- `ProtectSystem=strict` plus one `ReadWritePaths`: heads are the only thing this process
  writes.
- `Restart=on-failure`, not `always`: a config error that makes the encoder unloadable
  should stay down and visible, not loop while pulling weights.
- `/etc/jev/keys.env` should be `chmod 600`, owned by root, containing one line:
  `JEVIMAGE_API_KEYS=key1,key2`.

## A worked deployment: Docker

```dockerfile
FROM python:3.12-slim

# torch's default wheels carry the whole CUDA stack. For a CPU deployment this is the
# difference between a 400MB image and a 4GB one; drop the index line and use a
# nvidia/cuda base image when you have a GPU.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /src
COPY . /src
RUN pip install '.[serve]'

# Bake the weights into the image. Without this the first request after every deploy
# waits on a multi-gigabyte download, and a node with no egress never becomes ready.
ENV HF_HOME=/models JEVIMAGE_ENCODER=siglip2-base-224
RUN python -c "import jevimage; jevimage.load()" && chmod -R a+rX /models

# Heads are state. Mount a volume here or they die with the container.
ENV JEVIMAGE_HEADS_DIR=/var/lib/jev/heads
RUN mkdir -p /var/lib/jev/heads && chown 65534:65534 /var/lib/jev/heads
USER 65534

EXPOSE 8731
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8731/v1/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "jevimage.server:app", "--host", "0.0.0.0", "--port", "8731"]
```

```console
$ docker build -t jevimage .
$ docker run -d --name jevimage -p 127.0.0.1:8731:8731 \
    -e JEVIMAGE_API_KEYS="$(cat /etc/jev/key)" \
    -v jev-heads:/var/lib/jev/heads \
    jevimage
```

> Unlike everything else on this page, this Dockerfile was **not built here**. The
> machine it was written on has 2.4 GB of free disk and torch does not fit. It is
> assembled from behaviour that was verified (the env-var interface, `uvicorn
> jevimage.server:app`, the health route, where heads are written), not from a build log.
> Treat it as a starting point and read the build output.

`--host 0.0.0.0` inside the container is right; publishing it as `-p 127.0.0.1:8731:8731`
is what keeps it off the network. `-p 8731:8731` publishes it to the world, and the
container has no key unless you gave it one.

For a GPU: start from `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, drop the CPU index, and
run with `--gpus all`. `resolve_device()` picks CUDA automatically when torch sees it, and
`$JEVIMAGE_DEVICE` overrides.

## Scaling

**What is GPU-bound:** the encoder forward pass, and nothing else. Every question is a
matmul against a 768-float vector; a trained head is one more. That is why `total_ms` in
the transcripts above sits so close to `embed_ms`.

**What is CPU-bound:** base64 decoding, PIL decoding, EXIF transpose, `convert("RGB")`,
and the processor's resize/normalise. For a large JPEG this is not a rounding error; it
can rival a small model's forward pass. It is also where the request threads spend their
time.

**How the concurrency works.** The route handlers are ordinary `def`, so uvicorn runs them
in its threadpool. Decoding therefore happens in parallel across requests. With
`batch=True`, everything after decoding (the exif transpose, the preprocess and the
forward pass) runs in the single batcher thread, which is the serialisation point.

Given that, the knobs in order:

1. **Ask more questions per request.** Free for a question set the process has already
   seen: ten questions in one `/v1/ask` is one encode, and the second question costs a
   matmul. A caption this process has *not* encoded before costs a text-tower pass
   first, so a request full of fresh captions is not free; see
   [encoders.md](encoders.md#timings).
2. **Send lists.** `/v1/embed` with 8 images is one forward pass. Client-side,
   `Remote.embed()` chunks to 8 for you.
3. **Keep `batch=True`** (the default) if callers send single images concurrently. It
   turns concurrency into batch size, which is where the GPU's throughput is.
4. **Add workers only for CPU work.** `uvicorn ... --workers N` gives you N processes,
   and **each one loads its own copy of the model**. That is N times the GPU memory, and
   they contend for the same device. On one GPU, batching beats workers. Workers are the right
   answer when decoding is your bottleneck (large JPEGs, CPU inference) or when you have N
   GPUs and pin one per worker with `CUDA_VISIBLE_DEVICES`.
5. **Scale out.** Several boxes behind the proxy, all pointed at the same heads directory
   on shared storage.

Heads are shared between processes through the directory, not through memory: `HeadStore`
re-reads any file whose mtime, inode or size changed. Trained on one worker, visible from
the others. Verified with two workers over the same directory:

```console
$ uvicorn serve_toy:app --host 127.0.0.1 --port 8125 --workers 2 &
$ grep -c "Started server process" w.log
2
$ curl -s -X POST http://127.0.0.1:8125/v1/heads -H 'Authorization: Bearer s3cret' \
    -H 'Content-Type: application/json' --data-binary @train2.json -o /dev/null -w "%{http_code}\n"
200
$ for i in 1 2 3 4 5 6; do curl -s http://127.0.0.1:8125/v1/health \
    | python3 -c "import sys,json;print('heads:',json.load(sys.stdin)['heads'])"; done
heads: 2
heads: 2
heads: 2
heads: 2
heads: 2
heads: 2
```

Every worker reports the head that exactly one of them trained. Shared storage has to be
real shared storage. Two nodes with separate disks are two different sets of heads, and a
client that trains on one and asks on the other gets a 404 half the time.

## Failure modes

**Changing `JEVIMAGE_ENCODER` invalidates every head.** A head records the encoder that fitted
it and refuses to run under another one. Silently wrong results are the alternative, and
the widths often match by accident:

```console
$ curl -s -w " [%{http_code}]\n" -X POST http://127.0.0.1:8732/v1/ask \
    -H 'Content-Type: application/json' \
    -d "{\"image\":\"$IMG\",\"questions\":[{\"type\":\"head\",\"head\":\"colour\"}]}"
{"detail":"head 'colour' was trained on encoder 'toy' but this instance runs 'toy-calibrated'. Load 'toy', or retrain the head here."} [422]
```

Roll the encoder and the heads together, or run the old encoder until every head is
retrained. `/v1/health` tells a client which encoder it is talking to; `GET /v1/heads`
tells it which encoder each head wants.

**Single-caption `noul` depends on the encoder.** The sigmoid form needs an encoder that
publishes both `logit_scale` and `logit_bias`. SigLIP does; CLIP-family models and every
ensemble do not. Same request, two servers:

```console
$ # server running a calibrated encoder
{"answers":{"dark":{"type":"noul","noul":1.0,"basis":"sigmoid readout"}},"timing":{...}}

$ # server running an uncalibrated one
{"detail":"encoder 'toy' has no calibrated sigmoid, so a single-caption noul cannot produce
a probability. Give a {\"true\": ..., \"false\": ...} criteria pair instead, which works on
every encoder."} [422]
```

If your callers are not all yours, serve an encoder with a bias or tell them to use the
pair form. See [questions.md](questions.md).

**Head names are global and unversioned.** `POST /v1/heads` with an existing name
overwrites it, and returns 200 either way. There is no per-caller namespace: one client's
`colour` is every client's `colour`. If several teams share a server, prefix the names, or
give each team its own process and heads directory.

**An unreadable head file does not disappear.** A truncated or foreign `.pt` is recorded
as broken rather than skipped, and asking for it gives a 404 whose body says the file is
present and why it would not load, not "no such head", which would be a lie about a file
sitting right there. That 404 is the only place a server shows it: `GET /v1/heads` lists
only the heads that loaded, and `/v1/health`'s `heads` count leaves it out. To see the
broken ones as a set, look at the heads directory, or run `jevimage heads --json` beside it.

**A restart drops the caption cache,** so the first request after a deploy re-encodes its
captions. Milliseconds, but it shows up as a p99 spike on a busy dashboard.

**Timeouts on training.** The default client timeout is 60 s and a real training upload
exceeds it. Raise it at `connect()`, and at the proxy, before blaming the server for a
head that appears anyway.

## See also

- [reference-http.md](reference-http.md): the exact request and response shapes.
- [install.md](install.md): what each extra pulls in.
- [encoders.md](encoders.md): the registry, ensembles, and writing your own.
- [reference-python.md](reference-python.md): `Jev.train` and `compare`: what
  `/v1/heads` fits.
- [reference-cli.md](reference-cli.md): `jevimage ask/train/heads/rm --url` against the
  server you just started.

---

[← Choosing an encoder](encoders.md) · [Docs index](index.md) · [CLI reference →](reference-cli.md)
