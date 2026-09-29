"""The framework over HTTP - the same names, the same shapes, no new concepts.

    $ pip install 'jevimage[serve]'
    $ JEVIMAGE_ENCODER=siglip2-base-224 uvicorn jevimage.server:app

Every route under /v1 mirrors one method on `jevimage.Jev`, so the SDK docs are the
HTTP docs: /v1/ask is ask(), /v1/embed is embed(), /v1/heads is train/list/delete. A
client that speaks this API is `jevimage.connect(url)`, which is the same object you
get from `jevimage.load()`.

Images arrive as base64 - bare, or as a 'data:image/png;base64,...' URL. Everything
else about a request is the JSON the SDK already takes.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import time

import torch
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

import jevimage
from .answers import normalise_questions
from . import store, training
from .store import env

__all__ = ["build_app", "app"]

MAX_IMAGES = 8               # per ask/embed: one request should not monopolise the GPU
MAX_B64 = 12 * 1024 * 1024   # a 12MB base64 image is a ~9MB photo, far past any encoder's input
# Compressed size is no bound on decoded size: a solid-colour 12000x12000 PNG is 0.57MB
# of base64 and 1.1GB of RAM once decoded, so pixels are capped separately.
MAX_PIXELS = 50_000_000      # 50MP, well past any camera an encoder will see
# ... and the same reasoning across the batch: eight images each just under the per-image
# cap is 1.7MB on the wire and 1.2GB of decoded pixels before a single one is embedded,
# which a body-size cap at a proxy cannot see. 100MP still admits eight 12MP camera
# photos, which is the largest batch anyone sends without downscaling first.
MAX_TOTAL_PIXELS = 100_000_000
FORMATS = {"JPEG", "PNG", "WEBP"}
PRIMITIVES = ["choice", "score", "noul", "head"]


class AskIn(BaseModel):
    image: str | None = None
    images: list[str] | None = None
    questions: dict | list
    # bool first: `float` alone coerces JSON `true` to 1.0, which silently replaces the
    # encoder's calibrated default (~0.01) with a near-uniform distribution that still
    # looks like a valid answer. Kept as a bool, it reaches core._positive, which says so.
    temperature: bool | float | None = None


class EmbedIn(BaseModel):
    image: str | None = None
    images: list[str] | None = None


class Example(BaseModel):
    image: str
    label: str


class TrainIn(BaseModel):
    name: str
    examples: list[Example]
    # bounded because it is a loop count a caller can set: 5000 epochs on 4000 examples
    # is seconds, 5 million is an outage
    epochs: int = Field(500, ge=1, le=5000)


def _decode(b64: str, where: str):
    """One base64 string -> a PIL image that is certainly decodable and a format we serve."""
    if len(b64) > MAX_B64:
        # rounded up: floored, everything in [12MB, 13MB) reported "is 12MB ... the
        # limit is 12MB", a size and a limit that are the same number
        raise HTTPException(413, f"{where} is {-(-len(b64) // 1048576)}MB of base64; the "
                                 f"limit is {MAX_B64 // 1048576}MB per image")
    payload = b64.partition(",")[2] if b64.startswith("data:") else b64
    try:
        im = jevimage.to_image(base64.b64decode(payload))
    except Image.DecompressionBombError as e:
        # decoded fine, refused for size: saying "not a base64-encoded image" here sends
        # the caller down the wrong trail entirely
        # `e` already carries whichever guard fired and its number - ours at MAX_PIXELS
        # or PIL's own, which is larger - so naming a second limit here just misleads
        raise HTTPException(413, f"{where} is too large to decode: {e}")
    except Exception:
        raise HTTPException(422, f"{where} is not a base64-encoded image; send the raw "
                                 "bytes base64'd, or a 'data:image/png;base64,...' URL")
    # checked before any convert(), which is what drops .format
    if im.format not in FORMATS:
        raise HTTPException(415, f"{where} is {im.format or 'an unknown format'}; this "
                                 f"server reads {', '.join(sorted(FORMATS))}")
    # before im.load(): the header says how big it is, and allocating the pixels is the
    # expensive part - 1.1GB for a 12000x12000 PNG that arrived as 0.57MB of base64
    w, h = im.size
    if w * h > MAX_PIXELS:
        raise HTTPException(413, f"{where} is {w}x{h}; the limit is "
                                 f"{MAX_PIXELS // 10 ** 6}MP per image")
    try:
        im.load()       # PIL decodes lazily, so a truncated file would fail mid-request
    except Exception:
        raise HTTPException(422, f"{where} is a truncated or corrupt {im.format}")
    return im


def _images(one, many, limit=MAX_IMAGES, noun="images"):
    if one is not None and many:
        # they also select different response shapes, so picking one silently would
        # change what the caller gets back, not just which pixels were read
        raise HTTPException(422, "send 'image' or 'images', not both: 'image' returns one "
                                 "answer object, 'images' returns a list")
    srcs = [one] if one is not None else list(many or [])
    if not srcs:
        # `noun` is "examples" on the training route, whose model has no image/images
        # field at all, so the image wording would send that caller hunting for one
        raise HTTPException(422, "send 'image' (one base64 string) or 'images' (a list)"
                            if noun == "images" else f"send at least one {noun[:-1]}")
    if len(srcs) > limit:
        raise HTTPException(413, f"at most {limit} {noun} per request; got {len(srcs)}")
    out, pixels = [], 0
    for i, src in enumerate(srcs):
        im = _decode(src, f"{noun}[{i}]")
        pixels += im.size[0] * im.size[1]
        if pixels > MAX_TOTAL_PIXELS:
            raise HTTPException(413, f"{noun}[0..{i}] is {pixels // 10 ** 6}MP decoded in "
                                     f"total; the limit is {MAX_TOTAL_PIXELS // 10 ** 6}MP "
                                     "per request as well as "
                                     f"{MAX_PIXELS // 10 ** 6}MP per image")
        out.append(im)
    return out


# The store says "... in <dir> (have: ...)" and "... is present in <dir> but
# unreadable: ...". Anchoring on those two tails strips the directory however it was
# spelled - absolute, relative, or containing spaces - and leaves other messages alone.
_PATHLIKE = re.compile(r"\s+in\s+.+?(?=\s+\(have:|\s+but unreadable:)")


def _no_paths(detail: str) -> str:
    """Strip a filesystem path out of a message on its way to a client."""
    return _PATHLIKE.sub("", detail)


def _guard(keys: list[str]):
    def check(authorization: str | None = Header(None)):
        if not keys:
            return
        # split, not startswith("Bearer ")/removeprefix: RFC 7235 2.1 makes the scheme
        # token case-insensitive, and a lowercase `bearer` from a client or proxy would
        # otherwise 401 a caller whose key was right
        parts = (authorization or "").split(None, 1)
        sent = parts[1].strip() if len(parts) == 2 and parts[0].lower() == "bearer" else ""
        # compared as bytes: compare_digest raises TypeError on a non-ASCII str, and a
        # header is arbitrary bytes, so a str comparison turns a junk key into a 500.
        # latin-1 is how the header was decoded, so this recovers what was on the wire.
        raw = sent.encode("latin-1", "replace")
        if not any(hmac.compare_digest(raw, k.encode("utf-8")) for k in keys):
            raise HTTPException(401, "send 'Authorization: Bearer <key>' with a key this "
                                     "server was started with (JEVIMAGE_API_KEYS)")
    return check


def build_app(jev=None, *, api_keys=None) -> FastAPI:
    """The app. `jev` defaults to `jevimage.load()`, which reads JEVIMAGE_ENCODER and
    JEVIMAGE_HEADS_DIR itself, so there is no second place that parses those."""
    if jev is None:
        jev = jevimage.load(batch=True)      # a server is exactly what the batcher is for
    if api_keys is None:
        # commas or whitespace: the CLI spells it `--api-key K1 K2`, so a unit file that
        # mirrors that spelling must not end up with one key containing a space
        api_keys = env("JEVIMAGE_API_KEYS", "").replace(",", " ").split()
    # an empty string is not a key: api_keys=[env("JEVIMAGE_KEY", "")] on a host
    # where the variable is unset would otherwise build a guard that accepts a missing
    # header (compare_digest(b"", b"") is true) while /v1/health reports "bearer"
    api_keys = [k for k in (api_keys or []) if k]

    app = FastAPI(title="jevimage", version=jevimage.__version__)

    # The core's ValueErrors are the product - "head 'x' was trained on encoder 'y'" is
    # the answer, not a symptom - so they reach the caller intact, as 422.
    @app.exception_handler(ValueError)
    def _bad_request(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(KeyError)
    def _missing(request, exc):
        # The store names the directory it searched, which is the right thing to tell
        # an operator at a REPL and the wrong thing to hand a client: it discloses the
        # server's filesystem layout to anyone who can guess a head name.
        detail = _no_paths(exc.args[0] if exc.args else str(exc))
        return JSONResponse({"detail": detail}, status_code=404)

    # health stays open even when keys are set: a load balancer that needs a secret to
    # ask whether the process is alive is a load balancer that reports it is not
    @app.get("/v1/health")
    def health():
        return {"encoder": jev.encoder.name, "dim": jev.encoder.dim,
                "device": getattr(jev.encoder, "device", "cpu"),
                "primitives": PRIMITIVES, "heads": len(jev.heads.sync()),
                "auth": "bearer" if api_keys else "open",
                "version": jevimage.__version__}

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(_guard(api_keys))])

    @v1.post("/ask")
    def ask(body: AskIn):
        t0 = time.perf_counter()
        ims = _images(body.image, body.images)
        qs = normalise_questions(body.questions)
        if jev.engine.needs_grid(qs):
            # A grid-read question needs the patch grid, which an embedding no longer
            # records, so
            # these go through ask() per image. Still one vision pass each - the grid and
            # the pooled vector come out of the same forward.
            out = [jev.ask(im, qs, temperature=body.temperature) for im in ims]
            embed_ms = (time.perf_counter() - t0) * 1000
        else:
            embs = jev.embed(ims)
            embed_ms = (time.perf_counter() - t0) * 1000
            # one encode, then every question read against it - the point of the engine
            out = [jev.ask_embedding(e, qs, temperature=body.temperature) for e in embs]
        return {"answers": out[0] if body.image is not None else out,
                "timing": {"embed_ms": round(embed_ms, 1),
                           "total_ms": round((time.perf_counter() - t0) * 1000, 1)}}

    @v1.post("/embed")
    def embed(body: EmbedIn):
        embs = jev.embed(_images(body.image, body.images))
        return {"embeddings": embs.tolist(), "dim": int(embs.shape[1])}

    @v1.get("/heads")
    def list_heads():
        return jev.list_heads()

    @v1.get("/heads/{name}")
    def get_head(name: str):
        return training.public(name, jev.heads[name])

    @v1.get("/heads/{name}/compare")
    def compare_head(name: str):
        # a head keeps the embeddings it was fitted on, so "did training help" stays
        # answerable long after the training call returned
        cmp = jev.compare(name)
        if cmp is None:
            raise HTTPException(409, f"head {name!r} was saved without its embeddings, "
                                     "so it cannot be compared; retrain it to restore this")
        return cmp

    @v1.post("/heads")
    def train_head(body: TrainIn):
        # the MAX_IMAGES cap is about latency per request; training is a bulk upload and
        # is bounded by the core's own MAX_EXAMPLES instead
        # the name is checked before anything is decoded: a typo used to cost the whole
        # upload plus the encode pass, and only then 422
        store.valid_name(body.name)
        ims = _images(None, [e.image for e in body.examples],
                      limit=training.MAX_EXAMPLES, noun="examples")
        head = jev.train(body.name, list(zip(ims, (e.label for e in body.examples))),
                         epochs=body.epochs)
        return {**training.public(body.name, jev.heads[body.name]),
                "comparison": head.comparison}

    @v1.delete("/heads/{name}")
    def delete_head(name: str):
        if not jev.delete_head(name):
            # matches what GET /heads/{name} says, so the two routes do not disagree
            # about whether the same head exists
            raise HTTPException(404, f"no head named {name!r}")
        return {"deleted": name}

    app.include_router(v1)
    return app


def __getattr__(name):
    """`uvicorn jevimage.server:app` builds the app here, on first touch, so importing
    this module for build_app() never loads a model."""
    if name == "app":
        globals()["app"] = build_app()
        return globals()["app"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _Toy(jevimage.Encoder):
    """Deterministic, no download: the routes are what is under test, not a model.

    Module scope rather than inside demo(), because jevimage.client's selftest drives a
    real socket against this module and needs the same encoder on both ends of it.
    """

    name, dim = "toy", 16

    def __init__(self):
        torch.manual_seed(0)
        self.W = torch.randn(64, self.dim)

    def _e(self, xs):
        v = torch.stack([torch.tensor([b / 255 for b in
                                       hashlib.sha512(str(x).encode()).digest()])
                         for x in xs])
        return torch.nn.functional.normalize(v @ self.W, dim=-1)

    def img(self, ims): return self._e([i.tobytes() for i in ims])
    def txt(self, ts): return self._e(list(ts))


def demo():
    import io
    import tempfile

    from fastapi.testclient import TestClient

    def png(rgb, i):
        im = Image.new("RGB", (32, 32), rgb)
        im.putpixel((i, i), (0, 0, 0))     # distinct bytes per example, as real data is
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    red = [png((200, 30, 30), i) for i in range(6)]
    blue = [png((30, 30, 200), i) for i in range(6)]

    with tempfile.TemporaryDirectory() as tmp:
        jev = jevimage.load(_Toy(), heads_dir=tmp)
        c = TestClient(build_app(jev, api_keys=["secret"]))
        auth = {"Authorization": "Bearer secret"}

        h = c.get("/v1/health").json()
        assert h["encoder"] == "toy" and h["dim"] == 16, h
        assert h["primitives"] == PRIMITIVES and h["auth"] == "bearer", h

        r = c.post("/v1/ask", headers=auth, json={
            "image": red[0],
            "questions": {"colour": {"type": "choice",
                                     "criteria": {"red": "a red square",
                                                  "blue": "a blue square"}}}})
        assert r.status_code == 200, r.text
        a = r.json()["answers"]["colour"]
        assert a["choice"] in ("red", "blue") and abs(sum(a["probabilities"].values()) - 1) < 1e-4
        assert r.json()["timing"]["embed_ms"] <= r.json()["timing"]["total_ms"]

        e = c.post("/v1/embed", headers=auth, json={"images": [red[0], blue[0]]})
        assert e.json()["dim"] == 16 and len(e.json()["embeddings"]) == 2, e.text

        # train -> ask -> delete, the round trip the docs promise
        ex = [{"image": p, "label": "red"} for p in red]
        ex += [{"image": p, "label": "blue"} for p in blue]
        t = c.post("/v1/heads", headers=auth, json={"name": "colour", "examples": ex})
        assert t.status_code == 200, t.text
        assert t.json()["classes"] == ["blue", "red"] and t.json()["n"] == len(ex)
        assert "W" not in t.json() and t.json()["comparison"]["overall"]["zero_shot"] is not None
        assert [x["name"] for x in c.get("/v1/heads", headers=auth).json()] == ["colour"]
        assert c.get("/v1/heads/colour", headers=auth).json()["encoder"] == "toy"

        hq = c.post("/v1/ask", headers=auth, json={
            "image": red[0], "questions": [{"type": "head", "head": "colour"}]})
        assert hq.json()["answers"]["q1"]["choice"] in ("red", "blue"), hq.text
        assert c.delete("/v1/heads/colour", headers=auth).json() == {"deleted": "colour"}
        assert c.get("/v1/heads", headers=auth).json() == []

        # a missing head is 404 whichever way you reach for it, and the message says so
        assert c.get("/v1/heads/nope", headers=auth).status_code == 404
        assert c.delete("/v1/heads/nope", headers=auth).status_code == 404
        miss = c.post("/v1/ask", headers=auth, json={
            "image": red[0], "questions": [{"type": "head", "head": "nope"}]})
        assert miss.status_code == 422 and "unknown head" in miss.json()["detail"], miss.text

        # every malformed question is a 422 with a sentence, in both shapes; these were
        # 500s, and a list question was never type-checked at all
        for qs in ([None], ["hi"], [5], [[]],
                   {"a": {"type": "choice", "prompt": 5, "criteria": {"x": "a", "y": "b"}}},
                   {"a": {"type": "choice", "prompt": "a photo of {thing}",
                          "criteria": {"x": "a", "y": "b"}}},
                   {"a": {"type": "head", "head": ["x"]}}):
            bad = c.post("/v1/ask", headers=auth, json={"image": red[0], "questions": qs})
            assert bad.status_code == 422, (qs, bad.status_code, bad.text)
            assert isinstance(bad.json()["detail"], str), bad.text

        # a JSON bool must not be coerced into a temperature: 1.0 is a near-uniform
        # distribution on a real encoder and the answer would look fine
        t = c.post("/v1/ask", headers=auth, json={
            "image": red[0], "temperature": True,
            "questions": {"a": {"type": "choice", "criteria": {"x": "a", "y": "b"}}}})
        assert t.status_code == 422 and "must be a number" in t.json()["detail"], t.text

        # the training route has no image/images field, so its empty-body message
        # must not name one
        mt = c.post("/v1/heads", headers=auth, json={"name": "zz", "examples": []})
        assert mt.status_code == 422 and "example" in mt.json()["detail"], mt.text
        assert "images" not in mt.json()["detail"], mt.text

        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="GIF")
        gif = c.post("/v1/embed", headers=auth,
                     json={"image": base64.b64encode(buf.getvalue()).decode()})
        assert gif.status_code == 415 and "GIF" in gif.json()["detail"], gif.text

        for hdr in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "secret"}):
            assert c.post("/v1/embed", headers=hdr,
                          json={"image": red[0]}).status_code == 401, hdr
        # a non-ASCII key is still just a wrong key, not a 500 an anonymous caller can
        # trigger - hmac.compare_digest refuses non-ASCII str
        try:
            _guard(["secret"])("Bearer éé")
            raise AssertionError("non-ASCII bearer was accepted")
        except HTTPException as e:
            assert e.status_code == 401, e.status_code
        # the scheme token is case-insensitive (RFC 7235 2.1); the key is not
        for hdr in ("Bearer secret", "bearer secret", "BEARER secret", "Bearer  secret"):
            assert c.post("/v1/embed", headers={"Authorization": hdr},
                          json={"image": red[0]}).status_code == 200, hdr
        assert c.get("/v1/health").status_code == 200      # health never needs the key

        assert TestClient(build_app(jev)).get("/v1/health").json()["auth"] == "open"
        # a list of empty strings is no keys at all, and must not claim otherwise
        blank = TestClient(build_app(jev, api_keys=[""]))
        assert blank.get("/v1/health").json()["auth"] == "open"
        assert blank.get("/v1/heads").status_code == 200

        # the per-image pixel cap is also a per-request one: eight images just under it
        # is 1.2GB of decoded pixels that no body-size cap at a proxy can see
        wide = io.BytesIO()
        Image.new("RGB", (10000, 5000), (7, 7, 7)).save(wide, format="PNG")
        big = base64.b64encode(wide.getvalue()).decode()
        over = c.post("/v1/embed", headers=auth, json={"images": [big] * 8})
        assert over.status_code == 413 and "MP" in over.json()["detail"], over.text
        assert c.post("/v1/embed", headers=auth,
                      json={"images": [big]}).status_code == 200

        # a 404 body must not name the heads directory, however that directory was
        # spelled: a server started with a relative --heads-dir leaked it once
        for d in ("/srv/jev/heads", "heads", "./h", "../h", "/tmp/my heads", r"C:\jev"):
            assert d not in _no_paths(f"no head named 'nope' in {d} (have: none)"), d
            assert d not in _no_paths(f"junk.pt is present in {d} but unreadable: X. "
                                      "Delete it, or retrain the head."), d

    print("server selftest ok - health, choice ask, embed, train->ask->delete, 404 on a "
          "missing head both ways, no heads directory in a 404 body, 415 on a GIF, "
          "401 on a bad key in any scheme case, open mode says open, 422 (not 500) on "
          "every malformed question, 413 past the per-request pixel budget")


if __name__ == "__main__":
    demo()
