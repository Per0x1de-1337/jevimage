"""Encoders: the swappable part of the framework.

Everything above this file works on unit-norm embeddings and does not care where they
came from. An encoder is any object with four things:

    .name              str, for provenance - a head records which encoder fitted it
    .dim               int, embedding width
    .img(list[PIL])    -> (n, dim) fp32, L2-normalised
    .txt(list[str])    -> (n, dim) fp32, L2-normalised

Two optional extras let `noul` use a calibrated sigmoid instead of a contrast pair.
Leave them off entirely and the engine reads them as absent:

    .logit_scale       float, multiplies the cosine before the sigmoid
    .logit_bias        float, shifts it

SigLIP provides both because it was trained with a pairwise sigmoid loss. CLIP-family
models have a scale but no bias, and the framework degrades honestly rather than
inventing one (see core.noul).

Pick one by name:

    jevimage.load("siglip2-base-224")                    # the default
    jevimage.load("siglip2-so400m-384")                  # bigger, slower, sharper
    jevimage.load("siglip2-base-224+dfn5b-h-14-384")     # ensemble, '+' joins members

or bring your own:

    jevimage.register("mine", lambda: MyEncoder())
    jevimage.load("mine")
"""
from __future__ import annotations

# torch is imported inside the functions that use it, not here, so that `available()`
# and `jev encoders` work in an API-only install - seeing which encoders exist should
# not require the stack that runs them.

import os
from collections.abc import Callable

__all__ = ["load", "register", "available", "Encoder", "HFCLIP", "OpenCLIP", "Ensemble",
           "unit", "resolve_device", "resolve_dtype"]


def resolve_device(device: str | None = None) -> str:
    """Explicit argument, then $JEV_DEVICE, then CUDA if present, then CPU.

    A bad value is refused here rather than inside `.to()`, which runs after the weights
    have loaded: a typo'd device is the same kind of user typo as a typo'd encoder name,
    and it should not cost a model download or print a torch traceback.
    """
    import torch
    dev = device or os.environ.get("JEV_DEVICE")
    if not dev:
        return "cuda" if torch.cuda.is_available() else "cpu"
    where = "device" if device else "$JEV_DEVICE"
    try:
        parsed = torch.device(dev)
    except (RuntimeError, ValueError) as e:
        raise ValueError(f"{where}={dev!r} is not a device torch knows: "
                         f"{str(e).strip().splitlines()[0]}") from None
    if parsed.type == "cuda" and (parsed.index or 0) >= torch.cuda.device_count():
        have = torch.cuda.device_count()
        raise ValueError(f"{where}={dev!r}: this machine has {have} CUDA device(s); use "
                         + (f"cuda:0..cuda:{have - 1}, or cpu" if have else "cpu"))
    return dev


def resolve_dtype(device: str) -> "torch.dtype":
    """bfloat16 on GPU measured 4x the throughput of fp32 with no accuracy change;
    on CPU bf16 is slower than fp32, so CPU stays fp32."""
    import torch
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def unit(out) -> "torch.Tensor":
    """Pull the tensor out of whatever the model returned and L2-normalise it in fp32.

    Always fp32: bf16 has too few mantissa bits for a stable softmax, and a probability
    that flips with the dtype is not a probability.
    """
    import torch
    if torch.is_tensor(out):
        t = out
    elif isinstance(out, (tuple, list)):
        t = out[0]
    else:
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            v = getattr(out, attr, None)
            if torch.is_tensor(v):
                t = v
                break
        else:
            raise TypeError(f"no embedding tensor on {type(out).__name__}")
    t = t.float()
    if t.ndim == 3:            # token-level output: mean-pool to one vector per item
        t = t.mean(1)
    n = t.norm(dim=-1, keepdim=True)
    return t / n.clamp_min(1e-12)      # a zero row would otherwise become nan


class Encoder:
    """The interface. Subclass it for a custom model, or duck-type the four members."""

    name: str = "custom"
    dim: int = 0
    logit_scale: float | None = None
    logit_bias: float | None = None

    def img(self, images) -> "torch.Tensor":
        raise NotImplementedError

    def txt(self, texts: list[str]) -> "torch.Tensor":
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} {self.name} dim={self.dim}>"


class HFCLIP(Encoder):
    """Any transformers dual encoder: SigLIP, SigLIP2, CLIP, MetaCLIP."""

    def __init__(self, model_id: str, *, name: str | None = None, pad: str = "max_length",
                 device: str | None = None):
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.name = name or model_id
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device)
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.m = AutoModel.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        self.pad = pad
        # SigLIP exposes both; CLIP-family exposes scale only. Missing stays None so
        # core.py can tell "not calibrated" from "calibrated to 0".
        self.logit_scale = (self.m.logit_scale.exp().item()
                            if hasattr(self.m, "logit_scale") else None)
        self.logit_bias = (self.m.logit_bias.item()
                           if hasattr(self.m, "logit_bias") else None)
        self.dim = int(self.txt(["x"]).shape[1])

    def img(self, images) -> "torch.Tensor":
        import torch
        px = self.proc(images=[i.convert("RGB") for i in images],
                       return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            return unit(self.m.get_image_features(
                pixel_values=px.to(self.device, self.dtype)))

    def txt(self, texts: list[str]) -> "torch.Tensor":
        # SigLIP was trained on fixed-length 64-token captions and degrades measurably
        # with dynamic padding; CLIP and MetaCLIP want the opposite.
        # truncation on both branches: without it a caption over 64 whitespace tokens
        # reached the model as 65 positions and died with a transformers error about
        # max_position_embeddings - naming nothing the caller wrote, and advising an
        # argument the caller cannot reach. 64 tokens is the model's ceiling either way.
        kw = (dict(padding="max_length", max_length=64, truncation=True)
              if self.pad == "max_length"
              else dict(padding=True, truncation=True, max_length=77))
        batch = self.proc(text=list(texts), return_tensors="pt", **kw).to(self.device)
        batch.pop("token_type_ids", None)
        import torch
        with torch.no_grad():
            return unit(self.m.get_text_features(**batch))


class OpenCLIP(Encoder):
    """An open_clip checkpoint from the HF hub (DFN5B, EVA-CLIP, LAION CLIP).

    Requires `pip install jevimage[openclip]`.
    """

    def __init__(self, hub_id: str, *, name: str | None = None, device: str | None = None):
        try:
            import open_clip
        except ImportError as e:
            # the registry's one entry behind an optional extra, and the bare
            # ModuleNotFoundError named neither the encoder nor the extra that fixes it
            raise ImportError(
                f"encoder {name or hub_id!r} runs on open_clip: "
                "pip install 'jevimage[openclip]'\n(its weights carry Apple's own "
                f"licence - check it before redistributing.)\n"
                f"(underlying import error: {e})") from e

        self.hub_id = hub_id
        self.name = name or hub_id
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device)
        self.m, _, self.pp = open_clip.create_model_and_transforms(f"hf-hub:{hub_id}")
        self.tok = open_clip.get_tokenizer(f"hf-hub:{hub_id}")
        self.m = self.m.to(self.device, self.dtype).eval()
        self.logit_scale = (self.m.logit_scale.exp().item()
                            if hasattr(self.m, "logit_scale") else None)
        self.logit_bias = None      # open_clip contrastive models have no pairwise bias
        self.dim = int(self.txt(["x"]).shape[1])

    def img(self, images) -> "torch.Tensor":
        import torch
        px = torch.stack([self.pp(i.convert("RGB")) for i in images])
        with torch.no_grad():
            return unit(self.m.encode_image(px.to(self.device, self.dtype)))

    def txt(self, texts: list[str]) -> "torch.Tensor":
        import torch
        with torch.no_grad():
            return unit(self.m.encode_text(self.tok(list(texts)).to(self.device)))


class Ensemble(Encoder):
    """Several encoders concatenated, then renormalised.

    Because each member is already unit-norm, the cosine of the concatenation is the
    MEAN of the members' cosines - so an ensemble is an average of opinions, not a
    longer vector with arbitrary weighting. Costs the sum of the members' latency.
    """

    def __init__(self, members: list[Encoder], name: str | None = None):
        if len(members) < 2:
            raise ValueError("an ensemble needs at least 2 members")
        self.members = members
        self.name = name or "+".join(m.name for m in members)
        self.dim = sum(m.dim for m in members)
        # The concatenated cosine is the mean of the members', so the mean of their
        # scales is the scale that reads it - without this an ensemble of SigLIPs
        # silently dropped to the CLIP-convention temperature.
        # getattr: .logit_scale is optional in the encoder contract, so a member that
        # omits it must not turn ensembling into an AttributeError (core.py does the same).
        scales = [getattr(m, "logit_scale", None) for m in members]
        self.logit_scale = (sum(scales) / len(scales)
                            if all(x is not None for x in scales) else None)
        # The bias does not average: it was fitted against each member's own negative
        # set, and there is no pair to calibrate across a concatenation. So the sigmoid
        # noul stays refused on an ensemble rather than reporting a wrong probability.
        self.logit_bias = None

    # unit() per MEMBER, then again on the concatenation. Normalising only the result
    # let a member whose output is not unit-norm outvote the others - a 3x member takes
    # 9/10 of the cosine instead of 1/2 - and hid it, because the engine's L2 check sees
    # only the renormalised concatenation. That check is what catches the mistake on a
    # solo encoder; ensembling must not be the way to smuggle it past.
    def img(self, images) -> "torch.Tensor":
        import torch
        return unit(torch.cat([unit(m.img(images)) for m in self.members], -1))

    def txt(self, texts: list[str]) -> "torch.Tensor":
        import torch
        return unit(torch.cat([unit(m.txt(texts)) for m in self.members], -1))


# Names are stable identifiers: a head records the one that fitted it, so renaming a
# registry entry invalidates heads. Add, do not rename.
REGISTRY: dict[str, Callable[[], Encoder]] = {
    "siglip2-base-224":   lambda: HFCLIP("google/siglip2-base-patch16-224",
                                         name="siglip2-base-224"),
    "siglip2-base-384":   lambda: HFCLIP("google/siglip2-base-patch16-384",
                                         name="siglip2-base-384"),
    "siglip2-so400m-384": lambda: HFCLIP("google/siglip2-so400m-patch16-384",
                                         name="siglip2-so400m-384"),
    "siglip2-giant-384":  lambda: HFCLIP("google/siglip2-giant-opt-patch16-384",
                                         name="siglip2-giant-384"),
    "clip-vit-l-14":      lambda: HFCLIP("openai/clip-vit-large-patch14",
                                         name="clip-vit-l-14", pad="longest"),
    "metaclip2-huge":     lambda: HFCLIP("facebook/metaclip-2-worldwide-huge-quickgelu",
                                         name="metaclip2-huge", pad="longest"),
    # Apple's DFN5B weights carry Apple's own licence - check it before redistributing.
    # The checkpoint is 378px; Apple renamed the repo from -384 to -378, and the old id
    # resolves only through a hub redirect. The KEY stays -384 because heads record it.
    "dfn5b-h-14-384":     lambda: OpenCLIP("apple/DFN5B-CLIP-ViT-H-14-378",
                                           name="dfn5b-h-14-384"),
}

DEFAULT = "siglip2-base-224"

_LOADED: dict[str, Encoder] = {}


def register(name: str, factory: Callable[[], Encoder]) -> None:
    """Add an encoder to the registry. `factory` is called with no arguments the first
    time `load(name)` asks for it, and must return an object with the Encoder members.

        jevimage.register("mine", lambda: MyEncoder())
    """
    if "+" in name:
        raise ValueError("'+' is reserved for ensembles, so a name cannot contain it")
    if not callable(factory):
        raise TypeError("factory must be callable and take no arguments")
    REGISTRY[name] = factory
    _LOADED.pop(name, None)      # a re-register replaces any cached instance
    # and any cached ensemble built from it, which otherwise kept handing back the
    # instance the OLD factory made, under a name that says otherwise
    for key in [k for k in _LOADED if name in [p.strip() for p in k.split("+")]]:
        _LOADED.pop(key, None)


def available() -> list[str]:
    """Registry names, sorted. Ensembles are formed on demand and are not listed."""
    return sorted(REGISTRY)


def load(name: str | None = None) -> Encoder:
    """Load an encoder by registry name.

    Several names joined by '+' build an Ensemble: load("siglip2-base-224+dfn5b-h-14-384").
    Instances are cached per name, so calling this twice loads the weights once.
    """
    name = name or os.environ.get("JEV_ENCODER") or DEFAULT
    if name in _LOADED:
        return _LOADED[name]

    if "+" in name:
        parts = [p.strip() for p in name.split("+")]
        if not all(parts) or len(parts) < 2:
            raise ValueError(f"malformed ensemble name {name!r}: join two or more "
                             "encoder names with single '+', e.g. "
                             f"{DEFAULT}+dfn5b-h-14-384")
        enc = Ensemble([load(p) for p in parts], name="+".join(parts))
    else:
        if name not in REGISTRY:
            raise KeyError(f"unknown encoder {name!r}. Available: {', '.join(available())}. "
                           f"Register your own with jevimage.register(name, factory) - "
                           f"from Python, in the process that will serve it; the `jev` "
                           f"command only sees these built-in names, because it never "
                           f"imports your code.")
        enc = REGISTRY[name]()
        # The registry key is the provenance: a head records this string and is refused
        # under anything else. Letting a factory return its own .name meant load("x")
        # and the same object built directly disagreed about what fitted a head.
        prior = next((k for k, v in _LOADED.items() if v is enc), None)
        if prior is not None and prior != name:
            # Stamping here would RENAME the instance already cached under `prior`, so
            # every head trained as `prior` is refused - and the refusal tells you to
            # load `prior`, which now returns an encoder called something else. Provenance
            # is a correctness guarantee here, so this is an error, not a silent rename.
            raise ValueError(
                f"the factory for {name!r} returned the encoder already loaded as "
                f"{prior!r}. Two registry names must not share one instance, because a "
                f"head records the name it was fitted under. Give {name!r} a factory "
                f"that builds its own instance, or just load {prior!r}.")
        enc.name = name

    _LOADED[name] = enc
    return enc
