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

One more optional extra lets `choice` and `noul` read WHERE things are, not just what is
present, for the options only a grid can separate:

    .grid(list[PIL])   -> (n, patches, dim) fp32, L2-normalised, one row per patch
    .grid_side         int, patches per side (14 for a 16px patch at 224)

Only a transformers ViT can provide it, so `Ensemble` and `OpenCLIP` leave it off and a
question that needs it refuses rather than degrading to a pooled answer that would be at
chance (see core.Engine.grid_read).

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
# and `jevimage encoders` work in an API-only install - seeing which encoders exist should
# not require the stack that runs them.

import logging
import os
from collections.abc import Callable

from .store import env

__all__ = ["load", "register", "available", "check", "Encoder", "HFCLIP", "OpenCLIP", "Ensemble",
           "unit", "resolve_device", "resolve_dtype", "fast_cores", "tune_cpu_threads"]


_log = logging.getLogger("jevimage.encoders")


_TUNED = False


def fast_cores() -> int | None:
    """Physical performance cores on this machine, or None if they are all the same.

    Not a guess about topology: Linux publishes the hybrid split itself, under
    `/sys/devices/cpu_core` and `/sys/devices/cpu_atom`. Anywhere it does not - a
    uniform CPU, or not Linux - this returns None and nothing is changed.
    """
    try:
        if not os.path.isdir("/sys/devices/cpu_atom"):
            return None                 # uniform: torch's own default is right
        spec = open("/sys/devices/cpu_core/cpus").read().strip()
    except OSError:
        return None

    logical = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        try:
            logical.update(range(int(lo), int(hi or lo) + 1))
        except ValueError:
            return None
    # collapse hyperthread siblings: two threads on one core do not add a core's work
    cores = set()
    for c in logical:
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/"
                      "thread_siblings_list") as f:
                cores.add(f.read().strip())
        except OSError:
            cores.add(str(c))
    return len(cores) or None


def tune_cpu_threads() -> int | None:
    """Point torch at the fast cores when the two kinds differ. Returns what it set.

    torch defaults to the physical core count, which is wrong on a hybrid chip: every
    barrier in a forward pass waits on the slowest thread, so four E-cores set the pace
    for six P-cores as well. Measured on an i7-12650H (6 P + 4 E), one 224px forward:

        threads   1   2   3   4   5   6   8  10*  12   16
        ms      348 180 141 121 126  95 178 158  158  905     (* torch's default)

    Backing off to the six P-cores is 40% faster than the default and nine times faster
    than using all sixteen. `$JEVIMAGE_THREADS` sets it by hand, `JEVIMAGE_THREADS=0` turns this
    off, and an `$OMP_NUM_THREADS` already in the environment is left alone: a process
    that chose a thread count did so for the whole application, not just for us.
    """
    global _TUNED
    import torch

    if _TUNED:
        return None
    _TUNED = True

    want = env("JEVIMAGE_THREADS")
    if want:
        try:
            n = int(want)
        except ValueError:
            raise ValueError(f"$JEVIMAGE_THREADS={want!r} is not a number of threads") from None
        if n <= 0:
            return None                 # explicitly opted out
    elif os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS"):
        return None                     # already someone else's decision
    else:
        n = fast_cores()
        if not n or n >= torch.get_num_threads():
            return None                 # nothing to back off to

    torch.set_num_threads(n)
    _log.info("using %d cpu threads (JEVIMAGE_THREADS=0 to leave torch's default of %d)",
              n, torch.get_num_threads())
    return n


def check(name: str) -> None:
    """Refuse a name the registry does not have, without importing torch.

    `load()` needs this too, but only reaches it after transformers is in - so on an
    API-only install a typo'd `--encoder` answered "this needs torch". You installed
    torch, ran the same command, and only then learned the name was wrong. The CLI calls
    this first, and it is the same sentence either way.
    """
    for part in (p.strip() for p in str(name).split("+")):
        if part and part not in REGISTRY:
            raise KeyError(f"unknown encoder {part!r}. Available: {', '.join(available())}. "
                           f"Register your own with jevimage.register(name, factory) - "
                           f"from Python, in the process that will serve it; the `jevimage` "
                           f"command only sees these built-in names, because it never "
                           f"imports your code.")


def _place_with_fallback(enc, device: str | None) -> None:
    """Put `enc` on the best device it will actually hold, calling `enc._place()`.

    A GPU that exists is not a GPU with room. On a 4 GB laptop card a second copy of the
    model raises CUBLAS_STATUS_ALLOC_FAILED deep inside torch, and a classifier that
    refuses to run because something else is using the card is less useful than one that
    is slower. So CUDA is a preference: if it will not hold the model, fall back and say
    so once.

    A device you named - `device=` or `$JEVIMAGE_DEVICE` - is never silently swapped. You may
    have a reason to want that card and no other, and a job that quietly runs 40x slower
    is worse than one that stops.
    """
    wanted = resolve_device(device)
    if wanted == "cpu":
        tune_cpu_threads()
    try:
        enc._place(wanted)
        return
    except Exception as e:
        if wanted == "cpu" or device is not None or env("JEVIMAGE_DEVICE"):
            raise
        _log.warning("%s could not load on %s (%s); using the CPU instead",
                     enc.name, wanted, str(e).splitlines()[0][:90])
    _free_cuda()
    tune_cpu_threads()
    enc._place("cpu")


def _free_cuda() -> None:
    """Hand back whatever the failed attempt managed to allocate."""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_device(device: str | None = None) -> str:
    """Explicit argument, then $JEVIMAGE_DEVICE, then CUDA if present, then CPU.

    A bad value is refused here rather than inside `.to()`, which runs after the weights
    have loaded: a typo'd device is the same kind of user typo as a typo'd encoder name,
    and it should not cost a model download or print a torch traceback.
    """
    import torch
    dev = device or env("JEVIMAGE_DEVICE")
    if not dev:
        return "cuda" if torch.cuda.is_available() else "cpu"
    where = "device" if device else "$JEVIMAGE_DEVICE"
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
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.pad = pad

        _place_with_fallback(self, device)
        self.dim = int(self.txt(["x"]).shape[1])

    def _place(self, device: str) -> None:
        """Load onto one device. Separate so the CUDA attempt can be retried on CPU."""
        from transformers import AutoModel

        self.device = device
        self.dtype = resolve_dtype(device)
        self.m = AutoModel.from_pretrained(self.model_id,
                                           dtype=self.dtype).to(device).eval()
        # SigLIP exposes both; CLIP-family exposes scale only. Missing stays None so
        # core.py can tell "not calibrated" from "calibrated to 0".
        self.logit_scale = (self.m.logit_scale.exp().item()
                            if hasattr(self.m, "logit_scale") else None)
        self.logit_bias = (self.m.logit_bias.item()
                           if hasattr(self.m, "logit_bias") else None)

    def img(self, images) -> "torch.Tensor":
        import torch
        px = self.proc(images=[i.convert("RGB") for i in images],
                       return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            return unit(self.m.get_image_features(
                pixel_values=px.to(self.device, self.dtype)))

    def grid(self, images) -> "torch.Tensor":
        """Per-patch embeddings in the same space the text lives in, one forward pass.

        The pooled vector these models publish is out_proj(sum_i a_i * V x_i) plus the
        pooling head's tail. Dropping the attention weights a_i and keeping V x_i per
        patch gives each patch's own contribution in that same space - MaskCLIP's trick,
        applied to whichever pooling the checkpoint uses.

        Reading last_hidden_state directly instead does NOT work, and not by a little.
        Measured on 96 controlled images with one object on a textured background, asking
        which patch a phrase peaks on:

            raw last_hidden_state   12.5% correct, centroid error 2.68 patches
            this value projection  100.0% correct, centroid error 0.43 patches

        A random guess averages 5.3 patches of error, so the raw tokens are barely above
        noise. This is why patch tokens have a reputation for being unusable, and why
        crop-based methods pay 86-270 forward passes to avoid them.
        """
        import torch
        vm = self.m.vision_model
        while hasattr(vm, "vision_model"):
            vm = vm.vision_model
        px = self.proc(images=[i.convert("RGB") for i in images],
                       return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        with torch.no_grad():
            h = vm(pixel_values=px).last_hidden_state
            # SigLIP pools with a learned attention probe; CLIP pools a CLS token. Both
            # expose a value projection - for CLIP it is the last self-attention block.
            if hasattr(vm, "head") and hasattr(getattr(vm, "head", None), "attention"):
                att, d = vm.head.attention, h.shape[-1]
                W, b = att.in_proj_weight, att.in_proj_bias
                v = att.out_proj(h @ W[2 * d:].T + b[2 * d:])
                v = v + vm.head.mlp(vm.head.layernorm(v))
            else:
                blk = vm.encoder.layers[-1]
                h = h[:, 1:]                                  # the CLS token is not a patch
                v = blk.self_attn.out_proj(blk.self_attn.v_proj(h))
                v = vm.post_layernorm(v + blk.mlp(blk.layer_norm2(v)))
            proj = getattr(self.m, "visual_projection", None)
            if proj is not None:                              # CLIP: into the shared space
                v = proj(v)
            self.grid_side = int(round(v.shape[1] ** 0.5))
            return unit(v.reshape(-1, v.shape[-1])).reshape(v.shape[0], v.shape[1], -1)

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
        _place_with_fallback(self, device)
        self.dim = int(self.txt(["x"]).shape[1])

    def _place(self, device: str) -> None:
        """Load onto one device. Separate so the CUDA attempt can be retried on CPU."""
        import open_clip

        self.device = device
        self.dtype = resolve_dtype(device)
        self.m, _, self.pp = open_clip.create_model_and_transforms(
            f"hf-hub:{self.hub_id}")
        self.tok = open_clip.get_tokenizer(f"hf-hub:{self.hub_id}")
        self.m = self.m.to(device, self.dtype).eval()
        self.logit_scale = (self.m.logit_scale.exp().item()
                            if hasattr(self.m, "logit_scale") else None)
        self.logit_bias = None      # open_clip contrastive models have no pairwise bias

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
    name = name or env("JEVIMAGE_ENCODER") or DEFAULT
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
        check(name)
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
