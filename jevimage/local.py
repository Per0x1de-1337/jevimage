"""Running the encoder yourself: `jevimage.load()`.

Importing this module pulls in torch and transformers, which is why `import jevimage`
does not - an API-only install never touches it.
"""
from __future__ import annotations

import time

from PIL import ImageOps

import torch

from . import training
from .answers import Answer, check_template, normalise_questions
from .core import Engine
from .encoders import Encoder
from .encoders import load as load_encoder
from .images import Head, as_examples, to_image
from .store import HeadStore, valid_name

__all__ = ["Jev", "load"]


class Jev:
    """A loaded encoder plus its heads. Create one with `jevimage.load()`.

    Thread-safe for reads. `train` and `delete_head` write to the head directory.
    """

    def __init__(self, encoder=None, *, heads_dir=None, temperature=None, batch=False):
        # a name (or None for the default) is looked up; anything else is used as-is,
        # which is how a custom encoder gets in without registering it
        self.encoder = (load_encoder(encoder) if encoder is None or isinstance(encoder, str)
                        else encoder)
        self.heads = HeadStore(heads_dir)
        self.engine = Engine(self.encoder, self.heads, temperature)
        self._batcher = None
        if batch:
            from .batcher import Batcher
            # Requests arriving within a few ms ride one forward pass. Worth it in a
            # server, pure overhead in a script, hence off by default.
            self._batcher = Batcher(lambda ims: list(self._embed_now(ims)),
                                    max_batch=16, max_wait=0.006, name="embed")

    # -- embedding -------------------------------------------------------
    def _embed_now(self, images):
        # exif_transpose first: a portrait phone photo carries Orientation=6 and would
        # otherwise be encoded sideways, so every probability read off it would be the
        # probability for a rotated image. The server decodes into this same path, so
        # local and remote stay identical.
        return self.encoder.img([ImageOps.exif_transpose(to_image(i)).convert("RGB")
                                 for i in images])

    def embed(self, images):
        """Unit embeddings for one image or a list. Returns (n, dim) fp32.

        This is the whole cost of a request. Everything else is a matmul, so if you
        ask the same image many questions over time, cache what this returns.
        """
        single = not isinstance(images, (list, tuple))
        batch = [images] if single else list(images)
        if not batch:
            raise ValueError("embed() needs at least one image")
        if self._batcher is not None and len(batch) == 1:
            return self._batcher.submit(batch[0]).unsqueeze(0)
        return self._embed_now(batch)

    # -- asking ----------------------------------------------------------
    def ask(self, image, questions, *, temperature=None) -> dict[str, Answer]:
        """Answer every question about one image. Returns {question id: Answer}.

        `questions` is {id: question} or a list (ids become q1, q2, ...).

        One image. A list used to be encoded in full and then answered for its first
        element only - full cost, silently discarded - so it is refused.
        """
        if isinstance(image, (list, tuple)):
            raise TypeError(
                f"ask() takes one image, got a {type(image).__name__} of {len(image)}. "
                "Loop over them, or call embed() once and ask_embedding() per row.")
        return self.engine.answer_all(self.embed(image)[0:1],
                                      normalise_questions(questions), temperature)

    def ask_embedding(self, embedding, questions, *, temperature=None):
        """The same, against an embedding you already have. Skips the encoder entirely."""
        # The contract is a (dim,) or (n, dim) tensor, and it was never checked: a numpy
        # array, a JSON round-tripped list and a stray batch dim each died somewhere
        # inside the engine - the 3-D case as far in as "type list doesn't define
        # __round__", which tells the caller nothing about what it handed over.
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                f"ask_embedding() takes a torch.Tensor of shape (dim,) or (n, dim), got "
                f"{type(embedding).__name__}; wrap it with torch.as_tensor()")
        if embedding.ndim not in (1, 2):
            raise ValueError(f"embedding must be (dim,) or (n, dim), got shape "
                             f"{tuple(embedding.shape)}")
        if embedding.shape[-1] != self.encoder.dim:
            raise ValueError(f"embedding is {embedding.shape[-1]}-dim, but encoder "
                             f"{self.encoder.name!r} produces {self.encoder.dim}")
        emb = embedding if embedding.ndim == 2 else embedding.unsqueeze(0)
        return self.engine.answer_all(emb[0:1], normalise_questions(questions),
                                      temperature)

    # -- training --------------------------------------------------------
    def train(self, name, data, *, epochs=500, max_per_class=None, template=None) -> Head:
        """Fit a head on your own labelled images.

        `data` is a folder (one sub-folder per label), a {label: [images]} mapping, or
        a list of (image, label) pairs. Images are anything `to_image` accepts.

        Only a linear layer over the frozen encoder is fitted, so this is seconds, not
        hours, and the result is tens of kilobytes. The returned Head carries held-out
        accuracy per class and a comparison against zero-shot prompts, so you can see
        whether training actually helped - per class, not just on average.
        """
        valid_name(name)          # before the encoder runs, not after
        if template is not None:
            # before the encoder runs, and before the head is written: a bad template
            # used to raise out of the compare() below, AFTER the head was on disk, so
            # the caller saw a failure and had a working head they did not know about
            check_template(template, "template")
        images, labels = as_examples(data, max_per_class)
        if len(images) > training.MAX_EXAMPLES:
            raise ValueError(f"at most {training.MAX_EXAMPLES} examples; got {len(images)}")
        embs = self.embed(images)
        head = training.train(embs, labels, epochs=epochs)
        head["encoder"] = self.encoder.name
        head["created"] = time.time()
        saved = self.heads.save(name, head)
        # training.public(), not the raw record: that is what a server hands back, and
        # `connect()` is a drop-in, so both sides' to_dict() must carry the same keys.
        # The weights stay reachable as jev.heads[name] for anyone who wants them.
        return Head(name, training.public(name, saved),
                    self.compare(name, template=template))

    def compare(self, name, *, template=None):
        """Trained vs zero-shot prompts on the same images, per class.

        The trained side is the held-out number; the zero-shot side has nothing to hold
        out, so it is scored on every example. Returns None for a head saved without
        its embeddings.
        """
        valid_name(name)          # the same refusal delete_head() and Remote give
        if template is not None:
            # `template or "a photo of {}"` silently swallowed "", and a template with no
            # placeholder captioned every class identically - so the zero-shot side was
            # an argmax tie reported as a measurement, and `delta` said training helped
            check_template(template, "template")
        head = self.heads[name]
        # the same guard ask() runs: comparing this encoder's prompts against another
        # encoder's stored embeddings returns numbers that mean nothing
        self.engine._check_head(name, head, self.encoder.dim)
        if "emb" not in head:
            return None
        prompts = training.prompts_for(head["classes"],
                                       template or "a photo of {}")
        return training.compare(head, self.engine.encode_texts(prompts).cpu())

    def head(self, name) -> Head:
        valid_name(name)          # a malformed name is not a missing head
        return Head(name, training.public(name, self.heads[name]))

    def list_heads(self) -> list[dict]:
        self.heads.sync()
        return [training.public(n, h) for n, h in sorted(self.heads.items())]

    def delete_head(self, name) -> bool:
        return self.heads.delete(name)

    def __repr__(self):
        return (f"<Jev encoder={self.encoder.name!r} dim={self.encoder.dim} "
                f"heads={len(self.heads)}>")


def load(encoder: str | Encoder | None = None, *, heads_dir=None, temperature=None,
         batch=False) -> Jev:
    """Load an encoder and return a Jev you can ask questions of.

        jevimage.load()                              # siglip2-base-224, the default
        jevimage.load("siglip2-so400m-384")          # a bigger one
        jevimage.load("siglip2-base-224+dfn5b-h-14-384")   # an ensemble
        jevimage.load(MyEncoder())                   # your own

    `jevimage.available()` lists the registry. Weights download on first use and are
    cached by huggingface, so the first call is slow and the rest are not.
    """
    return Jev(encoder, heads_dir=heads_dir, temperature=temperature, batch=batch)
