"""jevimage end to end: encode one image, read four typed questions off it.

    python examples/quickstart.py photo.jpg

The first run downloads the default encoder, SigLIP2 base at 224px - about 1.5 GB
from huggingface, cached afterwards, so every later run starts in a second or two.
No GPU needed: it picks CPU when there is no CUDA, and JEV_DEVICE=cpu forces it.
"""
import sys

import jevimage

if len(sys.argv) != 2:
    sys.exit("usage: python examples/quickstart.py <image.jpg|png|webp>")

jev = jevimage.load()

questions = {
    "scene": {"type": "choice",
              "criteria": {"indoor": "an indoor scene", "outdoor": "the outdoors"}},
    "busy": {"type": "score",
             "criteria": ["an empty place", "a few people", "a packed crowd"]},
    "daylight": {"type": "noul",
                 "criteria": {"true": "a photo taken in daylight",
                              "false": "a photo taken at night"}},
}
# The fourth type reads a head you trained on your own images, so it only appears once
# you have one - examples/train_your_own.py makes one in about a second.
mine = [h["name"] for h in jev.list_heads() if h["encoder"] == jev.encoder.name]
if mine:
    questions["mine"] = {"type": "head", "head": mine[0]}

for qid, a in jev.ask(sys.argv[1], questions).items():
    if a.type == "noul":
        print(f"{qid:10} {a.noul:.0%} yes")
    elif a.type == "score":
        print(f"{qid:10} {a.score:.2f} of {len(a.legend) - 1}"
              f"   nearest: {a.legend[str(round(a.score))]}")
    else:
        print(f"{qid:10} {a.choice}   p={a.probabilities[a.choice]:.0%}"
              f"   confidence={a.confidence:.0%}")

if not mine:
    # naming the encoder matters: a head trained under a different JEV_ENCODER is on
    # disk but unusable here, and "no heads yet" would send you to retrain in a loop
    print(f"\nno head for encoder {jev.encoder.name!r}, so no 'head' question above - "
          "run examples/train_your_own.py to train one here")
