"""Train a head on your own images and see, per class, whether it beat the prompts.

    python examples/train_your_own.py dataset/     # dataset/red/*.png, dataset/blue/*.png

One sub-folder per label, at least two images each. The encoder stays frozen and only a
linear readout is fitted, so this takes seconds and the head is tens of kilobytes. The
trained column is held out k-fold; zero-shot has no training to hold out, so it is
scored on every image. That is the only comparison worth printing.
"""
import sys

import jevimage

if len(sys.argv) != 2:
    sys.exit("usage: python examples/train_your_own.py <folder of label sub-folders>")

folder = sys.argv[1]
jev = jevimage.load()
head = jev.train("my_head", folder)
assert jev.head("my_head").classes == head.classes      # saved, not just in memory

print(f"{head.n} images, {len(head.classes)} classes, encoder {head.encoder}, "
      f"{head.to_dict()['bytes'] / 1024:.1f} KB of weights\n")
print(f"{'class':<14}{'n':>4}{'zero-shot':>12}{'trained':>10}{'delta':>9}")
for row in head.comparison["per_class"]:
    zs, tr = row["zero_shot"], row["trained"]
    print(f"{row['label']:<14}{row['n']:>4}{zs:>12.0%}{tr:>10.0%}{tr - zs:>+9.0%}")
overall = head.comparison["overall"]
print(f"{'ALL':<14}{head.n:>4}{overall['zero_shot']:>12.0%}{overall['trained']:>10.0%}"
      f"{overall['delta']:>+9.0%}")

verdict = "helped" if overall["delta"] > 0 else "did not help"
print(f"\ntraining {verdict}: {overall['delta']:+.0%} held out, against the prompts "
      "'a photo of <class>'")

image = jevimage.read_folder(folder)[head.classes[0]][0]
a = jev.ask(image, {"q": {"type": "head", "head": "my_head"}})["q"]
print(f"\n{image.name} (labelled {head.classes[0]}): "
      f"{a.choice} at p={a.probabilities[a.choice]:.0%}")
