"""Write the fixture images the docs pages read, so every block on them runs as printed.

    python examples/make_fixtures.py [dir]      # default: the current directory

The stand-in encoders in the docs (toy.py, tint.py) see raw pixels and nothing else, so
the printed probabilities are a function of these exact images. Regenerating them here
is what makes those transcripts reproducible; swap in your own images and the shapes
stay, the numbers move.

    ds/       12 flat 64x64 squares, red/ and blue/, 6 each  - quickstart, questions,
              cookbook, reference-python, troubleshooting
    shots/    36 flat 64x64 squares in daylight/ flash/ nightshot/  - training
    colours/  26 flat 64x64 squares, red/ 12, blue/ 12, green/ 2    - training
"""
import os
import sys

from PIL import Image

# Per label: a base colour, the axis each image's offset moves along, and one offset per
# image. Flat squares, but not identical ones, so a fold split has something to be wrong
# about - and shots/ is deliberately tight enough that one daylight image lands on the
# wrong side of the boundary, which is what the held-out number on training.md shows.
SETS = {
    "ds": ("{i}.png", {
        "red":  ((200, 40, 40), (1, 1, 1), (4, 6, -18, -4, 12, 11)),
        "blue": ((40, 40, 200), (1, 1, 1), (5, -1, 10, 2, 17, -7))}),
    "shots": ("{i:02d}.png", {
        "daylight":  ((210, 205, 180), (1, 1, 0), range(-22, 23, 4)),
        "flash":     ((245, 245, 240), (1, 1, 1), range(-22, 23, 4)),
        "nightshot": ((30, 35, 70),    (0, 1, 2), range(-22, 23, 4))}),
    "colours": ("{i:02d}.png", {
        "red":   ((190, 45, 45), (1, 1, 1), range(-24, 36, 5)),
        "blue":  ((45, 45, 190), (1, 1, 1), range(-24, 36, 5)),
        # the two greens training.md describes: one yellow-green, one cyan-green
        "green": ((95, 190, 95), (1, 0, -1), (55, -55))}),
}


def write(root="."):
    counts = {}
    for name, (fmt, labels) in SETS.items():
        counts[name] = 0
        for label, (base, axis, offsets) in labels.items():
            os.makedirs(f"{root}/{name}/{label}", exist_ok=True)
            for i, d in enumerate(offsets):
                rgb = tuple(min(255, max(0, c + d * a)) for c, a in zip(base, axis))
                Image.new("RGB", (64, 64), rgb).save(
                    f"{root}/{name}/{label}/" + fmt.format(i=i))
                counts[name] += 1
    return counts


if __name__ == "__main__":
    counts = write(sys.argv[1] if len(sys.argv) > 1 else ".")
    assert counts == {"ds": 12, "shots": 36, "colours": 26}, counts
    print(counts)
