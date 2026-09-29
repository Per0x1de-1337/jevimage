#!/usr/bin/env python3
"""Build the Mintlify site from the GitHub-readable docs/ tree.

The two targets disagree about links and cannot be satisfied by one spelling:

  GitHub renders docs/*.md in place, so a link has to be `install.md` - a relative
  path to a file that exists beside it.
  Mintlify routes pages by URL and rejects both `install.md` and `install`; it wants
  the site-rooted `/docs/install`.

Rather than pick one and break the other, docs/ stays in the form GitHub needs - it is
the copy people read in the repository - and this rewrites it on the way to the site.
One source, transformed for the target, instead of two copies that drift.

    python3 tools/build_docs.py          # writes export.zip
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def rewrite(text: str) -> str:
    """install.md -> /docs/install, keeping any anchor."""
    return re.sub(r'\]\(([a-z-]+)\.md(#[^)]*)?\)',
                  lambda m: f"](/docs/{m.group(1)}{m.group(2) or ''})", text)


def main() -> int:
    if shutil.which("mint") is None:
        print("mint is not installed: npm install -g mint", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        stage = pathlib.Path(tmp) / "site"
        stage.mkdir()
        shutil.copy(ROOT / "docs.json", stage)
        (stage / "docs").mkdir()
        n = 0
        for page in sorted((ROOT / "docs").glob("*.md")):
            out = rewrite(page.read_text())
            n += len(re.findall(r'\]\(/docs/', out))
            (stage / "docs" / page.name).write_text(out)
        print(f"staged {len(list((stage / 'docs').glob('*.md')))} pages, "
              f"{n} links rewritten")

        r = subprocess.run(["mint", "broken-links"], cwd=stage)
        if r.returncode:
            print("broken links: not exporting", file=sys.stderr)
            return r.returncode
        r = subprocess.run(["mint", "export"], cwd=stage)
        if r.returncode:
            return r.returncode

        shutil.copy(stage / "export.zip", ROOT / "export.zip")
        print(f"wrote {ROOT / 'export.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
