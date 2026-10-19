#!/usr/bin/env python3
"""Lists the engines StemDeck Compare bundles.

    engines_for_compare.py <repo> <out dir>

Reads <repo>/engines/engines.txt and takes the engines marked "yes". If the
app's current engine (<repo>/engine) is not one of them, it is copied to
<out dir>/head and added as "this build", so an engine being worked on can
be compared with the kept ones.

Writes <out dir>/engines.txt, one line per engine: id|label|description|source dir
"""
import hashlib
import shutil
import sys
from pathlib import Path


def fingerprint(d: Path) -> str:
    h = hashlib.sha256()
    files = [d / "engine.cpp", d / "engine.hpp"] + sorted((d / "demucs").glob("*"))
    for f in files:
        if f.is_file() and f.name not in ("LICENSE", "UPSTREAM_COMMIT"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def main() -> int:
    repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lines, prints = [], set()
    for raw in (repo / "engines" / "engines.txt").read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        eid, label, desc, keep = [f.strip() for f in raw.split("|")]
        src = repo / "engines" / eid
        prints.add(fingerprint(src))
        if keep == "yes":
            lines.append(f"{eid}|{label}|{desc}|{src}")
    current = repo / "engine"
    if fingerprint(current) not in prints:
        head = out / "head"
        if head.exists():
            shutil.rmtree(head)
        head.mkdir()
        for name in ("engine.cpp", "engine.hpp"):
            shutil.copy2(current / name, head / name)
        shutil.copytree(current / "demucs", head / "demucs")
        lines.append(f"head|this build|The engine in this build, not yet in engines/|{head.resolve()}")
    (out / "engines.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
