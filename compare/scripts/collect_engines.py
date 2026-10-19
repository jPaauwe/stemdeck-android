#!/usr/bin/env python3
"""Collects every distinct engine version from the git history for the
StemDeck Compare app.

For each `build-N` tag, the engine sources (engine.cpp, engine.hpp and the
demucs/ folder) are located: under engine/ since build 12, under
app/src/main/cpp/ before that. Builds whose engine is byte-for-byte the same
as an earlier build's are folded into that one. The working tree's engine is
added as "this build" when it differs from the newest tag.

Writes <out>/<id>/{engine.cpp, engine.hpp, demucs/...} per version, plus
<out>/engines.txt with one line per version: id|label|description.

usage: collect_engines.py <repo root> <out dir> [--max N]
"""
import argparse
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile

LOCATIONS = ["engine", "app/src/main/cpp"]


def git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True)
    if r.returncode:
        return None
    return r.stdout if binary else r.stdout.decode().strip()


def engine_location(repo, ref):
    for loc in LOCATIONS:
        if git(repo, "cat-file", "-e", f"{ref}:{loc}/engine.cpp") is not None:
            return loc
    return None


def tree_hash(repo, ref, loc):
    parts = [git(repo, "rev-parse", f"{ref}:{loc}/{p}") for p in ("demucs", "engine.cpp", "engine.hpp")]
    return None if None in parts else hashlib.sha1("".join(parts).encode()).hexdigest()


def worktree_hash(repo):
    h = hashlib.sha1()
    base = os.path.join(repo, "engine")
    files = ["engine.cpp", "engine.hpp"] + sorted(
        os.path.join("demucs", f) for f in os.listdir(os.path.join(base, "demucs")))
    for f in files:
        with open(os.path.join(base, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read())
    return h.hexdigest()


def blob_tree_digest(repo, ref, loc):
    """Same digest as worktree_hash, but over a committed tree."""
    h = hashlib.sha1()
    names = git(repo, "ls-tree", "--name-only", f"{ref}:{loc}/demucs")
    if names is None:
        return None
    files = ["engine.cpp", "engine.hpp"] + sorted(os.path.join("demucs", n) for n in names.splitlines())
    for f in files:
        data = git(repo, "show", f"{ref}:{loc}/{f}", binary=True)
        if data is None:
            return None
        h.update(f.encode() + b"\0" + data)
    return h.hexdigest()


def export(repo, ref, loc, dest):
    data = git(repo, "archive", "--format=tar", ref, f"{loc}/engine.cpp", f"{loc}/engine.hpp",
               f"{loc}/demucs", binary=True)
    if data is None:
        raise RuntimeError(f"git archive failed for {ref}")
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            rel = m.name[len(loc) + 1:]
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as fh:
                fh.write(tar.extractfile(m).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--max", type=int, default=8, help="newest N versions to keep")
    a = ap.parse_args()

    tags = git(a.repo, "tag", "-l", "build-*") or ""
    tags = sorted(tags.split(), key=lambda t: int(re.sub(r"\D", "", t) or 0))

    versions = []  # [id, label, first, last, digest, ref, loc]
    for t in tags:
        loc = engine_location(a.repo, t)
        if not loc:
            continue
        d = tree_hash(a.repo, t, loc)
        n = int(t.split("-")[1])
        if versions and versions[-1][4] == d:
            versions[-1][3] = n
            continue
        versions.append([f"b{n}", "", n, n, d, t, loc])

    # This build's engine, if it differs from the newest tagged one.
    wt = worktree_hash(a.repo)
    newest_same = bool(versions) and blob_tree_digest(a.repo, versions[-1][5], versions[-1][6]) == wt
    versions = versions[-a.max:]

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    lines = []
    for v in versions:
        vid, _, first, last, _, ref, loc = v
        export(a.repo, ref, loc, os.path.join(a.out, vid))
        label = f"build {first}" if first == last else f"builds {first}–{last}"
        subject = git(a.repo, "log", "-1", "--format=%s", ref) or ""
        lines.append(f"{vid}|{label}|{subject}")
    if not newest_same:
        dest = os.path.join(a.out, "head")
        os.makedirs(dest)
        src = os.path.join(a.repo, "engine")
        shutil.copy(os.path.join(src, "engine.cpp"), dest)
        shutil.copy(os.path.join(src, "engine.hpp"), dest)
        shutil.copytree(os.path.join(src, "demucs"), os.path.join(dest, "demucs"))
        subject = git(a.repo, "log", "-1", "--format=%s") or ""
        lines.append(f"head|this build|{subject} (plus any changes since)")

    with open(os.path.join(a.out, "engines.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
