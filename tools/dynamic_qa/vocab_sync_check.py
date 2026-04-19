from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _read_vocab(path: Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _dims(path: Path) -> Tuple[int, ...]:
    arr = np.load(str(path))
    return tuple(arr.shape)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check detection vocab + co-occur matrix consistency")
    ap.add_argument("--object", action="append", default=[], help="Object category to assert exists in vocab")
    ap.add_argument("--vocab", default="tools/object_vocabulary.txt")
    ap.add_argument("--obj_mtx", default="tools/obj.npy")
    ap.add_argument("--room_mtx", default="tools/room.npy")
    args = ap.parse_args()

    vocab_path = Path(args.vocab)
    vocab = _read_vocab(vocab_path)
    vocab_lower = {v.lower() for v in vocab}

    missing = [o for o in args.object if o.lower() not in vocab_lower]
    if missing:
        print(f"[Vocab] Missing in {vocab_path}: {missing}")
        print("Add them to tools/object_vocabulary.txt so GroundingDINO can detect CAD-inserted objects.")
    else:
        if args.object:
            print(f"[Vocab] OK: all requested objects present in {vocab_path}")

    obj_dims = _dims(Path(args.obj_mtx))
    room_dims = _dims(Path(args.room_mtx))
    print(f"[Matrices] obj.npy shape={obj_dims} | room.npy shape={room_dims}")
    print("Note: These matrices are used for navigation scoring (GLIP categories), not GroundingDINO vocab.")


if __name__ == "__main__":
    main()
