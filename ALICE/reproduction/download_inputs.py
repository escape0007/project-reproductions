#!/usr/bin/env python3
"""Download the four public assets linked by the upstream ALICE README."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ASSETS = [
    ("1ETJlSTsljAezfSEQyls_C7qo1ZyFmZEG", "RoBERTa_SeqGAN_RankFiltPro_FE_1000.csv", 20916,
     "89964275210ebbbc91c60fc22b5344662d1f4341a4f5db5d9f41b4c4d8aa3fdd"),
    ("13sQxZ2PuedCfBFiU8545r3x2B2WE4VFH", "Prod_Fit_model.sav", 49061170,
     "138418abda92c600b27530c8189acf53d0922e2269875b9ce5b073a024edf71a"),
    ("1yEwHrI2yoDWOEY3C90tuzhVY-F10fXnw", "LY6C1_model.sav", 38981932,
     "29621aa3454f7b1d8a48f916b64dafed439aaba20ae9bdde20299e63ed045564"),
    ("15Lmjcnzsvu2eamtezXEjB5-NDNgFshLa", "LY6A_model.sav", 38891561,
     "aae84ccad7543493354a094dabfd137aff055e80bb72cac6dde51cabcbb235b8"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = BASE / "external/alice_original"
    directory.mkdir(parents=True, exist_ok=True)
    receipts = []
    for file_id, name, size, expected in ASSETS:
        path = directory / name
        if not path.exists():
            if args.verify_only:
                raise FileNotFoundError(path)
            import gdown
            temporary = path.with_suffix(path.suffix + ".part")
            if gdown.download(id=file_id, output=str(temporary), quiet=False) is None:
                raise RuntimeError(f"Download unavailable: {name}")
            if temporary.stat().st_size != size or hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Downloaded file differs from recorded upstream asset: {name}")
            temporary.rename(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != size or actual != expected:
            raise ValueError(f"Asset verification failed: {name}; existing file was left unchanged")
        receipts.append({"id": file_id, "name": name, "size": size, "sha256": actual})
        print(f"[VERIFIED] {name}")
    if not args.verify_only:
        (directory / "download_manifest.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
