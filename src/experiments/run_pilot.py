from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.experiments.pilot.run_pilot import run_pilot as run_registry_pilot
from src.utils.io import load_jsonl, save_jsonl


def _ratio_key(path: Path) -> int:
    """Sort helper for files like ..._r75.jsonl."""

    name = path.stem
    if "_r" not in name:
        return 10**9

    suffix = name.rsplit("_r", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 10**9


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the GSM8K pilot (Generator→Verifier) and save a consolidated JSONL.",
    )

    p.add_argument("--pilot-config", default="pilot_default")
    p.add_argument("--gen-config", default="gen_medium")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-every", type=int, default=25)

    p.add_argument("--output-dir", default="data/experiments/pilot")
    p.add_argument(
        "--output",
        default="data/experiments/pilot/pilot_gsm8k.jsonl",
        help="Path to write the consolidated results JSONL.",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    run_registry_pilot(
        datasets=["gsm8k"],
        pilot_config_name=args.pilot_config,
        gen_config_name=args.gen_config,
        model=args.model,
        temperature=args.temperature,
        limit=args.limit,
        output_dir=args.output_dir,
        save_every=args.save_every,
    )

    output_dir = Path(args.output_dir)
    pattern = f"gsm8k_{args.pilot_config}_{args.gen_config}_r*.jsonl"
    parts = sorted(output_dir.glob(pattern), key=_ratio_key)

    consolidated: list[dict[str, Any]] = []
    for p in parts:
        consolidated.extend(load_jsonl(str(p)))

    save_jsonl(consolidated, args.output)
    print(f"Saved consolidated GSM8K pilot → {args.output} ({len(consolidated)} rows)")


if __name__ == "__main__":
    main()
