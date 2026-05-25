from __future__ import annotations

import argparse
import time

from src.experiments.pilot.run_pilot import run_pilot
from src.registry.configs import CONFIGS, PILOT_CONFIGS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pilot experiments over a grid of pilot_config × gen_config (ratios come from pilot_config)."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "strategyqa", "truthfulqa"],
        help="Datasets to run (default: all pilot datasets)",
    )

    parser.add_argument(
        "--pilot-configs",
        nargs="+",
        default=["all"],
        help="Pilot config names to run, or 'all' (default: all)",
    )

    parser.add_argument(
        "--gen-configs",
        nargs="+",
        default=["all"],
        help="Gen config names to run, or 'all' (default: all)",
    )

    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="data/experiments/pilot")
    parser.add_argument("--save-every", type=int, default=25)

    return parser.parse_args()


def _expand(names: list[str], *, universe: list[str]) -> list[str]:
    lowered = [n.lower() for n in names]
    if "all" in lowered:
        return universe
    return names


if __name__ == "__main__":
    t0 = time.perf_counter()
    args = _parse_args()

    pilot_config_names = _expand(args.pilot_configs, universe=sorted(PILOT_CONFIGS.keys()))
    gen_config_names = _expand(args.gen_configs, universe=sorted(CONFIGS.keys()))

    for pilot_config_name in pilot_config_names:
        for gen_config_name in gen_config_names:
            run_pilot(
                datasets=args.datasets,
                pilot_config_name=pilot_config_name,
                gen_config_name=gen_config_name,
                model=args.model,
                temperature=args.temperature,
                limit=args.limit,
                output_dir=args.output_dir,
                save_every=args.save_every,
            )

    elapsed_s = time.perf_counter() - t0
    hh, rem = divmod(int(elapsed_s), 3600)
    mm, ss = divmod(rem, 60)
    print(f"Total runtime (grid): {hh:02d}:{mm:02d}:{ss:02d} ({elapsed_s:.2f}s)")
