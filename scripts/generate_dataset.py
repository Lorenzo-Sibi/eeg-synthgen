#!/usr/bin/env python3
"""Generate a synthetic EEG dataset.

Usage:
    python scripts/generate_dataset.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

MONTAGES_MAP = {
    "standard_1005_32": {"name": "standard_1005_32",  "n_channels": 32,  "split_role": "core"},
    "standard_1005_64": {"name": "standard_1005_64",  "n_channels": 64,  "split_role": "core"},
    "standard_1005_76": {"name": "standard_1005_76",  "n_channels": 76,  "split_role": "core"},
    "standard_1005_90": {"name": "standard_1005_90",  "n_channels": 90,  "split_role": "core"},
    "standard_1005_128": {"name": "standard_1005_128", "n_channels": 128, "split_role": "core"},
    "standard_1005_256": {"name": "standard_1005_256", "n_channels": 256, "split_role": "core"},
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EEG dataset.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Path to where save the dataset")
    parser.add_argument("--montage", type=str, default=None)
    args = parser.parse_args()

    from synthgen.config import GenerationConfig, MontageConfig, MontageEntry
    from synthgen.pipeline_runner import PipelineRunner

    config = GenerationConfig.from_yaml(args.config)
    if args.montage is not None:
        if args.montage not in MONTAGES_MAP:
            raise ValueError(f"Montage specified {args.montage} not valid.")
        config.montages = MontageConfig(montages=[MontageEntry(**MONTAGES_MAP[args.montage])])
    config.writer.output_dir = args.output_dir
    print(f"Generating {config.n_samples} samples -> {config.writer.output_dir}")
    PipelineRunner(config).run()


if __name__ == "__main__":
    main()
