#!/usr/bin/env python3
"""Generate a synthetic EEG dataset.

Usage:
    python scripts/generate_dataset.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EEG dataset.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    from synthgen.config import GenerationConfig
    from synthgen.pipeline_runner import PipelineRunner

    config = GenerationConfig.from_yaml(args.config)
    print(f"Generating {config.n_samples} samples -> {config.writer.output_dir}")
    PipelineRunner(config).run()


if __name__ == "__main__":
    main()
