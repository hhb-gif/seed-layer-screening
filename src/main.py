"""CLI entry point for seed layer screening pipeline."""

import argparse
import sys
from pathlib import Path

from seed_layer.config import load_config
from seed_layer.pipeline import SeedLayerPipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed layer material screening for lithium metal batteries"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for output directory naming",
    )
    parser.add_argument(
        "--materials",
        type=str,
        default=None,
        help="Path to materials list file",
    )
    parser.add_argument(
        "--skip-neb",
        action="store_true",
        help="Skip NEB diffusion calculation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted run (not yet implemented)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode (offline, no API needed)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Setup output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Run pipeline
    if args.demo:
        print("Demo mode not yet implemented")
        sys.exit(0)

    pipeline = SeedLayerPipeline(config, output_dir, tag=args.tag)
    pipeline.run(materials_file=args.materials, skip_neb=args.skip_neb)


if __name__ == "__main__":
    main()
