#!/usr/bin/env python3
"""
Build or rebuild the ArcFace+YuNet feature cache from dataset.
Usage:
    python build_cache.py [--dataset DATASET_ROOT] [--output CACHE_FILE] [--force]

Examples:
    python build_cache.py                          # Use defaults from arcface_core
    python build_cache.py --force                  # Rebuild even if cache exists
    python build_cache.py --dataset datasource_v2 --output my_cache.npz

Features:
- Face alignment using eye landmarks
- ArcFace embeddings (512-dim, L2-normalized)
- Optional caching for faster reruns
"""

import argparse
from pathlib import Path
import sys

import arcface_core as core


def main():
    parser = argparse.ArgumentParser(
        description="Build or rebuild face embedding cache from dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=core.DATASET_ROOT,
        help=f"Dataset root directory (default: {core.DATASET_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=core.CACHE_FILE,
        help=f"Output cache file path (default: {core.CACHE_FILE})",
    )
    parser.add_argument(
        "--max-per-person",
        type=int,
        default=core.GALLERY_IMAGES_PER_PERSON,
        help=f"Max images per person (default: {core.GALLERY_IMAGES_PER_PERSON})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if cache exists",
    )

    args = parser.parse_args()

    dataset_root = args.dataset
    cache_file = args.output
    max_per_person = args.max_per_person

    # Remove existing cache if --force is used
    if args.force and cache_file.exists():
        print(f"Removing existing cache: {cache_file}")
        cache_file.unlink()

    # Build or load cache
    try:
        arcface_session, arcface_input_name, arcface_output_name = core.load_arcface_model()
        face_detector, _ = core.load_yunet_detector()

        features, image_paths, labels = core.build_or_load_gallery_features(
            dataset_root,
            cache_file,
            face_detector,
            arcface_session,
            arcface_input_name,
            arcface_output_name,
            max_images_per_person=max_per_person,
        )
        print(f"\n✓ Cache ready at: {cache_file}")
        print(f"  Features shape: {features.shape}")
        print(f"  Cache size: {cache_file.stat().st_size / (1024*1024):.2f} MB")
        return 0
    except Exception as e:
        print(f"✗ Error building cache: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
