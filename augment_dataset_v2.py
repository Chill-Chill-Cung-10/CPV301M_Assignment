from pathlib import Path
import argparse

import cv2
import numpy as np


SRC_ROOT = Path("datasource")
DST_ROOT = Path("datasource_v2")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def augment(img):
    """
    Input: color image (H, W, 3)
    Output: list of augmented variants including original
    """
    results = [img]

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-20, -15, -10, -5, 5, 10, 15, 20]:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        results.append(rotated)

    for beta in [-30, 30]:
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        results.append(bright)

    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    results.append(noisy)

    return results


def read_image_bgr(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
    suffix = path.suffix.lower()
    ext = ".jpg" if suffix in {".jpeg", ".jpg"} else suffix
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def iter_images(root):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def ensure_bgr3(image):
    """Ensure image is 3-channel BGR."""
    if image is None:
        return None

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def build_augmented_dataset():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC_ROOT}")

    total_input = 0
    total_written = 0
    total_failed = 0

    for src_path in iter_images(SRC_ROOT):
        total_input += 1
        rel = src_path.relative_to(SRC_ROOT)
        dst_dir = DST_ROOT / rel.parent
        dst_dir.mkdir(parents=True, exist_ok=True)

        img = read_image_bgr(src_path)
        if img is None:
            total_failed += 1
            continue

        img = ensure_bgr3(img)
        variants = augment(img)

        stem = src_path.stem
        suffix = src_path.suffix.lower()
        ext = ".jpg" if suffix in {".jpeg", ".jpg"} else suffix

        for idx, aug_img in enumerate(variants):
            out_name = f"{stem}_aug{idx:02d}{ext}"
            out_path = dst_dir / out_name
            if write_image(out_path, aug_img):
                total_written += 1
            else:
                total_failed += 1

        if total_input % 200 == 0:
            print(
                f"Processed {total_input} input images | wrote {total_written} outputs | failed {total_failed}"
            )

    print("Done")
    print(f"Input images: {total_input}")
    print(f"Output images: {total_written}")
    print(f"Failures: {total_failed}")


def convert_existing_dataset_to_rgb(root):
    """Convert all images under a directory to 3-channel RGB-compatible BGR files."""
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    total = 0
    converted = 0
    failed = 0

    for img_path in iter_images(root):
        total += 1
        img = read_image_bgr(img_path)
        if img is None:
            failed += 1
            continue

        img_bgr = ensure_bgr3(img)
        if img_bgr is None:
            failed += 1
            continue

        if write_image(img_path, img_bgr):
            converted += 1
        else:
            failed += 1

        if total % 500 == 0:
            print(f"Converted {total} images | ok {converted} | failed {failed}")

    print("Convert done")
    print(f"Total images: {total}")
    print(f"Converted: {converted}")
    print(f"Failures: {failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--convert-existing",
        action="store_true",
        help="Convert existing datasource_v2 images in place to 3-channel RGB-compatible files.",
    )
    args = parser.parse_args()

    if args.convert_existing:
        convert_existing_dataset_to_rgb(DST_ROOT)
    else:
        build_augmented_dataset()


if __name__ == "__main__":
    main()

