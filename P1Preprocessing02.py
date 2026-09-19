import argparse
import csv
from dataclasses import asdict, replace
import glob
import hashlib
import json
import os
import struct
import time
import warnings

import cv2
from astropy.io import fits
from PIL import Image
import numpy as np

from UtilityFunction.calculate_centroid import calculate_grayscale_centroid
from SpaceStarOtsu import (
    SpaceOtsuConfig,
    apply_float_threshold,
    build_binned_2d_histogram,
    choose_histogram_upper,
    compare_searches,
    connected_component_metrics,
    estimate_persistent_bad_pixels,
    neighborhood_mean,
    replace_bad_pixels,
    restore_unsigned_int16,
    robust_background,
    save_histogram_2d,
    save_histogram_3d_safe,
)

FITS_PATH = "rst19\\rst19\\20260330163205413_9901.fits"
save_dir = "rst19\\note_fits01"
TOP_NUM = 5000


def _fits_length_status(fits_path, data_offset, shape, bitpix):
    """Distinguish missing FITS block padding from an incomplete pixel payload."""
    payload_bytes = int(np.prod(shape, dtype=np.int64)) * (abs(int(bitpix)) // 8)
    required_length = int(data_offset) + payload_bytes
    padded_length = int(data_offset) + ((payload_bytes + 2879) // 2880) * 2880
    actual_length = os.path.getsize(fits_path)
    if actual_length < required_length:
        status = "pixel_data_truncated"
    elif actual_length < padded_length:
        status = "pixel_data_complete_padding_missing"
    else:
        status = "complete"
    return {
        "status": status,
        "actual_length": actual_length,
        "required_pixel_length": required_length,
        "expected_padded_length": padded_length,
        "missing_pixel_bytes": max(0, required_length - actual_length),
        "missing_padding_bytes": max(0, padded_length - actual_length),
    }


def load_space_fits(fits_path, config=None, hdu_index=0):
    """Load one FITS frame and restore this camera's unsigned 16-bit convention."""
    config = config or SpaceOtsuConfig()
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        with fits.open(fits_path, memmap=False) as hdul:
            hdu = hdul[hdu_index]
            if hdu.data is None:
                raise ValueError(f"HDU[{hdu_index}] 没有图像数据")
            header = hdu.header.copy()
            file_info = hdul.fileinfo(hdu_index) or {}
            signed = np.array(hdu.data, copy=True)

    if signed.ndim != 2:
        raise ValueError(f"只支持二维 FITS 图像，实际 shape={signed.shape}")
    data_offset = file_info.get("datLoc")
    if data_offset is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_offset = len(header.tostring())
    length_status = _fits_length_status(
        fits_path,
        data_offset,
        signed.shape,
        header.get("BITPIX", signed.dtype.itemsize * 8),
    )
    if length_status["status"] == "pixel_data_truncated":
        raise OSError(
            f"FITS 像素数据不完整，缺少 {length_status['missing_pixel_bytes']} 字节: {fits_path}"
        )

    restored, was_reinterpreted = restore_unsigned_int16(
        signed,
        header.get("BZERO", 0),
        enabled=config.reinterpret_unsigned,
    )
    restored_f32 = np.asarray(restored, dtype=np.float32)
    finite = restored_f32[np.isfinite(restored_f32)]
    percentiles = np.percentile(finite, [0.1, 1, 50, 99, 99.9])
    stats = {
        "shape": list(restored_f32.shape),
        "source_dtype": str(signed.dtype),
        "working_dtype": str(restored_f32.dtype),
        "unsigned_reinterpreted": bool(was_reinterpreted),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p0.1": float(percentiles[0]),
        "p1": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "p99.9": float(percentiles[4]),
        "signed_negative_pixels": int(np.count_nonzero(signed < 0)),
        "fits_length": length_status,
        "warnings": [str(item.message) for item in warning_list],
    }
    return signed, restored_f32, header, stats

def save_float_as_png_uint16(img_f32, save_path, lo_q=0.1, hi_q=99.9):
    finite = img_f32[np.isfinite(img_f32)]
    lo, hi = np.percentile(finite, [lo_q, hi_q])
    x = np.clip(img_f32, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    x16 = (x * 65535.0).astype(np.uint16)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(x16).save(save_path)

def save_centroids_to_csv(centroids, output_dir):
    """Save ``(x, y)`` centroid pairs as NumPy and CSV files."""
    save_path_npy = os.path.join(output_dir, "centroids.npy")
    save_path_csv = os.path.join(output_dir, "centroids.csv")

    os.makedirs(output_dir, exist_ok=True)
    np.save(save_path_npy, centroids)
    with open(save_path_csv, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["centroid_x", "centroid_y"])
        writer.writerows(centroids)

    print(f"质心数据已保存到：{save_path_npy} 和 {save_path_csv}")

def centroid_hash_u64(x_f4: np.float32, y_f4: np.float32) -> np.uint64:
    b = struct.pack("<ff", float(np.float32(x_f4)), float(np.float32(y_f4)))  # 固定 float32 序列化
    d = hashlib.blake2b(b, digest_size=8).digest()  # 8 bytes = 64-bit
    return np.frombuffer(d, dtype=np.uint64)[0]

def _save_centroid_artifacts(grayscale_centroids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    centroids = [(star["centroid_x"], star["centroid_y"]) for star in grayscale_centroids]
    save_centroids_to_csv(centroids, output_dir)

    all_count = len(grayscale_centroids)
    all_hash = np.empty(
        (all_count,),
        dtype=[("hash", "u8"), ("x", "f4"), ("y", "f4"), ("flux", "f8"), ("area", "i4")],
    )
    all_x = np.array([star["centroid_x"] for star in grayscale_centroids], dtype=np.float32)
    all_y = np.array([star["centroid_y"] for star in grayscale_centroids], dtype=np.float32)
    all_flux = np.array([star["flux"] for star in grayscale_centroids], dtype=np.float64)
    all_area = np.array([star["area"] for star in grayscale_centroids], dtype=np.int32)
    all_hash["x"], all_hash["y"] = all_x, all_y
    all_hash["flux"], all_hash["area"] = all_flux, all_area
    all_hash["hash"] = np.array(
        [centroid_hash_u64(x, y) for x, y in zip(all_x, all_y)], dtype=np.uint64
    )
    np.savez_compressed(os.path.join(output_dir, "centroid_all.npz"), all_hash=all_hash)

    with open(os.path.join(output_dir, "centroid_all.csv"), "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "hash", "x", "y", "flux", "area"])
        for index in range(all_count):
            writer.writerow(
                [
                    index + 1,
                    int(all_hash["hash"][index]),
                    float(all_hash["x"][index]),
                    float(all_hash["y"][index]),
                    float(all_hash["flux"][index]),
                    int(all_hash["area"][index]),
                ]
            )

    top_count = min(TOP_NUM, all_count)
    top_hash = np.empty((top_count,), dtype=[("hash", "u8"), ("x", "f4"), ("y", "f4")])
    top_hash["x"], top_hash["y"] = all_x[:top_count], all_y[:top_count]
    top_hash["hash"] = all_hash["hash"][:top_count]
    np.savez_compressed(os.path.join(output_dir, "centroid_top.npz"), top_hash=top_hash)


def _save_mask_overlay(image, mask_u8, save_path):
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, [1.0, 99.9])
    display = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    display_u8 = (display * 255).astype(np.uint8)
    overlay = cv2.cvtColor(display_u8, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
    cv2.imwrite(save_path, overlay)


def _remove_stale_search_artifacts(output_dir, preset_name, selected_search):
    """Remove method-specific images left by an earlier run with another search."""
    other_search = "nine_grid" if selected_search == "exact" else "exact"
    stale_names = (
        f"mask_{preset_name}_{other_search}.png",
        f"overlay_{preset_name}_{other_search}.png",
        f"mask_{preset_name}_difference.png",
    )
    for name in stale_names:
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def _preset_to_dict(result, component_stats):
    return {
        "percentile": result.percentile,
        "upper": result.upper,
        "valid_pixels": result.valid_pixels,
        "excluded_pixels": result.excluded_pixels,
        "exact": asdict(result.exact),
        "nine_grid": asdict(result.nine_grid),
        "score_gap": result.score_gap,
        "bin_distance": result.bin_distance,
        "nine_grid_unstable": result.nine_grid_unstable,
        "exact_threshold_s": result.exact_threshold_s,
        "exact_threshold_t": result.exact_threshold_t,
        "nine_threshold_s": result.nine_threshold_s,
        "nine_threshold_t": result.nine_threshold_t,
        "components": component_stats,
    }


def _normalize_search_method(search_method):
    normalized = str(search_method).strip().lower().replace("-", "_")
    if normalized not in {"exact", "nine_grid"}:
        raise ValueError("search_method 必须是 'exact' 或 'nine-grid'")
    return normalized


def _validate_preset(preset, config):
    normalized = str(preset).strip().lower()
    if normalized not in config.preset_percentiles:
        choices = ", ".join(config.preset_percentiles)
        raise ValueError(f"preset 必须是以下之一: {choices}")
    return normalized


def _process_one(
    fits_path,
    output_dir,
    config,
    bad_pixel_mask=None,
    interactive=False,
    emit_3d=False,
    save_overlays=False,
    preset="balanced",
    search_method="exact",
    all_presets=False,
):
    started = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    centroid_dir = os.path.join(output_dir, "Centroid_Top")

    signed, restored, _header, fits_stats = load_space_fits(fits_path, config=config)
    raw_img = replace_bad_pixels(restored, bad_pixel_mask)
    del signed, restored
    print(f"[{os.path.basename(fits_path)}] shape={raw_img.shape}, min={raw_img.min():.1f}, max={raw_img.max():.1f}")
    if fits_stats["fits_length"]["status"] == "pixel_data_complete_padding_missing":
        print(
            "注意：FITS 像素数据完整，仅缺少尾部 padding "
            f"{fits_stats['fits_length']['missing_padding_bytes']} 字节。"
        )

    img_median = cv2.medianBlur(raw_img, ksize=3)
    img_gaussian = cv2.GaussianBlur(img_median, ksize=(5, 5), sigmaX=1.0, sigmaY=1.0)
    del img_median
    mean_img = neighborhood_mean(img_gaussian, size=config.neighborhood_size)
    background, sigma = robust_background(img_gaussian)

    np.save(os.path.join(output_dir, "img_gaussian.npy"), img_gaussian)
    np.save(os.path.join(output_dir, "I_D_raw.npy"), img_gaussian)
    save_float_as_png_uint16(img_gaussian, os.path.join(output_dir, "img_gaussian.png"))
    save_float_as_png_uint16(raw_img, os.path.join(output_dir, "raw_img.png"))

    selected_preset = _validate_preset(preset, config)
    selected_search = _normalize_search_method(search_method)
    if all_presets:
        preset_items = config.preset_percentiles.items()
    else:
        preset_items = ((selected_preset, config.preset_percentiles[selected_preset]),)

    preset_stats = {}
    production_mask = None
    selected_threshold_s = None
    selected_threshold_t = None
    for preset_name, percentile in preset_items:
        upper = choose_histogram_upper(
            img_gaussian,
            percentile,
            background,
            sigma,
            min_upper_sigma=config.min_upper_sigma,
        )
        histogram, valid_pixels, excluded_pixels = build_binned_2d_histogram(
            img_gaussian,
            mean_img,
            upper,
            bins=config.hist_bins,
            chunk_rows=config.hist_chunk_rows,
        )
        result = compare_searches(
            preset_name,
            percentile,
            upper,
            histogram,
            valid_pixels,
            excluded_pixels,
            config,
        )
        exact_mask_u8 = apply_float_threshold(
            img_gaussian,
            mean_img,
            result.exact_threshold_s,
            result.exact_threshold_t,
        )
        nine_mask_u8 = apply_float_threshold(
            img_gaussian,
            mean_img,
            result.nine_threshold_s,
            result.nine_threshold_t,
        )
        difference_u8 = np.where(exact_mask_u8 != nine_mask_u8, 255, 0).astype(np.uint8)
        component_stats = {
            "exact": connected_component_metrics(exact_mask_u8),
            "nine_grid": connected_component_metrics(nine_mask_u8),
            "different_pixels": int(np.count_nonzero(difference_u8)),
            "different_ratio": float(np.count_nonzero(difference_u8) / difference_u8.size),
        }
        if selected_search == "exact":
            chosen_mask = exact_mask_u8
            chosen_s = result.exact_threshold_s
            chosen_t = result.exact_threshold_t
        else:
            chosen_mask = nine_mask_u8
            chosen_s = result.nine_threshold_s
            chosen_t = result.nine_threshold_t

        preset_stats[preset_name] = _preset_to_dict(result, component_stats)
        preset_stats[preset_name]["selected_search"] = selected_search
        preset_stats[preset_name]["selected_threshold_s"] = chosen_s
        preset_stats[preset_name]["selected_threshold_t"] = chosen_t
        preset_stats[preset_name]["selected_components"] = component_stats[selected_search]
        _remove_stale_search_artifacts(
            output_dir, preset_name, selected_search
        )
        # Only the explicitly selected method is a production image artifact.
        Image.fromarray(chosen_mask).save(
            os.path.join(output_dir, f"mask_{preset_name}_{selected_search}.png")
        )
        # Compatibility alias follows the explicitly selected search method.
        Image.fromarray(chosen_mask).save(os.path.join(output_dir, f"mask_{preset_name}.png"))
        if preset_name == selected_preset:
            production_mask = chosen_mask.copy()
            selected_threshold_s = chosen_s
            selected_threshold_t = chosen_t
            Image.fromarray(chosen_mask).save(os.path.join(output_dir, "mask.png"))
        save_histogram_2d(
            histogram,
            result.exact,
            result.nine_grid,
            os.path.join(output_dir, f"hist2d_{preset_name}.png"),
            title=f"2-D Otsu histogram - {preset_name}",
        )
        if emit_3d and preset_name == selected_preset:
            save_histogram_3d_safe(
                histogram, os.path.join(output_dir, f"hist3d_{preset_name}.png")
            )
        if save_overlays:
            _save_mask_overlay(
                img_gaussian,
                chosen_mask,
                os.path.join(
                    output_dir,
                    f"overlay_{preset_name}_{selected_search}.png",
                ),
            )
        print(
            f"  {preset_name}: upper={upper:.3f}, "
            f"exact=({result.exact_threshold_s:.3f}, {result.exact_threshold_t:.3f}), "
            f"nine-grid=({result.nine_threshold_s:.3f}, {result.nine_threshold_t:.3f}), "
            f"selected={selected_search}, "
            f"unstable={result.nine_grid_unstable}"
        )
        if (
            preset_name == selected_preset
            and selected_search == "nine_grid"
            and result.nine_grid_unstable
        ):
            print(
                "  警告：所选九宫格结果与精确搜索差异超过稳定性标准；"
                "仍按用户选择生成 mask.png 和质心表。"
            )

    if production_mask is None:
        raise RuntimeError(f"未生成所选档位掩膜: {selected_preset}")
    grayscale_centroids = calculate_grayscale_centroid(production_mask > 0, raw_img)
    _save_centroid_artifacts(grayscale_centroids, centroid_dir)

    stats = {
        "source": os.path.abspath(fits_path),
        "output_dir": os.path.abspath(output_dir),
        "fits": fits_stats,
        "bad_pixels_applied": int(np.count_nonzero(bad_pixel_mask)) if bad_pixel_mask is not None else 0,
        "filtered": {
            "min": float(img_gaussian.min()),
            "max": float(img_gaussian.max()),
            "background_median": background,
            "background_sigma_mad": sigma,
        },
        "config": asdict(config),
        "selected_preset": selected_preset,
        "selected_search": selected_search,
        "selected_threshold_s": selected_threshold_s,
        "selected_threshold_t": selected_threshold_t,
        "presets": preset_stats,
        "centroid_count": len(grayscale_centroids),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as stream:
        json.dump(stats, stream, ensure_ascii=False, indent=2)

    if interactive:
        cv2.imshow(f"2D Otsu {selected_preset} / {selected_search}", production_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return grayscale_centroids, stats


def main_p1(
    fits_path=None,
    output_dir=None,
    config=None,
    bad_pixel_mask=None,
    interactive=False,
    emit_3d=False,
    save_overlays=True,
    preset="balanced",
    search="exact",
    all_presets=False,
):
    """Process one frame; ``preset`` and ``search`` drive mask.png and centroids."""
    fits_path = fits_path or FITS_PATH
    output_dir = output_dir or save_dir
    config = config or SpaceOtsuConfig()
    centroids, _ = _process_one(
        fits_path,
        output_dir,
        config,
        bad_pixel_mask=bad_pixel_mask,
        interactive=interactive,
        emit_3d=emit_3d,
        save_overlays=save_overlays,
        preset=preset,
        search_method=search,
        all_presets=all_presets,
    )
    return centroids


def _save_bad_pixel_outputs(output_root, bad_mask, occurrence, frame_count):
    np.save(os.path.join(output_root, "bad_pixel_mask.npy"), bad_mask)
    Image.fromarray(bad_mask.astype(np.uint8) * 255).save(
        os.path.join(output_root, "bad_pixel_mask.png")
    )
    coordinates = np.argwhere(occurrence > 0)
    with open(os.path.join(output_root, "bad_pixel_candidates.csv"), "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["y", "x", "occurrences", "persistence", "auto_repaired"])
        for y, x in coordinates:
            writer.writerow(
                [
                    int(y),
                    int(x),
                    int(occurrence[y, x]),
                    float(occurrence[y, x] / frame_count),
                    bool(bad_mask[y, x]),
                ]
            )


def _batch_summary_rows(frame_index, stats):
    rows = []
    for preset_name, preset in stats["presets"].items():
        rows.append(
            {
                "frame_index": frame_index,
                "source": stats["source"],
                "preset": preset_name,
                "is_selected_preset": preset_name == stats["selected_preset"],
                "selected_search": preset["selected_search"],
                "background_median": stats["filtered"]["background_median"],
                "background_sigma_mad": stats["filtered"]["background_sigma_mad"],
                "histogram_upper": preset["upper"],
                "exact_threshold_s": preset["exact_threshold_s"],
                "exact_threshold_t": preset["exact_threshold_t"],
                "nine_threshold_s": preset["nine_threshold_s"],
                "nine_threshold_t": preset["nine_threshold_t"],
                "selected_threshold_s": preset["selected_threshold_s"],
                "selected_threshold_t": preset["selected_threshold_t"],
                "exact_s_bin": preset["exact"]["s_bin"],
                "exact_t_bin": preset["exact"]["t_bin"],
                "exact_score": preset["exact"]["score"],
                "exact_seconds": preset["exact"]["seconds"],
                "nine_s_bin": preset["nine_grid"]["s_bin"],
                "nine_t_bin": preset["nine_grid"]["t_bin"],
                "nine_score": preset["nine_grid"]["score"],
                "nine_seconds": preset["nine_grid"]["seconds"],
                "score_gap": preset["score_gap"],
                "bin_distance": preset["bin_distance"],
                "nine_grid_unstable": preset["nine_grid_unstable"],
                "different_pixels": preset["components"]["different_pixels"],
                "different_ratio": preset["components"]["different_ratio"],
                **{
                    f"selected_{key}": value
                    for key, value in preset["selected_components"].items()
                },
                **{
                    f"exact_{key}": value
                    for key, value in preset["components"]["exact"].items()
                },
                **{
                    f"nine_{key}": value
                    for key, value in preset["components"]["nine_grid"].items()
                },
            }
        )
    return rows


def main_batch(
    input_pattern,
    output_root,
    config=None,
    emit_3d=False,
    preset="balanced",
    search="exact",
    all_presets=False,
):
    """Process a batch; ``preset`` and ``search`` drive every mask.png and centroid table."""
    config = config or SpaceOtsuConfig()
    paths = sorted(glob.glob(input_pattern))
    if not paths:
        raise FileNotFoundError(f"没有匹配到 FITS: {input_pattern}")
    os.makedirs(output_root, exist_ok=True)

    def frame_factory():
        for path in paths:
            signed, restored, _, _ = load_space_fits(path, config=config)
            yield signed, restored

    print(f"使用 {len(paths)} 帧估计保守坏点掩膜……")
    bad_mask, occurrence = estimate_persistent_bad_pixels(
        frame_factory, len(paths), config
    )
    _save_bad_pixel_outputs(output_root, bad_mask, occurrence, len(paths))
    print(f"自动修复坏点数: {int(bad_mask.sum())}")

    summary_rows = []
    all_centroids = []
    for index, path in enumerate(paths, start=1):
        frame_output = os.path.join(output_root, f"note_fits{index:02d}")
        centroids, stats = _process_one(
            path,
            frame_output,
            config,
            bad_pixel_mask=bad_mask,
            interactive=False,
            emit_3d=emit_3d,
            save_overlays=True,
            preset=preset,
            search_method=search,
            all_presets=all_presets,
        )
        all_centroids.append(centroids)
        summary_rows.extend(_batch_summary_rows(index, stats))

    summary_path = os.path.join(output_root, "threshold_compare.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return all_centroids


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="天基星图固定分箱二维 Otsu 预处理")
    parser.add_argument("--input", default=FITS_PATH, help="单个 FITS 路径或批处理通配符")
    parser.add_argument("--output", default=None, help="单帧输出目录或批处理输出根目录")
    parser.add_argument("--batch", action="store_true", help="按通配符执行批处理")
    parser.add_argument("--signed", action="store_true", help="禁用本设备的 int16→uint16 按位恢复")
    parser.add_argument("--interactive", action="store_true", help="单帧完成后显示所选掩膜")
    parser.add_argument("--emit-3d", action="store_true", help="保存最多 64×64 柱体的安全 3D 图")
    parser.add_argument(
        "--preset",
        choices=("balanced", "recall", "purity"),
        default="balanced",
        help="决定 mask.png 和质心表的灰度统计档位（默认 balanced）",
    )
    parser.add_argument(
        "--search",
        choices=("exact", "nine-grid"),
        default="exact",
        help="决定 mask.png 和质心表的阈值搜索方法（默认 exact）",
    )
    parser.add_argument(
        "--all-presets",
        action="store_true",
        help="额外生成三种档位；未指定时只处理 --preset 所选档位",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    runtime_config = replace(SpaceOtsuConfig(), reinterpret_unsigned=not args.signed)
    wildcard_input = any(char in args.input for char in "*?[")
    if args.batch or wildcard_input:
        main_batch(
            args.input,
            args.output or "rst19",
            config=runtime_config,
            emit_3d=args.emit_3d,
            preset=args.preset,
            search=args.search,
            all_presets=args.all_presets,
        )
    else:
        main_p1(
            args.input,
            args.output or save_dir,
            config=runtime_config,
            interactive=args.interactive,
            emit_3d=args.emit_3d,
            preset=args.preset,
            search=args.search,
            all_presets=args.all_presets,
        )

