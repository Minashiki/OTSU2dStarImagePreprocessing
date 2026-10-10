"""Regression checks for preset binning and clipping before 2-D Otsu."""

import json
from pathlib import Path
import subprocess
import sys

import cv2
from astropy.io import fits
import numpy as np
import pytest

import P1Preprocessing02 as pipeline
from SpaceStarOtsu import (
    PRESET_PERCENTILES,
    SpaceOtsuConfig,
    bin_threshold_to_gray,
    build_binned_2d_histogram,
    compare_searches,
    compare_searches_blocked,
)


def test_default_histogram_retains_existing_bins_and_exclusions():
    gray = np.array([[0, 1, 2, 8], [-1, 9, np.nan, 7]], dtype=np.float32)
    mean = np.array([[0, 2, 1, 8], [0, 0, 0, 9]], dtype=np.float32)
    histogram, valid, excluded = build_binned_2d_histogram(
        gray, mean, upper=8, bins=8, chunk_rows=1
    )
    expected = np.zeros((8, 8))
    expected[0, 0] = expected[1, 2] = expected[2, 1] = expected[7, 7] = 0.25
    np.testing.assert_array_equal(histogram, expected)
    assert (valid, excluded) == (4, 4)
    assert SpaceOtsuConfig().gray_scaling == "preset"
    assert (SpaceOtsuConfig().gray_lower, SpaceOtsuConfig().gray_upper) == (0, 255)
    assert bin_threshold_to_gray(0, 8, 8) == 1
    assert bin_threshold_to_gray(6, 8, 8) == 7


@pytest.mark.parametrize("preset", PRESET_PERCENTILES)
def test_default_search_thresholds_match_existing_result(preset):
    # Existing exact and nine-grid searches split these clusters at bin (1, 1).
    histogram = np.zeros((8, 8))
    histogram[1, 1] = histogram[6, 6] = 0.5
    result = compare_searches(
        preset, PRESET_PERCENTILES[preset], 8, histogram, 2, 0,
        SpaceOtsuConfig(hist_bins=8),
    )
    assert (result.exact.s_bin, result.exact.t_bin) == (1, 1)
    assert (result.exact_threshold_s, result.exact_threshold_t) == (2, 2)
    assert (result.nine_threshold_s, result.nine_threshold_t) == (2, 2)
    assert result.exact.score == 12.5


def test_histogram_and_thresholds_use_negative_lower_bound():
    gray = np.array([[-4, -3, 0, 3], [4, -5, 5, np.nan]], dtype=np.float32)
    mean = np.array([[-4, 0, 3, 4], [-3, 0, 0, 0]], dtype=np.float32)
    histogram, valid, excluded = build_binned_2d_histogram(
        gray, mean, 4, bins=8, chunk_rows=1, lower=-4
    )
    expected = np.zeros((8, 8))
    for location in ((0, 0), (1, 4), (4, 7), (7, 7), (7, 1)):
        expected[location] = 0.2
    np.testing.assert_array_equal(histogram, expected)
    assert (valid, excluded) == (5, 3)
    assert bin_threshold_to_gray(0, 4, 8, lower=-4) == -3
    assert bin_threshold_to_gray(6, 4, 8, lower=-4) == 3
    clusters = np.zeros((8, 8))
    clusters[1, 1] = clusters[6, 6] = 0.5
    result = compare_searches(
        "balanced", 99.9, 4, clusters, 2, 0,
        SpaceOtsuConfig(hist_bins=8, gray_scaling="clip", gray_lower=-4, gray_upper=4),
    )
    assert (result.exact_threshold_s, result.exact_threshold_t) == (-2, -2)
    assert (result.nine_threshold_s, result.nine_threshold_t) == (-2, -2)


def test_clip_blocks_keep_fixed_range_and_incomplete_edges():
    image = np.where(np.indices((5, 7)).sum(axis=0) % 2, 4, -4).astype(np.float32)
    config = SpaceOtsuConfig(hist_bins=8, gray_scaling="clip", gray_lower=-4, gray_upper=4)
    results, exact, nine = compare_searches_blocked(image, image, (3, 4), "balanced", 99.9, config)
    assert [(block.height, block.width) for block in results] == [(3, 4), (3, 3), (2, 4), (2, 3)]
    assert sum(block.valid_pixels for block in results) == image.size
    assert all(block.upper == 4 and block.excluded_pixels == 0 for block in results)
    np.testing.assert_array_equal(exact, np.where(image == 4, 255, 0).astype(np.uint8))
    np.testing.assert_array_equal(nine, exact)


@pytest.mark.parametrize("lower,upper", [(0, 0), (10, -1), (np.nan, 255), (0, np.inf), (-np.inf, 255)])
def test_clip_rejects_invalid_bounds(lower, upper):
    with pytest.raises(ValueError):
        SpaceOtsuConfig(gray_scaling="clip", gray_lower=lower, gray_upper=upper)


def test_cli_parses_scaling_options():
    parser = pipeline._build_arg_parser()
    defaults = parser.parse_args([])
    assert (defaults.gray_scaling, defaults.gray_lower, defaults.gray_upper) == ("preset", 0, 255)
    custom = parser.parse_args(["--gray-scaling", "clip", "--gray-lower", "-40", "--gray-upper", "80"])
    assert (custom.gray_scaling, custom.gray_lower, custom.gray_upper) == ("clip", -40, 80)
    with pytest.raises(SystemExit):
        parser.parse_args(["--gray-scaling", "stretch"])
    with pytest.raises(ValueError):
        SpaceOtsuConfig(gray_scaling="stretch")


def _synthetic_frame(tmp_path):
    raw = np.full((9, 11), 40, dtype=np.float32)
    raw[:, :3] = -80
    raw[:, 7:] = 400
    source = tmp_path / "frame.fits"
    fits.writeto(source, raw)
    filtered = cv2.GaussianBlur(cv2.medianBlur(raw, 3), (5, 5), sigmaX=1, sigmaY=1)
    return source, raw, filtered


@pytest.mark.parametrize("batch,block_size,bounds", [
    (False, None, None), (False, None, (20, 80)),
    (False, None, (-40, 80)), (True, "5x6", (-40, 80)),
])
def test_cli_pipeline_clips_filtered_image_and_preserves_raw_outputs(tmp_path, batch, block_size, bounds):
    source, raw, filtered = _synthetic_frame(tmp_path)
    output = tmp_path / "result"
    command = [sys.executable, "-B", str(Path(pipeline.__file__)), "--input", str(source),
               "--output", str(output), "--signed", "--save-fits", "--gray-scaling", "clip"]
    if bounds is not None:
        command += ["--gray-lower", str(bounds[0]), "--gray-upper", str(bounds[1])]
    if batch:
        command += ["--batch"]
    if block_size:
        command += ["--block-size", block_size]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if batch:
        assert (output / "threshold_compare.csv").is_file()
        output = output / "note_fits01"
    lower, upper = bounds or (0, 255)
    expected = np.clip(filtered, lower, upper)
    np.testing.assert_array_equal(np.load(output / "img_gaussian.npy"), expected)
    np.testing.assert_array_equal(fits.getdata(output / "img_gaussian.fits"), expected)
    interior = (filtered > lower) & (filtered < upper)
    assert interior.any() and (filtered < lower).any() and (filtered > upper).any()
    np.testing.assert_array_equal(expected[interior], filtered[interior])
    np.testing.assert_array_equal(fits.getdata(output / "raw_img.fits"), raw)
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert (stats["config"]["gray_scaling"], stats["config"]["gray_lower"], stats["config"]["gray_upper"]) == ("clip", lower, upper)
    for result in stats["presets"].values():
        assert result["upper"] == upper
        assert (result["valid_pixels"], result["excluded_pixels"]) == (raw.size, 0)
    mask = fits.getdata(output / "mask.fits") > 0
    assert mask.any()
    np.testing.assert_array_equal(fits.getdata(output / "clean_star.fits"), np.where(mask, raw, 0))
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    fluxes = [float(raw[labels == label].astype(np.float64).sum()) for label in range(1, count)]
    with np.load(output / "Centroid_Top" / "centroid_all.npz") as centroids:
        np.testing.assert_array_equal(centroids["all_hash"]["flux"], sorted((flux for flux in fluxes if flux > 0), reverse=True))


def test_neighborhood_mean_receives_clipped_image_and_roundoff_is_clipped(tmp_path, monkeypatch):
    source, _, filtered = _synthetic_frame(tmp_path)
    expected = np.clip(filtered, -40, 80)
    original_mean = pipeline.neighborhood_mean
    original_histogram = pipeline.build_binned_2d_histogram

    def mean_spy(image, size):
        np.testing.assert_array_equal(image, expected)
        mean = original_mean(image, size)
        mean[0, 0] = np.nextafter(np.float32(-40), np.float32(-np.inf))
        mean[-1, -1] = np.nextafter(np.float32(80), np.float32(np.inf))
        return mean

    def histogram_spy(image, mean, *args, **kwargs):
        assert mean.min() >= -40 and mean.max() <= 80
        return original_histogram(image, mean, *args, **kwargs)

    monkeypatch.setattr(pipeline, "neighborhood_mean", mean_spy)
    monkeypatch.setattr(pipeline, "build_binned_2d_histogram", histogram_spy)
    pipeline.main_p1(source, tmp_path / "result", save_overlays=False,
                     config=SpaceOtsuConfig(hist_bins=8, gray_scaling="clip", gray_lower=-40, gray_upper=80))
