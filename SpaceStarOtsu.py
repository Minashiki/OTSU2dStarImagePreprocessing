"""Memory-bounded 2-D Otsu helpers for the space-based FITS pipeline.

This module intentionally lives outside ``UtilityFunction``.  The legacy
ground-based implementation uses a dense raw-gray-level histogram and is kept
unchanged for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import time
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRESET_PERCENTILES: Mapping[str, float] = {
    "recall": 99.5,
    "balanced": 99.9,
    "purity": 99.95,
}


@dataclass(frozen=True)
class SpaceOtsuConfig:
    hist_bins: int = 256
    hist_chunk_rows: int = 256
    neighborhood_size: int = 7
    preset_percentiles: Mapping[str, float] = field(
        default_factory=lambda: dict(PRESET_PERCENTILES)
    )
    min_upper_sigma: float = 8.0
    reinterpret_unsigned: bool = True
    bad_pixel_persistence: float = 0.80
    bad_pixel_residual_sigma: float = 10.0
    bad_pixel_neighbor_sigma: float = 5.0
    nine_grid_samples_per_cell: int = 32
    nine_grid_stop_size: int = 8
    nine_grid_seed: int = 0
    unstable_score_gap: float = 0.01
    unstable_bin_distance: int = 4

    def __post_init__(self) -> None:
        if self.hist_bins < 8:
            raise ValueError("hist_bins must be at least 8")
        if self.hist_chunk_rows < 1:
            raise ValueError("hist_chunk_rows must be positive")
        if self.neighborhood_size < 1 or self.neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be a positive odd number")
        if not 0 < self.bad_pixel_persistence <= 1:
            raise ValueError("bad_pixel_persistence must be in (0, 1]")


@dataclass(frozen=True)
class SearchResult:
    s_bin: int
    t_bin: int
    score: float
    evaluations: int
    seconds: float


@dataclass(frozen=True)
class PresetResult:
    name: str
    percentile: float
    upper: float
    valid_pixels: int
    excluded_pixels: int
    exact: SearchResult
    nine_grid: SearchResult
    score_gap: float
    bin_distance: int
    nine_grid_unstable: bool
    exact_threshold_s: float
    exact_threshold_t: float
    nine_threshold_s: float
    nine_threshold_t: float


@dataclass(frozen=True)
class OtsuWorkspace:
    probability: np.ndarray
    prefix_p: np.ndarray
    prefix_i: np.ndarray
    prefix_j: np.ndarray
    total_i: float
    total_j: float

    @classmethod
    def from_histogram(cls, histogram: np.ndarray) -> "OtsuWorkspace":
        p = np.asarray(histogram, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] != p.shape[1]:
            raise ValueError("histogram must be a square 2-D array")
        total = float(p.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("histogram is empty")
        p = p / total
        levels = np.arange(p.shape[0], dtype=np.float64)
        weighted_i = p * levels[:, None]
        weighted_j = p * levels[None, :]
        prefix_p = p.cumsum(axis=0).cumsum(axis=1)
        prefix_i = weighted_i.cumsum(axis=0).cumsum(axis=1)
        prefix_j = weighted_j.cumsum(axis=0).cumsum(axis=1)
        return cls(
            probability=p,
            prefix_p=prefix_p,
            prefix_i=prefix_i,
            prefix_j=prefix_j,
            total_i=float(prefix_i[-1, -1]),
            total_j=float(prefix_j[-1, -1]),
        )

    @property
    def bins(self) -> int:
        return int(self.probability.shape[0])

    def score_points(
        self, s: np.ndarray | Sequence[int] | int, t: np.ndarray | Sequence[int] | int
    ) -> np.ndarray:
        s_arr = np.asarray(s, dtype=np.intp)
        t_arr = np.asarray(t, dtype=np.intp)
        s_arr, t_arr = np.broadcast_arrays(s_arr, t_arr)
        if np.any(s_arr < 0) or np.any(t_arr < 0):
            raise ValueError("threshold bins must be non-negative")
        if np.any(s_arr >= self.bins - 1) or np.any(t_arr >= self.bins - 1):
            raise ValueError("threshold bins must be smaller than bins - 1")

        p = self.prefix_p
        pi = self.prefix_i
        pj = self.prefix_j
        w0 = p[s_arr, t_arr]
        w1 = p[-1, -1] - p[s_arr, -1] - p[-1, t_arr] + p[s_arr, t_arr]
        i0_sum = pi[s_arr, t_arr]
        j0_sum = pj[s_arr, t_arr]
        i1_sum = pi[-1, -1] - pi[s_arr, -1] - pi[-1, t_arr] + pi[s_arr, t_arr]
        j1_sum = pj[-1, -1] - pj[s_arr, -1] - pj[-1, t_arr] + pj[s_arr, t_arr]

        valid = (w0 > 1e-12) & (w1 > 1e-12)
        u0_i = np.divide(i0_sum, w0, out=np.zeros_like(w0, dtype=np.float64), where=valid)
        u0_j = np.divide(j0_sum, w0, out=np.zeros_like(w0, dtype=np.float64), where=valid)
        u1_i = np.divide(i1_sum, w1, out=np.zeros_like(w1, dtype=np.float64), where=valid)
        u1_j = np.divide(j1_sum, w1, out=np.zeros_like(w1, dtype=np.float64), where=valid)
        score = w0 * ((u0_i - self.total_i) ** 2 + (u0_j - self.total_j) ** 2)
        score += w1 * ((u1_i - self.total_i) ** 2 + (u1_j - self.total_j) ** 2)
        return np.where(valid, score, -np.inf)


def restore_unsigned_int16(
    data: np.ndarray, bzero: int | float | None, enabled: bool = True
) -> tuple[np.ndarray, bool]:
    """Restore a non-standard unsigned sensor image stored as BITPIX=16/BZERO=0."""

    arr = np.asarray(data)
    should_restore = (
        enabled
        and arr.dtype.kind == "i"
        and arr.dtype.itemsize == 2
        and float(0 if bzero is None else bzero) == 0.0
    )
    if not should_restore:
        return arr, False
    byte_order = arr.dtype.byteorder
    unsigned_dtype = np.dtype(f"{byte_order}u2") if byte_order in ("<", ">") else np.dtype("u2")
    return arr.view(unsigned_dtype), True


def robust_background(image: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("image has no finite pixels")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = max(1.4826 * mad, np.finfo(np.float32).eps)
    return median, sigma


def neighborhood_mean(image: np.ndarray, size: int = 7) -> np.ndarray:
    return cv2.boxFilter(
        np.asarray(image, dtype=np.float32),
        ddepth=-1,
        ksize=(size, size),
        normalize=True,
        borderType=cv2.BORDER_DEFAULT,
    )


def replace_bad_pixels(image: np.ndarray, bad_mask: np.ndarray | None) -> np.ndarray:
    output = np.asarray(image, dtype=np.float32).copy()
    if bad_mask is None or not np.any(bad_mask):
        return output
    if bad_mask.shape != output.shape:
        raise ValueError("bad-pixel mask shape does not match image")
    local_median = cv2.medianBlur(output, 3)
    output[np.asarray(bad_mask, dtype=bool)] = local_median[np.asarray(bad_mask, dtype=bool)]
    return output


def estimate_persistent_bad_pixels(
    frame_factory: Callable[[], Iterable[tuple[np.ndarray, np.ndarray]]],
    frame_count: int,
    config: SpaceOtsuConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Find persistent isolated wrapped-hot pixels without retaining all frames."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    occurrence: np.ndarray | None = None
    expected_shape: tuple[int, ...] | None = None
    seen = 0
    for signed, restored in frame_factory():
        signed_arr = np.asarray(signed)
        restored_f32 = np.asarray(restored, dtype=np.float32)
        if expected_shape is None:
            expected_shape = restored_f32.shape
            occurrence = np.zeros(expected_shape, dtype=np.uint16)
        if restored_f32.shape != expected_shape or signed_arr.shape != expected_shape:
            raise ValueError("all calibration frames must have the same shape")
        background, sigma = robust_background(restored_f32)
        local_median = cv2.medianBlur(restored_f32, 3)
        residual = restored_f32 - local_median
        negative = signed_arr < 0
        negative_neighbors = cv2.boxFilter(
            negative.astype(np.uint8),
            ddepth=cv2.CV_16U,
            ksize=(3, 3),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        candidate = negative & (negative_neighbors <= 1)
        candidate &= residual > config.bad_pixel_residual_sigma * sigma
        candidate &= local_median <= background + config.bad_pixel_neighbor_sigma * sigma
        occurrence += candidate.astype(np.uint16)
        seen += 1
    if occurrence is None or seen != frame_count:
        raise ValueError(f"expected {frame_count} calibration frames, received {seen}")
    required = int(math.ceil(frame_count * config.bad_pixel_persistence))
    return occurrence >= required, occurrence


def choose_histogram_upper(
    image: np.ndarray,
    percentile: float,
    background: float,
    sigma: float,
    min_upper_sigma: float = 8.0,
) -> float:
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("image has no finite pixels")
    upper = max(float(np.percentile(finite, percentile)), background + min_upper_sigma * sigma)
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError("histogram upper bound must be positive")
    return upper


def build_binned_2d_histogram(
    image: np.ndarray,
    mean_image: np.ndarray,
    upper: float,
    bins: int = 256,
    chunk_rows: int = 256,
) -> tuple[np.ndarray, int, int]:
    """Build a fixed-size histogram while excluding the high-gray search tail."""

    gray = np.asarray(image, dtype=np.float32)
    mean = np.asarray(mean_image, dtype=np.float32)
    if gray.shape != mean.shape or gray.ndim != 2:
        raise ValueError("image and mean_image must be same-shaped 2-D arrays")
    if upper <= 0 or bins < 2:
        raise ValueError("upper and bins must be positive")
    counts = np.zeros(bins * bins, dtype=np.uint64)
    valid_total = 0
    scale = bins / float(upper)
    for row_start in range(0, gray.shape[0], chunk_rows):
        row_end = min(row_start + chunk_rows, gray.shape[0])
        g = gray[row_start:row_end]
        m = mean[row_start:row_end]
        valid = np.isfinite(g) & np.isfinite(m)
        valid &= (g >= 0.0) & (m >= 0.0) & (g <= upper) & (m <= upper)
        if not np.any(valid):
            continue
        g_bin = np.minimum((g[valid] * scale).astype(np.int32), bins - 1)
        m_bin = np.minimum((m[valid] * scale).astype(np.int32), bins - 1)
        flat = g_bin * bins + m_bin
        counts += np.bincount(flat, minlength=bins * bins).astype(np.uint64)
        valid_total += int(flat.size)
    if valid_total == 0:
        raise ValueError("no pixels fall inside the histogram search range")
    histogram = counts.reshape(bins, bins).astype(np.float64) / valid_total
    return histogram, valid_total, int(gray.size - valid_total)


def exact_threshold_search(workspace: OtsuWorkspace) -> SearchResult:
    start = time.perf_counter()
    levels = np.arange(workspace.bins - 1, dtype=np.intp)
    s_grid, t_grid = np.meshgrid(levels, levels, indexing="ij")
    score = workspace.score_points(s_grid, t_grid)
    flat_index = int(np.argmax(score))
    s_bin, t_bin = np.unravel_index(flat_index, score.shape)
    return SearchResult(
        s_bin=int(s_bin),
        t_bin=int(t_bin),
        score=float(score[s_bin, t_bin]),
        evaluations=int(score.size),
        seconds=float(time.perf_counter() - start),
    )


def nine_grid_threshold_search(
    workspace: OtsuWorkspace,
    samples_per_cell: int = 32,
    stop_size: int = 8,
    seed: int = 0,
) -> SearchResult:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    s_lo = t_lo = 0
    s_hi = t_hi = workspace.bins - 2
    best_s = best_t = 0
    best_score = -np.inf
    evaluations = 0

    while (s_hi - s_lo + 1) > stop_size or (t_hi - t_lo + 1) > stop_size:
        s_parts = [part for part in np.array_split(np.arange(s_lo, s_hi + 1), 3) if part.size]
        t_parts = [part for part in np.array_split(np.arange(t_lo, t_hi + 1), 3) if part.size]
        cell_results: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
        for s_part in s_parts:
            for t_part in t_parts:
                width = int(t_part.size)
                area = int(s_part.size * t_part.size)
                sample_count = min(samples_per_cell, area)
                offsets = rng.choice(area, size=sample_count, replace=False)
                s_values = s_part[offsets // width]
                t_values = t_part[offsets % width]
                scores = workspace.score_points(s_values, t_values)
                evaluations += sample_count
                local_index = int(np.argmax(scores))
                local_score = float(scores[local_index])
                cell_results.append(
                    (local_score, int(s_values[local_index]), int(t_values[local_index]), s_part, t_part)
                )
                if local_score > best_score:
                    best_score = local_score
                    best_s = int(s_values[local_index])
                    best_t = int(t_values[local_index])
        _, _, _, best_s_part, best_t_part = max(cell_results, key=lambda item: item[0])
        s_lo, s_hi = int(best_s_part[0]), int(best_s_part[-1])
        t_lo, t_hi = int(best_t_part[0]), int(best_t_part[-1])

    s_values, t_values = np.meshgrid(
        np.arange(s_lo, s_hi + 1, dtype=np.intp),
        np.arange(t_lo, t_hi + 1, dtype=np.intp),
        indexing="ij",
    )
    final_scores = workspace.score_points(s_values, t_values)
    evaluations += int(final_scores.size)
    final_index = int(np.argmax(final_scores))
    final_s = int(s_values.ravel()[final_index])
    final_t = int(t_values.ravel()[final_index])
    final_score = float(final_scores.ravel()[final_index])
    if final_score > best_score:
        best_s, best_t, best_score = final_s, final_t, final_score
    return SearchResult(
        s_bin=best_s,
        t_bin=best_t,
        score=best_score,
        evaluations=evaluations,
        seconds=float(time.perf_counter() - start),
    )


def bin_threshold_to_gray(bin_index: int, upper: float, bins: int) -> float:
    if not 0 <= bin_index < bins - 1:
        raise ValueError("bin_index must be in [0, bins - 2]")
    return float((bin_index + 1) * upper / bins)


def compare_searches(
    name: str,
    percentile: float,
    upper: float,
    histogram: np.ndarray,
    valid_pixels: int,
    excluded_pixels: int,
    config: SpaceOtsuConfig,
) -> PresetResult:
    workspace = OtsuWorkspace.from_histogram(histogram)
    exact = exact_threshold_search(workspace)
    nine = nine_grid_threshold_search(
        workspace,
        samples_per_cell=config.nine_grid_samples_per_cell,
        stop_size=config.nine_grid_stop_size,
        seed=config.nine_grid_seed,
    )
    score_gap = max(0.0, (exact.score - nine.score) / max(abs(exact.score), 1e-12))
    bin_distance = max(abs(exact.s_bin - nine.s_bin), abs(exact.t_bin - nine.t_bin))
    unstable = score_gap > config.unstable_score_gap or bin_distance > config.unstable_bin_distance
    return PresetResult(
        name=name,
        percentile=float(percentile),
        upper=float(upper),
        valid_pixels=int(valid_pixels),
        excluded_pixels=int(excluded_pixels),
        exact=exact,
        nine_grid=nine,
        score_gap=float(score_gap),
        bin_distance=int(bin_distance),
        nine_grid_unstable=bool(unstable),
        exact_threshold_s=bin_threshold_to_gray(exact.s_bin, upper, config.hist_bins),
        exact_threshold_t=bin_threshold_to_gray(exact.t_bin, upper, config.hist_bins),
        nine_threshold_s=bin_threshold_to_gray(nine.s_bin, upper, config.hist_bins),
        nine_threshold_t=bin_threshold_to_gray(nine.t_bin, upper, config.hist_bins),
    )


def apply_float_threshold(
    image: np.ndarray, mean_image: np.ndarray, threshold_s: float, threshold_t: float
) -> np.ndarray:
    gray = np.asarray(image, dtype=np.float32)
    mean = np.asarray(mean_image, dtype=np.float32)
    if gray.shape != mean.shape:
        raise ValueError("image and mean_image must have the same shape")
    mask = (gray > threshold_s) & (mean > threshold_t)
    return mask.astype(np.uint8) * 255


def connected_component_metrics(mask_u8: np.ndarray) -> dict[str, float | int]:
    mask = np.asarray(mask_u8) > 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) if num_labels > 1 else np.array([])
    metrics: dict[str, float | int] = {
        "foreground_pixels": int(mask.sum()),
        "foreground_ratio": float(mask.mean()),
        "component_count": int(max(0, num_labels - 1)),
    }
    for label, percentile in (("area_p50", 50), ("area_p90", 90), ("area_p99", 99)):
        metrics[label] = float(np.percentile(areas, percentile)) if areas.size else 0.0
    return metrics


def save_histogram_2d(
    histogram: np.ndarray,
    exact: SearchResult,
    nine_grid: SearchResult,
    save_path: str,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    display = np.log10(np.asarray(histogram, dtype=np.float64) + 1e-12)
    image = ax.imshow(display.T, cmap="hot", origin="lower", aspect="auto")
    ax.scatter(exact.s_bin, exact.t_bin, c="cyan", marker="x", s=70, label="exact")
    ax.scatter(
        nine_grid.s_bin,
        nine_grid.t_bin,
        facecolors="none",
        edgecolors="lime",
        marker="o",
        s=70,
        label="nine-grid",
    )
    ax.set_xlabel("Gray-level bin")
    ax.set_ylabel("Neighborhood-mean bin")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.colorbar(image, ax=ax, label="log10 probability")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def aggregate_histogram(histogram: np.ndarray, max_bins: int = 64) -> np.ndarray:
    source = np.asarray(histogram, dtype=np.float64)
    if source.shape[0] <= max_bins:
        return source
    factor = int(math.ceil(source.shape[0] / max_bins))
    padded_size = int(math.ceil(source.shape[0] / factor) * factor)
    padded = np.zeros((padded_size, padded_size), dtype=np.float64)
    padded[: source.shape[0], : source.shape[1]] = source
    return padded.reshape(padded_size // factor, factor, padded_size // factor, factor).sum(axis=(1, 3))


def save_histogram_3d_safe(histogram: np.ndarray, save_path: str, max_bins: int = 64) -> None:
    reduced = aggregate_histogram(histogram, max_bins=max_bins)
    size = reduced.shape[0]
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    z = reduced.ravel()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.hot(z / max(float(z.max()), 1e-12))
    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z), 0.8, 0.8, z, color=colors, shade=True)
    ax.set_xlabel("Aggregated gray bin")
    ax.set_ylabel("Aggregated mean bin")
    ax.set_zlabel("Probability")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
