import numpy as np
import torch
import math
import onnxruntime as ort
import librosa
import pyloudnorm as pyln

import typer
from dataclasses import dataclass
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from rich.progress import track

from scipy.signal import find_peaks
from pathlib import Path
from functools import lru_cache
from speech_encode import ecapa_encode_batch
from vad import silero_vad_segments, frame_audio


@dataclass
class Segment:
    start: float
    end: float
    spk: int | None = None
    score: float | None = None


def diar_read_audio(
    path_wav: str | tuple[np.ndarray, int], sr: int = 16000, lufs: float | None = -18.0
):
    if isinstance(path_wav, (str, Path)):
        wav, sr = librosa.load(path_wav, sr=sr, mono=True, res_type="kaiser_fast")  # type: ignore
    else:
        wav, orig_sr = path_wav
        if wav.ndim == 2 and wav.shape[1] <= 2:
            wav = np.transpose(wav)
        if orig_sr != sr:
            wav = librosa.resample(
                wav, orig_sr=orig_sr, target_sr=sr, res_type="kaiser_fast"
            )

    wav = librosa.to_mono(wav)

    if lufs is not None:
        wav = loudness_normalize(wav, sr, target_lufs=lufs)

    wav = wav - np.mean(wav) # type: ignore
    wav = librosa.effects.preemphasis(wav, coef=0.97)
    return wav.astype(np.float32), sr


def loudness_normalize(
    y: np.ndarray, sr: int, target_lufs: float = -18.0
) -> np.ndarray:
    meter = pyln.Meter(sr)

    loudness = meter.integrated_loudness(y)
    y = pyln.normalize.loudness(y, loudness, target_lufs)

    return np.clip(y, -0.99, 0.99)


"""def sliding_windows(
    start: float, end: float, win: float, step: float
) -> np.ndarray:
    if end < start + win:
        return np.empty((0, 2), dtype=np.float32)

    starts = np.arange(start, end - win + 1e-9, step)
    ends = starts + win

    return np.column_stack((starts, ends)) # np.vstack((starts, ends)).T # (N, 2)
"""


# SCD：段内变更检测（滑窗嵌入 + 峰值）
def scd_split_segments(
    y: np.ndarray,
    sr: int,
    segments: list[Segment],
    win_ms: float = 1000.0,
    hop_ms: float = 200.0,
    thr: float = 1.25,
    min_speech_ms: float = 1000.0,
) -> list[Segment]:
    # 长语音段中，再根据音色变化进行二次切分
    assert y.ndim == 1 and y.dtype == np.float32
    min_speech_s = min_speech_ms / 1000.0

    out: list[Segment] = []

    for seg in track(segments, description="[cyan]SCD splitting"):
        subaudio = y[int(seg.start * sr) : int(seg.end * sr)]
        snippets = frame_audio(subaudio, sr, win_ms=win_ms, hop_ms=hop_ms)
        if len(snippets) < 3:
            out.append(seg)
            continue

        embs = ecapa_encode_batch(snippets)

        sims = np.einsum("id,id->i", embs[:-1], embs[1:]) / (
            np.linalg.norm(embs[:-1], axis=1) * np.linalg.norm(embs[1:], axis=1) + 1e-8
        )
        dists = 1 - sims
        if np.std(dists) > 1e-6:
            z = (dists - dists.mean()) / dists.std()
        else:
            z = dists

        peaks, _ = find_peaks(z, height=thr)
        if peaks.size == 0:
            out.append(seg)
            continue

        mid_points = seg.start + (peaks + 0.5) * hop_ms / 1000.0
        cuts = sorted(list(set(mid_points)))

        last_cut = seg.start
        for cut_time in cuts:
            if cut_time - last_cut >= min_speech_s:
                out.append(Segment(last_cut, cut_time))
                last_cut = cut_time

        if seg.end - last_cut >= min_speech_s:
            out.append(Segment(last_cut, seg.end))
    return out


def embed_segments(
    y: np.ndarray,
    sr: int,
    segs: list[Segment],
    batch_size: int = 32,
    min_duration_ms: float = 500.0,
    pad_duration_ms: float = 150.0,
) -> np.ndarray:
    """
    Returns:
        np.ndarray: An array of embeddings of shape (num_segments, embed_dim).
    """
    num_segs = len(segs)
    if num_segs == 0:
        return np.empty((0, 192), dtype=np.float32)
    min_duration_samples = int(min_duration_ms / 1000.0 * sr)
    pad_samples = int(pad_duration_ms / 1000.0 * sr)

    embs = []
    desc = "[magenta]Embedding segments"
    for i in track(range(0, num_segs, batch_size), description=desc):
        batch_segs = segs[i : i + batch_size]

        batch_snippets = []
        for seg in batch_segs:
            s, e = int(seg.start * sr), int(seg.end * sr)
            snippet = y[s:e]
            if snippet.shape[0] < min_duration_samples:
                s_padded = max(0, s - pad_samples)
                e_padded = min(len(y), e + pad_samples)
                snippet = y[s_padded:e_padded]
            batch_snippets.append(snippet)

        max_len = max(len(s) for s in batch_snippets)
        wav_batch = np.zeros((len(batch_snippets), max_len), dtype=np.float32)
        for i, s in enumerate(batch_snippets):
            wav_batch[i, : len(s)] = s

        embs_batch = ecapa_encode_batch(wav_batch)
        embs.append(embs_batch)

    embs = np.concatenate(embs, axis=0)
    return embs


def cluster_hdbscan(embs: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    D = 1 - cosine_similarity(embs_norm)

    clu = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        allow_single_cluster=True,
        metric="precomputed",
    )
    labels = clu.fit_predict(D)
    return labels


def cluster_hdbscan_two_stage(
    embs: np.ndarray,
    min_cluster_size: int = 2,
) -> np.ndarray:
    """
    Args:
        embs: 嵌入向量数组 (N, D)。
        min_cluster_size: 第一阶段过聚类的最小簇大小。值越小，微簇越多。

    Returns:
        最终的聚类标签数组。
    """
    num_segments = embs.shape[0]
    # 归一化嵌入向量，使其在单位球面上，此时欧氏距离与余弦相似度单调相关
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    print("[Cluster] Stage 1: Over-clustering with HDBSCAN...")
    stage1_clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        metric="euclidean",  # 在归一化的嵌入上，欧氏距离与余弦距离相关
        allow_single_cluster=True,
    )
    stage1_labels = stage1_clusterer.fit_predict(embs_norm)

    # 获取第一阶段识别出的微簇数量（排除噪声-1）
    n_micro_clusters = np.max(stage1_labels) + 1
    print(f"[Cluster] Stage 1 found {n_micro_clusters} micro-clusters.")

    if n_micro_clusters < 1:
        # 如果没有找到任何簇，将所有片段视为一个说话人或噪声
        return np.zeros(num_segments, dtype=int) if num_segments > 0 else np.array([])

    # --- 计算每个微簇的质心 ---
    centroids = []
    # 记录每个质心对应的原始微簇标签
    centroid_map = {}
    centroid_idx = 0
    for i in range(n_micro_clusters):
        members = embs[stage1_labels == i]
        if len(members) > 0:
            centroid = np.mean(members, axis=0)
            centroids.append(centroid)
            centroid_map[i] = centroid_idx
            centroid_idx += 1

    if not centroids:
        return np.zeros(num_segments, dtype=int)

    centroids = np.array(centroids)

    # 对质心进行HDBSCAN聚类
    print("[Cluster] Stage 2: Re-clustering centroids with HDBSCAN...")
    if len(centroids) < min_cluster_size:
        # 如果质心数量少于最小簇大小，则将它们视为一个簇
        stage2_labels = np.zeros(len(centroids), dtype=int)
    else:
        stage2_clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None,
            metric="euclidean",  # 同样使用欧氏距离
            allow_single_cluster=True,
        )
        centroids_norm = centroids / (
            np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
        )
        stage2_labels = stage2_clusterer.fit_predict(centroids_norm)

    # --- 将质心的标签映射回原始的每个片段 ---
    final_labels = np.full(num_segments, -1, dtype=int)  # 默认为噪声

    for stage1_label, stage2_idx in centroid_map.items():
        final_spk_label = stage2_labels[stage2_idx]
        if final_spk_label != -1:  # 如果质心本身不是噪声
            # 找到所有属于这个微簇的原始片段，并赋予最终标签
            original_indices = np.where(stage1_labels == stage1_label)[0]
            final_labels[original_indices] = final_spk_label

    n_speakers = np.max(final_labels) + 1
    print(f"[Cluster] Stage 2 identified {n_speakers} final speakers.")

    return final_labels


def conservative_merge(
    segs: list[Segment],
    embs: np.ndarray,  # Takes pre-computed embeddings as input
    max_gap_s: float = 0.5,  # Using a more realistic default
    max_turn_s: float = 30.0,
    min_cos: float = 0.80,  # Using a more realistic default
) -> list[Segment]:
    if not segs:
        return []

    # Preparation: Combine segments with their embeddings and sort by time.
    # The 'spk' attribute should already be set on the segments from the clustering step.

    indexed_items = sorted(
        [(seg, embs[i]) for i, seg in enumerate(segs)],
        key=lambda item: (item[0].start, item[0].end),
    )

    merged_items: list[tuple[Segment, np.ndarray]] = []

    for current_seg, current_emb in track(
        indexed_items, description="[green]Merging segments"
    ):
        if not merged_items:
            merged_items.append((current_seg, current_emb))
            continue

        last_seg, last_emb = merged_items[-1]

        # Check conditions for a potential merge
        conditions_met = (
            current_seg.spk == last_seg.spk
            and current_seg.start - last_seg.end <= max_gap_s
            and current_seg.end - last_seg.start <= max_turn_s
        )

        if conditions_met:
            last_emb_norm = last_emb / (np.linalg.norm(last_emb) + 1e-8)
            current_emb_norm = current_emb / (np.linalg.norm(current_emb) + 1e-8)
            cos_sim = np.dot(last_emb_norm, current_emb_norm)

            if cos_sim >= min_cos:
                last_seg.end = current_seg.end

                new_emb = last_emb + current_emb
                new_emb_normalized = new_emb / (np.linalg.norm(new_emb) + 1e-8)

                merged_items[-1] = (last_seg, new_emb_normalized)
            else:
                # Similarity check failed, do not merge
                merged_items.append((current_seg, current_emb))
        else:
            # Initial conditions failed, do not merge
            merged_items.append((current_seg, current_emb))

    # 5. Extract just the final segments from our list of tuples
    final_segments = [item[0] for item in merged_items]
    return final_segments


def speaker_centroids(segs:list[Segment], embs: np.ndarray)->tuple[np.ndarray, np.ndarray]:
    spk_ids = sorted(list({s.spk for s in segs if s.spk is not None and s.spk >= 0}))
    labels = np.array([s.spk for s in segs])

    centroids: dict[int, np.ndarray] = {}
    for sid in spk_ids:
        # Use boolean indexing to get all embeddings for a speaker in one go
        embs_for_spk = embs[labels == sid]
        centroids[sid] = np.mean(embs_for_spk, axis=0)
        # Normalize centroid for efficient cosine similarity calculation later
        centroids[sid] /= np.linalg.norm(centroids[sid]) + 1e-8
    if not centroids:
        return np.empty(0, dtype=int), np.empty((0, 192), dtype=np.float32)

    labels = np.array(centroids.keys())
    centroid_embs = np.stack(list(centroids.values()))
    return labels, centroid_embs


def _get_speech_windows(y: np.ndarray, sr: int, speech_mask: list[Segment], win_samples: int, step_samples: int):
    """生成所有落在语音区域内的滑动窗口索引。"""
    max_t = len(y) / sr
    hop_s = 0.01  # VAD掩码的分辨率
    n_frames = math.ceil(max_t / hop_s)
    smask = np.zeros(n_frames, dtype=bool)
    for sm in speech_mask:
        s, e = int(sm.start / hop_s), int(sm.end / hop_s)
        smask[s:e] = True

    window_starts = np.arange(0, len(y) - win_samples, step_samples)
    window_centers_s = (window_starts + win_samples / 2) / sr
    window_center_frames = np.clip((window_centers_s / hop_s).astype(int), 0, n_frames - 1)

    valid_indices = np.where(smask[window_center_frames])[0]
    return window_starts, valid_indices


def _labels_to_segments(window_starts: np.ndarray, valid_indices: np.ndarray,
                        window_labels: np.ndarray, sr: int, max_t: float) -> list[Segment]:
    """将帧级标签向量化地转换为Segment列表。"""
    full_labels = np.full(len(window_starts), -1, dtype=int)
    full_labels[valid_indices] = window_labels

    change_points = np.where(np.diff(full_labels, prepend=np.nan))[0]

    refined_segs = []
    for start_idx, end_idx in zip(change_points, list(change_points[1:]) + [len(full_labels)]):
        spk_id = int(full_labels[start_idx])
        if spk_id != -1:
            start_time = window_starts[start_idx] / sr
            end_time = window_starts[end_idx] / sr if end_idx < len(window_starts) else max_t
            if end_time > start_time:
                refined_segs.append(Segment(start_time, end_time, spk_id)) # type: ignore
    return refined_segs


# 帧级重分配
def frame_reassign(
    y: np.ndarray,
    sr: int,
    speech_mask: list[Segment],  # The initial VAD result
    segs: list[Segment],  # The segments after merging (e.g., speech3)
    embs: np.ndarray,  # Pre-computed embeddings for the segments
    smooth_step: float = 0.1,
    win: float = 1.0,  # A longer window is more stable
    batch_size: int = 128,
) -> list[Segment]:
    if not segs or embs.size == 0:
        return []

    print("[Reassign] Calculating speaker centroids...")
    spk_ids, c_matrix = speaker_centroids(segs, embs)

    if c_matrix.size == 0:
        return segs  # Return the merged segments if no valid centroids were found

    # Generate all sliding window timestamps for the entire audio
    max_t = y.shape[0] / sr
    win_samples = int(win * sr)
    step_samples = int(smooth_step * sr)
    window_starts, valid_indices = _get_speech_windows(y, sr, speech_mask, win_samples, step_samples)

    if valid_indices.size == 0:
        return segs  # Return previous stage if no speech regions are found for reassignment

    all_window_embs = []
    desc = "[cyan]Reassigning Frames (mini-batches)..."
    for i in track(range(0, len(valid_indices), batch_size), description=desc):
        batch_indices = valid_indices[i : i + batch_size]

        snippets = [y[window_starts[j] : window_starts[j] + win_samples] for j in batch_indices]
        wav_batch = np.stack(snippets, axis=0) # 所有窗口长度相同，直接堆叠

        embs_batch = ecapa_encode_batch(wav_batch)
        all_window_embs.append(embs_batch)

    window_embs = np.concatenate(all_window_embs, axis=0)
    window_embs /= np.linalg.norm(window_embs, axis=1, keepdims=True) + 1e-8

    # Resulting shape: [num_windows, num_speakers]
    sim_matrix = np.dot(window_embs, c_matrix.T)
    best_spk_indices = np.argmax(sim_matrix, axis=1)
    window_labels = np.array([spk_ids[i] for i in best_spk_indices])

    refined_segs = _labels_to_segments(window_starts, valid_indices, window_labels, sr, len(y) / sr)
    # Create a full timeline of labels, marking non-speech as -1
    full_labels = np.full(len(window_starts), -1, dtype=int)
    full_labels[valid_indices] = window_labels

    # Find where the speaker label changes
    change_points = np.where(np.diff(full_labels, prepend=np.nan))[0]

    refined: list[Segment] = []
    for start_idx, end_idx in zip(
        change_points, list(change_points[1:]) + [len(full_labels)]
    ):
        speaker_id = full_labels[start_idx]
        if speaker_id != -1:
            start_time = window_starts[start_idx] / sr
            # The end time is the start of the *next* segment
            end_time = (
                window_starts[end_idx] / sr if end_idx < len(window_starts) else max_t
            )
            refined.append(Segment(start_time, end_time, speaker_id))  # type: ignore

    # The final step from the original code is still useful
    final = merge_adjacent(refined, gap=0.05)
    return final



def merge_adjacent(segments: list[Segment], gap: float=0.05) -> list[Segment]:
    # 辅助函数：合并属于同一个说话人的相邻片段
    if not segments:
        return []
    merged = [segments[0]]
    for next_seg in segments[1:]:
        last_seg = merged[-1]
        if next_seg.spk == last_seg.spk and (next_seg.start - last_seg.end) <= gap:
            merged[-1] = Segment(last_seg.start, next_seg.end, last_seg.spk)
        else:
            merged.append(next_seg)
    return merged


# OSD
# def detect_overlap_pyannote(path: str) -> List[Segment]:
#     try:
#         from pyannote.audio import Pipeline
#         pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection")
#         diar = pipeline(path)
#         overlaps: List[Segment] = []
#         for turn, _, _ in diar.itertracks(yield_label=True):
#             overlaps.append(Segment(float(turn.start), float(turn.end)))
#         return overlaps
#     except Exception:
#         return []


# 主流程
def diarize(
    wav_path: str | tuple[np.ndarray, int],
    sr: int = 16000,
    target_lufs: float = -18.0,
    vad_on_thr: float = 0.6,
    vad_off_thr: float = 0.4,
    min_speech_ms: float = 250,
    min_silence_ms: float = 100,
    speech_pad_ms: float = 70.,
    morph_open_ms: float = 80.0,
    morph_close_ms: float = 40.0,
    scd_win_ms: float = 1000.,
    scd_hop_ms: float = 200,
    scd_thr: float = 1.50,
    merge_max_gap_s: float = 0.5,
    merge_max_speech_s: float = 30.0,
    merge_mincos: float = 0.8,
    reseg: int = 1,
) -> list[Segment]:
    y, sr = diar_read_audio(wav_path, sr, lufs=target_lufs)

    # 1) 短停顿友好的 VAD
    speech = silero_vad_segments(
        y, sr,
        on_threshold=vad_on_thr,
        off_threshold=vad_off_thr,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        morph_open_ms=morph_open_ms,
        morph_close_ms=morph_close_ms,
    )
    if not speech:
        return []

    speech = [Segment(s, e) for s, e in speech]
    speech2 = scd_split_segments(y, sr, speech, win_ms=scd_win_ms, hop_ms=scd_hop_ms, thr=scd_thr)
    print("segments", len(speech), len(speech2))

    embs = embed_segments(y, sr, speech2)
    print("embeddings", embs.shape)

    labels = cluster_hdbscan_two_stage(embs, min_cluster_size=2)
    for s, lab in zip(speech2, labels):
        s.spk = int(lab)
    print("cluster", labels)

    speech3 = conservative_merge(
        speech2,
        labels,
        max_gap_s=merge_max_gap_s,
        max_turn_s=merge_max_speech_s,
        min_cos=merge_mincos,
    )
    embs_for_reassign = embed_segments(y, sr, speech3)
    # 5) 帧级重分配
    if reseg:
        speech4 = frame_reassign(
            y, sr, speech, speech3, embs_for_reassign, smooth_step=0.10, win=1.0
        )
    else:
        speech4 = speech3

    # overlaps = detect_overlap_pyannote(wav_path)
    # meta = {"overlaps": [{"start": o.start, "end": o.end} for o in overlaps]}
    final = merge_adjacent(speech4, gap=merge_max_gap_s)
    print("final", len(final))
    return final


def main(
    wav_path: str,
    sr: int = 16000,
    target_lufs: float = -18.0,
    vad_thr: float = 0.55,
    min_speech: float = 0.15,
    min_silence: float = 0.10,
    speech_pad: float = 0.04,
    morph_bridge_ms: float = 80.0,
    scd_win: float = 0.8,
    scd_step: float = 0.2,
    scd_thr: float = 1.2,
    cluster_cos: float = 0.65,
    merge_gap: float = 0.25,
    merge_maxturn: float = 10.0,
    merge_mincos: float = 0.7,
    reseg: int = 1,
):
    print(f"[bold]Loading:[/bold] {wav_path}")
    final = diarize(
        wav_path,
        sr,
        target_lufs,
        vad_thr,
        min_speech,
        min_silence,
        speech_pad,
        morph_bridge_ms,
        scd_win,
        scd_step,
        scd_thr,
        cluster_cos,
        merge_gap,
        merge_maxturn,
        merge_mincos,
        reseg,
    )
    print(
        f"[green]Segments:{len(final)}; Speakers:{len(set([s.spk for s in final]))}[/green]"
    )
    for i, s in enumerate(final[:10], 1):
        print(f"{i:02d}  {s.start:.2f}–{s.end:.2f}  SPK_{s.spk}")


if __name__ == "__main__":
    typer.run(main)
