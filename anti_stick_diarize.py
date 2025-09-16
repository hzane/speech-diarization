import numpy as np
import numba
import torch
import math
import onnxruntime as ort
import librosa
import pyloudnorm as pyln
import torchaudio.compliance.kaldi as Kaldi
import typer
from dataclasses import dataclass
from typing import List, Tuple, Callable

from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

from rich.progress import track
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import find_peaks
from speechbrain.pretrained import EncoderClassifier
from pathlib import Path
from functools import lru_cache


@dataclass
class Segment:
    start: float
    end: float
    spk: int | None = None
    score: float | None = None


def diar_read_audio(path_wav: str | tuple[np.ndarray, int], sr: int = 16000, lufs: float|None = -18.0):
    if isinstance(path_wav, (str, Path)):
        wav, sr = librosa.load(path_wav, sr=sr, mono=True, res_type="kaiser_fast")  # type: ignore
    else:
        wav, orig_sr = path_wav
        if wav.ndim ==2 and wav.shape[1] <= 2:
            wav = np.transpose(wav)
        if orig_sr != sr:
            wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr, res_type="kaiser_fast")

    wav = librosa.to_mono(wav)

    if lufs is not None:
        wav = loudness_normalize(wav, sr, target_lufs=lufs)

    wav = wav - np.mean(wav)
    wav = librosa.effects.preemphasis(wav, coef=0.97)
    return wav.astype(np.float32), sr


def loudness_normalize(
    y: np.ndarray, sr: int, target_lufs: float = -18.0
) -> np.ndarray:
    meter = pyln.Meter(sr)

    loudness = meter.integrated_loudness(y)
    y = pyln.normalize.loudness(y, loudness, target_lufs)

    return np.clip(y, -0.99, 0.99)


def fbank_batch(wavs: np.ndarray, # [B, n_samples]
                sr: int = 16000, n_mels: int = 80, mean_nor: bool = True,
                dither:float = 0)->np.ndarray:
    assert wavs.ndim == 2

    feat = Kaldi.fbank(
        torch.from_numpy(wavs),
        num_mel_bins=n_mels,
        sample_frequency=sr,
        dither=dither,
    )  # [B, T, n_mels]
    if mean_nor:
        feat = feat - feat.mean(1, keepdim=True)

    return feat.numpy()  # [1, T, n_mels]


@lru_cache(maxsize=1)
def using_ecapa_encoder(device: str = "cuda"):
    encoder = EncoderClassifier.from_hparams(
        source="LanceaKing/spkrec-ecapa-cnceleb",
        run_opts={"device": device}
    )
    return encoder


def ecapa_encode_batch(wavs: np.ndarray, sr: int = 16000, num_mels:int = 80):
    encoder = using_ecapa_encoder()
    x = torch.from_numpy(wavs).float()
    y = encoder.encode_batch(x).squeeze(0)
    return y


@lru_cache(maxsize=1)
def using_eres2netv2_encoder():
    # iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
    onnx_path = "models/iic-speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common.onnx"
    session = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    return session


def speaker_encoder_batch(wavs: np.ndarray, sr: int = 16000, num_mels:int = 80) -> np.ndarray:
    assert wavs.ndim == 2
    session = using_eres2netv2_encoder()

    features = fbank_batch(wavs, sr = sr, n_mels=num_mels, mean_nor=True)
    y = session.run(None, dict(feature=features))[0]
    return y.squeeze(0)  # [B, 192] #type: ignore


def frame_audio(y: np.ndarray, sr: int, win_ms: float = 30.0, hop_ms: float = 10.0):
    win = int(round(win_ms / 1000.0 * sr))
    hop = int(round(hop_ms / 1000.0 * sr))

    # 它默认会将数据填充到中心，我们可以通过 center=False 和 pad_mode='constant' 来模拟原始行为
    frames = librosa.util.frame(y, frame_length=win, hop_length=hop)

    return frames.T  # 转置以匹配原始输出形状 [n, win]


class SileroVAD:
    def __init__(self, device: str = "cpu"):
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad',
                                       force_reload=False, onnx=False, skip_validation=True) # type: ignore
        self.model.to(device).eval()
        self.device = device


@torch.no_grad()
def silero_probs_raw(vad: SileroVAD, y: np.ndarray, sr: int = 16000,
                     win_ms: float = 30.0, hop_ms: float = 10.0,
                     batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    frames = frame_audio(y, sr, win_ms, hop_ms)
    num_frames = frames.shape[0]
    probs = np.zeros(num_frames, dtype=np.float32)

    for i in range(0, num_frames, batch_size):
        batch = frames[i : i + batch_size]
        fb = torch.from_numpy(batch).to(vad.device)
        pb = vad.model(fb, sr)
        probs[i : i + len(pb)] = pb.cpu().numpy()

    return probs  # 对应 hop_ms 时间步


@numba.jit(nopython=True) # nopython=True 确保生成最高性能的代码
def hysteresis_binarize_numba(probs: np.ndarray, on: float = 0.6, off: float = 0.4) -> np.ndarray:
    mask = np.zeros(probs.shape, dtype=np.bool_)
    talking = False

    for i in range(probs.shape[0]):
        p = probs[i]
        if not talking and p >= on:
            talking = True
        elif talking and p < off:
            talking = False
        mask[i] = talking

    return mask


def morph_open_close(mask: np.ndarray, hop_ms: float,
                     open_ms: float = 80.0, close_ms: float = 40.0) -> np.ndarray:
    out = mask.copy()
    if open_ms > 0:
        se_open = np.ones(max(1, int(round(open_ms / hop_ms))), dtype=bool)
        out = binary_opening(out, structure=se_open)
    if close_ms > 0:
        se_close = np.ones(max(1, int(round(close_ms / hop_ms))), dtype=bool)
        out = binary_closing(out, structure=se_close)
    return out


def mask_to_segments(mask: np.ndarray, hop_ms: float,
                               min_speech_ms: float = 250.0,
                               min_gap_ms: float = 100.0) -> list[list[float]]:
    """
    将布尔VAD掩码转换为语音片段时间戳列表。

    参数:
        mask (np.ndarray): VAD输出的布尔掩码数组。
        hop_ms (float): VAD帧移的毫秒数。
        min_speech_ms (float): 被视为有效语音的最短持续时间（毫秒）。
        min_gap_ms (float): 用于合并两个相邻语音片段的最大静音间隙（毫秒）。

    返回:
        list[list[float]]: 一个包含 [开始时间, 结束时间] 的列表。
    """
    if not mask.any():  # 如果掩码中没有True，直接返回空列表
        return []

    min_speech_frames = round(min_speech_ms / hop_ms)
    min_gap_frames = round(min_gap_ms / hop_ms)
    hop_s = hop_ms / 1000.0

    # 在掩码首尾填充False，以捕捉边缘状态变化
    padded_mask = np.pad(mask, 1, constant_values=False)

    # 计算差分，1表示语音开始，-1表示语音结束
    diffs = np.diff(padded_mask.astype(np.int8))

    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return []

    # --- 过滤掉太短的语音片段 ---
    durations = ends - starts
    long_enough_indices = np.where(durations >= min_speech_frames)[0]

    starts = starts[long_enough_indices]
    ends = ends[long_enough_indices]

    if len(starts) == 0:
        return []

    # --- 合并相邻的小静音间隙 ---
    merged_segments = []
    # 初始化第一个片段
    current_start, current_end = starts[0], ends[0]

    for i in range(1, len(starts)):
        gap = starts[i] - current_end
        if gap <= min_gap_frames:
            current_end = ends[i]
        else:
            merged_segments.append([current_start, current_end])
            current_start, current_end = starts[i], ends[i]

    merged_segments.append([current_start, current_end])
    final_segments = [[round(s * hop_s, 3), round(e * hop_s, 3)] for s, e in merged_segments]

    return final_segments



# VAD：Silero + 去除那些持续时间极短的、被误判为语音的信号j
@lru_cache(maxsize=1)
def using_silero_vad_segments(
    threshold=0.5,
    min_speech=0.25, # default 250
    min_silence=0.10, # 100
    speech_pad=0.04, # 30
    bridge_ms=80,
    hop_s: float = 0.01,
    min_segment_duration_s: float = 0.10,
) -> Callable[[np.ndarray, int], list[Segment]]:
    model, utils = torch.hub.load(
        "snakers4/silero-vad",
        "silero_vad",
        onnx=False,
        force_reload=False,
        skip_validation=True,
    )  # type: ignore
    get_speech_timestamps = utils[0]

    def call_vad(
        y: np.ndarray,
        sr: int,
    ) -> list[Segment]:
        assert sr == 16000, "Sample rate must be 16000 Hz."
        assert y.ndim == 1 and y.dtype == np.float32, (
            "Audio must be a 1D float32 array."
        )
        wav = torch.from_numpy(y)
        with torch.no_grad():
            speech_timestamps = get_speech_timestamps(
                wav,
                model,
                sampling_rate=sr,
                threshold=threshold,
                min_speech_duration_ms=int(min_speech * 1000),
                min_silence_duration_ms=int(min_silence * 1000),
                # window_size_samples=512,
                speech_pad_ms=int(speech_pad * 1000),
            )
        if not speech_timestamps:
            return []

        n_frames = math.ceil(len(y) / sr / hop_s)
        mask = np.zeros(n_frames, dtype=bool)
        for seg in speech_timestamps:
            start_frame = int(seg["start"] / sr / hop_s)
            end_frame = int(seg["end"] / sr / hop_s)
            mask[start_frame:end_frame] = True

        # 应用形态学操作
        bridge_frames = max(1, int(round((bridge_ms / 1000.0) / hop_s)))
        structure = np.ones(bridge_frames, dtype=bool)
        processed_mask = binary_closing(mask, structure=structure) # or binary_opening

        # 性 能优化：使用NumPy将掩码转换回时间戳
        processed_mask = np.concatenate(([False], processed_mask, [False]))
        diffs = np.diff(processed_mask.astype(np.int8))

        starts = np.where(diffs == 1)[0] * hop_s
        ends = np.where(diffs == -1)[0] * hop_s

        # 组合并过滤最终结果
        final_segments = []
        for start, end in zip(starts, ends):
            if end - start >= min_segment_duration_s:
                final_segments.append(Segment(start, end))

        return final_segments

    return call_vad


def sliding_windows(
    start: float, end: float, win: float, step: float
) -> np.ndarray:
    if end < start + win:
        return np.empty((0, 2), dtype=np.float32)

    starts = np.arange(start, end - win + 1e-9, step)
    ends = starts + win

    return np.column_stack((starts, ends)) # np.vstack((starts, ends)).T # (N, 2)



# SCD：段内变更检测（滑窗嵌入 + 峰值）
def scd_split_segments(
    y: np.ndarray,
    sr: int,
    segments: list[Segment],
    win: float = 1.0,
    step: float = 0.2,
    thr: float = 1.25,
) -> List[Segment]:
    # 长语音段中，再根据音色变化进行二次切分
    assert y.ndim == 1 and y.dtype == np.float32 and sr == 16000

    out: List[Segment] = []
    min_win_samples = int(win * sr*0.8)

    for seg in track(segments, description="[cyan]SCD splitting"):
        if seg.end - seg.start < win * 1.5:
            out.append(seg)
            continue
        wins = sliding_windows(seg.start, seg.end, win, step)
        if len(wins) < 3:
            out.append(seg)
            continue

        snippets = []
        for s, e in wins:
            s, e = int(s*sr), int(e*sr)
            if e-s > min_win_samples:
                snippet = y[s:e]
                snippets.append(snippet)

        if len(snippets) < 3:
            out.append(seg)
            continue
        max_len = max(len(s) for s in snippets)
        wav_batch = np.zeros((len(snippets), max_len), dtype=np.float32)
        for i, s in enumerate(snippets):
            wav_batch[i, :len(s)] = s
        embs = speaker_encoder_batch(wav_batch, sr)

        sims = np.einsum('id,id->i', embs[:-1], embs[1:]) / (
            np.linalg.norm(embs[:-1], axis=1) * np.linalg.norm(embs[1:], axis=1) + 1e-8
        )
        dists = 1 - sims
        if np.std(dists)> 1e-6:
            z = (dists - dists.mean()) / dists.std()
        else:
            z = dists

        peaks, _ = find_peaks(z, height=thr)
        if peaks.size == 0:
            out.append(seg)
            continue

        mid_points = (wins[peaks, 1] + wins[peaks + 1, 0]) / 2.0
        # Filter out cuts too close to segment boundaries
        valid_cuts = mid_points[(mid_points > seg.start + 0.2) & (mid_points < seg.end - 0.2)]
        # Create final list of cut points (Python loop is fine here as #cuts is small)
        cuts = sorted(list({round(c, 2) for c in valid_cuts}))

        last_cut = seg.start
        for c in cuts:
            if c - last_cut >= 0.12:
                out.append(Segment(last_cut, c))
                last_cut = c

        if seg.end - last_cut >= 0.08:
            out.append(Segment(last_cut, seg.end))
    return out


def embed_segments(y: np.ndarray, sr: int, segs: List[Segment]) -> np.ndarray:
    snippets = []
    for seg in track(segs, description="[magenta]Embedding segments"):
        s, e = int(seg.start*sr), int(seg.end*sr)

        a = y[s:e]
        if len(a) < int(0.2 * sr):
            pad = int(0.15 * sr)
            s_padded = max(0, s - pad)
            e_padded = min(len(y), e + pad)
            a = y[s_padded:e_padded]
        if a.size > 0:
            snippets.append(a)

    if not snippets:
        return np.empty((0, 192), dtype=np.float32)

    max_len = max(len(s) for s in snippets)
    wav_batch = np.zeros((len(snippets), max_len), dtype=np.float32)
    for i, s in enumerate(snippets):
        wav_batch[i, :len(s)] = s

    print("[magenta]Embedding segments (batch inference)...")
    embs = speaker_encoder_batch( wav_batch, sr)
    return embs


def cluster_hdbscan(embs: np.ndarray, min_cluster_size:int = 2) -> np.ndarray:
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    D = 1 - cosine_similarity(embs_norm)

    clu = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=None, allow_single_cluster=True, metric='precomputed')
    labels = clu.fit_predict(D)
    return labels


def cluster_hdbscan_two_stage(
    embs: np.ndarray,
    min_cluster_size: int = 2,
) -> np.ndarray:
    """
    两阶段HDBSCAN聚类方法。

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
        metric='euclidean', # 在归一化的嵌入上，欧氏距离与余弦距离相关
        allow_single_cluster=True
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

    # --- 阶段二：对质心进行HDBSCAN聚类 ---
    print("[Cluster] Stage 2: Re-clustering centroids with HDBSCAN...")
    if len(centroids) < min_cluster_size:
        # 如果质心数量少于最小簇大小，则将它们视为一个簇
        stage2_labels = np.zeros(len(centroids), dtype=int)
    else:
        stage2_clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None,
            metric='euclidean', # 同样使用欧氏距离
            allow_single_cluster=True
        )
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        stage2_labels = stage2_clusterer.fit_predict(centroids_norm)

    # --- 将质心的标签映射回原始的每个片段 ---
    final_labels = np.full(num_segments, -1, dtype=int) # 默认为噪声

    for stage1_label, stage2_idx in centroid_map.items():
        final_spk_label = stage2_labels[stage2_idx]
        if final_spk_label != -1: # 如果质心本身不是噪声
            # 找到所有属于这个微簇的原始片段，并赋予最终标签
            original_indices = np.where(stage1_labels == stage1_label)[0]
            final_labels[original_indices] = final_spk_label

    n_speakers = np.max(final_labels) + 1
    print(f"[Cluster] Stage 2 identified {n_speakers} final speakers.")

    return final_labels



def conservative_merge_optimized(
    segs: list[Segment],
    embs: np.ndarray, # Takes pre-computed embeddings as input
    max_gap: float = 0.25, # Using a more realistic default
    max_turn: float = 10.0,
    min_cos: float = 0.80, # Using a more realistic default
) -> list[Segment]:
    """
    Merges adjacent segments of the same speaker conservatively using pre-computed embeddings.
    This version is dramatically faster by avoiding inference inside the loop.
    """
    if not segs:
        return []

    # 1. Preparation: Combine segments with their embeddings and sort by time.
    # We assume segs[i] corresponds to embs[i].
    # The 'spk' attribute should already be set on the segments from the clustering step.

    # Create a list of tuples to keep everything synchronized after sorting
    # We store (segment, embedding)
    indexed_items = sorted(
        [(seg, embs[i]) for i, seg in enumerate(segs)],
        key=lambda item: (item[0].start, item[0].end)
    )

    merged_items: List[tuple[Segment, np.ndarray]] = []

    for current_seg, current_emb in track(indexed_items, description="[green]Merging segments"):
        if not merged_items:
            merged_items.append((current_seg, current_emb))
            continue

        last_seg, last_emb = merged_items[-1]

        # 2. Check conditions for a potential merge
        conditions_met = (
            current_seg.spk == last_seg.spk and
            current_seg.start - last_seg.end <= max_gap and
            current_seg.end - last_seg.start <= max_turn
        )

        if conditions_met:
            # 3. Perform the similarity check (now extremely fast)
            # Normalize embeddings for accurate cosine similarity calculation
            last_emb_norm = last_emb / (np.linalg.norm(last_emb) + 1e-8)
            current_emb_norm = current_emb / (np.linalg.norm(current_emb) + 1e-8)
            cos_sim = np.dot(last_emb_norm, current_emb_norm)

            if cos_sim >= min_cos:
                # 4. Merge the segments
                # Update the end time of the last segment
                last_seg.end = current_seg.end

                # IMPORTANT: Update the embedding of the merged segment.
                # A simple weighted average (by duration) or a simple average is robust.
                # Let's use a simple average and re-normalize for the next comparison.
                new_emb = last_emb + current_emb
                new_emb_normalized = new_emb / (np.linalg.norm(new_emb) + 1e-8)

                # Update the tuple in the list
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


# 帧级重分配
def frame_reassign_optimized(
    y: np.ndarray,
    sr: int,
    speech_mask: List[Segment], # The initial VAD result
    segs: List[Segment],        # The segments after merging (e.g., speech3)
    embs: np.ndarray,           # Pre-computed embeddings for the segments
    smooth_step: float = 0.1,
    win: float = 1.0,           # A longer window is more stable
) -> List[Segment]:
    if not segs or embs.size == 0:
        return []

    # --- 1. Optimized Centroid Calculation (No Inference) ---
    # We use the pre-computed embeddings passed into the function.
    print("[Reassign] Calculating speaker centroids...")
    spk_ids = sorted(list({s.spk for s in segs if s.spk is not None and s.spk >= 0}))
    labels = np.array([s.spk for s in segs])

    centroids: dict[int, np.ndarray] = {}
    for sid in spk_ids:
        # Use boolean indexing to get all embeddings for a speaker in one go
        embs_for_spk = embs[labels == sid]
        centroids[sid] = np.mean(embs_for_spk, axis=0)
        # Normalize centroid for efficient cosine similarity calculation later
        centroids[sid] /= (np.linalg.norm(centroids[sid]) + 1e-8)

    if not centroids:
        return segs # Return the merged segments if no valid centroids were found

    # --- 2. Prepare for Batch Processing ---
    # Generate all sliding window timestamps for the entire audio
    max_t = y.shape[0] / sr
    win_samples = int(win * sr)
    step_samples = int(smooth_step * sr)

    # Create an array of window start samples
    window_starts = np.arange(0, len(y) - win_samples, step_samples)
    window_ends = window_starts + win_samples

    # --- 3. Efficient VAD Filtering ---
    # Create a high-resolution VAD mask for quick lookups
    hop_s = 0.01 # Small hop for the mask
    n_frames = math.ceil(max_t / hop_s)
    smask = np.zeros(n_frames, dtype=bool)
    for sm in speech_mask:
        start_frame = int(sm.start / hop_s)
        end_frame = int(sm.end / hop_s)
        smask[start_frame:end_frame] = True

    # Filter windows to only those whose center falls within a speech region
    window_centers_s = (window_starts + win_samples / 2) / sr
    window_center_frames = (window_centers_s / hop_s).astype(int)
    # Ensure indices are within bounds
    window_center_frames = np.clip(window_center_frames, 0, n_frames - 1)

    valid_indices = np.where(smask[window_center_frames])[0]
    if valid_indices.size == 0:
        return segs # Return previous stage if no speech regions are found for reassignment

    # --- 4. Batch Inference (The Core Optimization) ---
    # Extract all valid audio snippets
    audio_snippets = [y[window_starts[i]:window_ends[i]] for i in valid_indices]

    # Pad and stack into a single batch
    wav_batch = np.zeros((len(audio_snippets), win_samples), dtype=np.float32)
    for i, snippet in enumerate(audio_snippets):
        wav_batch[i, :len(snippet)] = snippet

    print(f"[Reassign] Embedding {len(wav_batch)} windows for reassignment (1 batch call)...")
    window_embs = speaker_encoder_batch(wav_batch, sr)
    window_embs /= (np.linalg.norm(window_embs, axis=1, keepdims=True) + 1e-8)

    # --- 5. Batch Similarity Calculation ---
    # Create centroid matrix [num_speakers, emb_dim]
    c_matrix = np.array([centroids[sid] for sid in spk_ids])

    # Calculate cosine similarity for all windows against all centroids in one go
    # Resulting shape: [num_windows, num_speakers]
    sim_matrix = np.dot(window_embs, c_matrix.T)

    # Find the best speaker for each window
    best_spk_indices = np.argmax(sim_matrix, axis=1)

    # Map back to original speaker IDs
    window_labels = np.array([spk_ids[i] for i in best_spk_indices])

    # --- 6. Vectorized Conversion from Labels to Segments ---
    # Create a full timeline of labels, marking non-speech as -1
    full_labels = np.full(len(window_starts), -1, dtype=int)
    full_labels[valid_indices] = window_labels

    # Find where the speaker label changes
    change_points = np.where(np.diff(full_labels, prepend=np.nan))[0]

    refined: List[Segment] = []
    for start_idx, end_idx in zip(change_points, list(change_points[1:]) + [len(full_labels)]):
        speaker_id = full_labels[start_idx]
        if speaker_id != -1:
            start_time = window_starts[start_idx] / sr
            # The end time is the start of the *next* segment
            end_time = window_starts[end_idx] / sr if end_idx < len(window_starts) else max_t
            refined.append(Segment(start_time, end_time, speaker_id)) # type: ignore

    # --- 7. Final Merging ---
    # The final step from the original code is still useful
    final = merge_adjacent(refined, gap=0.05)
    return final



def merge_adjacent(segs: List[Segment], gap: float = 0.05) -> List[Segment]:
    print('merge-adjacent', len(segs))
    segs = sorted(segs, key=lambda s: (s.start, s.end))
    out: List[Segment] = []
    for s in segs:
        if not out:
            out.append(s)
            continue
        last = out[-1]
        if s.spk == last.spk and s.start - last.end <= gap:
            last.end = max(last.end, s.end)
        else:
            out.append(s)
    return out


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
    wav_path: str|tuple[np.ndarray, int],
    sr: int = 16000,
    target_lufs: float = -18.0,
    vad_thr: float = 0.55,
    min_speech: float = 0.15,
    min_silence: float = 0.10,
    speech_pad: float = 0.07,
    morph_bridge_ms: float = 80.0,
    scd_win: float = 0.8,
    scd_step: float = 0.2,
    scd_thr: float = 1.50,
    merge_gap: float = .25,
    merge_maxturn: float = 10.0,
    merge_mincos: float = 0.8,
    reseg: int = 1,
) -> list[Segment]:
    y, sr = diar_read_audio(wav_path, sr, lufs = target_lufs)

    # 1) 短停顿友好的 VAD
    speech = using_silero_vad_segments(
        threshold=vad_thr,
        min_speech=min_speech,
        min_silence=min_silence,
        speech_pad=speech_pad,
        bridge_ms=morph_bridge_ms,
    )(y, sr)
    if not speech:
        return []

    # 2) 段内 SCD
    speech2 = scd_split_segments(y, sr, speech, win=scd_win, step=scd_step, thr=scd_thr)
    print('segments',len(speech), len(speech2))

    # 3) 嵌入 + 聚类
    embs = embed_segments(y, sr, speech2)
    print('embeddings', embs.shape)

    labels = cluster_hdbscan_two_stage(embs, min_cluster_size=2)
    for s, lab in zip(speech2, labels):
        s.spk = int(lab)
    print('cluster', labels)

    # 4) 防粘连合并
    speech3 = conservative_merge_optimized(
        speech2,
        labels,
        max_gap=merge_gap,
        max_turn=merge_maxturn,
        min_cos=merge_mincos,
    )
    embs_for_reassign = embed_segments(y, sr, speech3)
    # 5) 帧级重分配
    if reseg:
        speech4 = frame_reassign_optimized(y, sr, speech, speech3, embs_for_reassign, smooth_step=0.10, win=1.0)
    else:
        speech4 = speech3

    # overlaps = detect_overlap_pyannote(wav_path)
    # meta = {"overlaps": [{"start": o.start, "end": o.end} for o in overlaps]}
    final = merge_adjacent(speech4, gap=0.05)
    print('final', len(final))
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
