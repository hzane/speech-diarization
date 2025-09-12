import numpy as np
import torch
import math
import onnxruntime as ort
import librosa
import pyloudnorm as pyln
import torchaudio.compliance.kaldi as Kaldi
import typer
from dataclasses import dataclass
from typing import List, Tuple, Callable

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from rich import print
from rich.progress import track
from scipy.ndimage import binary_closing
from scipy.signal import find_peaks
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
def using_speaker_encoder(num_mels: int = 80):
    # iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
    onnx_path = "models/iic-speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common.onnx"
    session = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    def batch(wavs: np.ndarray, sr: int = 16000) -> np.ndarray:
        assert wavs.ndim == 2

        features = fbank_batch(wavs, sr = sr, n_mels=num_mels, mean_nor=True)
        y = session.run(None, dict(feature=features))[0]
        return y.squeeze(0)  # [B, 192] #type: ignore

    return batch


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


# SCD：段内变更检测（滑窗嵌入 + 峰值）
def sliding_windows(
    start: float, end: float, win: float, step: float
) -> np.ndarray:
    if end < start + win:
        return np.empty((0, 2), dtype=np.float32)

    starts = np.arange(start, end - win + 1e-9, step)
    ends = starts + win

    return np.vstack((starts, ends)).T # (N, 2)


def local_peaks_1d(x: np.ndarray, thr: float) -> List[int]:
    peak_indices, _ = find_peaks(x, height=thr)
    return peak_indices.tolist()


def scd_split_segments(
    y: np.ndarray,
    sr: int,
    segments: list[Segment],
    win: float = 0.8,
    step: float = 0.2,
    thr: float = 1.25,
) -> List[Segment]:
    # 长语音段中，再根据音色变化进行二次切分
    assert y.ndim == 1 and y.dtype == np.float32 and sr == 16000

    out: List[Segment] = []
    win_samples = int(win * sr)
    min_win_samples = int(step * sr*0.8)

    for seg in track(segments, description="[cyan]SCD splitting"):
        if seg.end - seg.start < win * 1.5:
            out.append(seg)
            continue
        wins = sliding_windows(seg.start, seg.end, win, step)
        if len(wins) < 3:
            out.append(seg)
            continue

        embs = []
        for s, e in wins:
            a = y[int(s * sr) : int(e * sr)]
            if len(a) < int(win * sr * 0.8):
                continue
            embs.append(using_speaker_encoder()(a, sr))
        if len(embs) < 3:
            out.append(seg)
            continue

        embs = np.vstack(embs)
        sims = np.sum(embs[:-1] * embs[1:], axis=1) / (
            np.linalg.norm(embs[:-1], axis=1) * np.linalg.norm(embs[1:], axis=1) + 1e-8
        )
        dists = 1 - sims
        z = (
            (dists - dists.mean()) / (dists.std() + 1e-8)
            if np.std(dists) > 1e-6
            else dists
        )
        peaks = local_peaks_1d(z, thr)
        if not peaks:
            out.append(seg)
            continue

        cuts, last = [], seg.start
        for p in peaks:
            mid = (wins[p][1] + wins[p + 1][0]) / 2.0
            if seg.start + 0.2 <= mid <= seg.end - 0.2:
                cuts.append(mid)
        cuts = sorted(list({round(c, 2) for c in cuts}))
        for c in cuts:
            if c - last >= 0.12:
                out.append(Segment(last, c))
                last = c
        if seg.end - last >= 0.08:
            out.append(Segment(last, seg.end))
    return out


def embed_segments(y: np.ndarray, sr: int, segs: List[Segment]) -> np.ndarray:
    embs = []
    for seg in track(segs, description="[magenta]Embedding segments"):
        a = y[int(seg.start * sr) : int(seg.end * sr)]
        if len(a) < int(0.2 * sr):
            pad = int(0.15 * sr)
            s = max(0, int(seg.start * sr) - pad)
            e = min(len(y), int(seg.end * sr) + pad)
            a = y[s:e]
        embs.append(using_speaker_encoder()(a, sr))
    return np.vstack(embs)


def cluster_by_threshold(embs: np.ndarray, cos_thr: float = 0.6) -> np.ndarray:
    S = cosine_similarity(embs)
    D = 1 - S
    print('emb-cosine', D.shape)

    clu = AgglomerativeClustering(
        n_clusters=None,
        linkage="average",
        metric="precomputed",
        distance_threshold=1 - cos_thr,
    )

    return clu.fit_predict(D)


# 防粘连合并与相似度计算
def conservative_merge(
    y: np.ndarray,
    sr: int,
    segs: list[Segment],
    labels: np.ndarray,
    max_gap: float = 0.10,
    max_turn: float = 10.0,
    min_cos: float = 0.5,
) -> list[Segment]:
    merged: List[Segment] = []
    for i, seg in enumerate(segs):
        seg.spk = int(labels[i])
    segs = sorted(segs, key=lambda s: (s.start, s.end))

    # 计算跨缝相似度时，用 encoder 提取两端嵌入（与聚类一致）
    def seg_cos(a: Segment, b: Segment) -> float:
        aw = y[int(a.start * sr) : int(a.end * sr)]
        bw = y[int(b.start * sr) : int(b.end * sr)]
        ea = using_speaker_encoder()(aw, sr)
        eb = using_speaker_encoder()(bw, sr)
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))

    for seg in track(segs, description="[magenta]Embedding segments"):
        if not merged:
            merged.append(seg)
            continue
        last = merged[-1]
        if (
            (seg.spk == last.spk)
            and (seg.start - last.end <= max_gap)
            and ((seg.end - last.start) <= max_turn)
        ):
            cos = seg_cos(last, seg)
            if cos >= min_cos:
                last.end = seg.end
                last.score = cos
            else:
                merged.append(seg)
        else:
            merged.append(seg)
    return merged


# 帧级重分配（简化版）
def frame_reassign(
    y: np.ndarray,
    sr: int,
    speech_mask: List[Segment],
    segs: List[Segment],
    smooth_step: float = 0.10,
    win: float = 0.50,
) -> List[Segment]:
    if not segs:
        return segs
    spk_ids = sorted(list(set(s.spk for s in segs if s.spk is not None)))
    centroids: dict[int, np.ndarray] = {}
    for sid in spk_ids:
        vecs = []
        for s in segs:
            if s.spk == sid:
                a = y[int(s.start * sr) : int(s.end * sr)]
                if len(a) > int(0.2 * sr):
                    vecs.append(using_speaker_encoder()(a, sr))
        if vecs:
            centroids[sid] = np.mean(np.vstack(vecs), axis=0)

    hop = 0.01
    max_t = max([sm.end for sm in speech_mask]) if speech_mask else len(y) / sr
    n_frames = int(math.ceil(max_t / hop))
    smask = np.zeros(n_frames, dtype=bool)
    for sm in speech_mask:
        s = int(sm.start / hop)
        e = int(sm.end / hop)
        smask[s:e] = True

    print('max-t', max_t)
    times, out_labels, t = [], [], 0.0
    while t < max_t:
        idx = int(t / hop)
        if 0 <= idx < len(smask) and smask[idx]:
            s = max(0.0, t - win / 2)
            e = min(max_t, t + win / 2)
            a = y[int(s * sr) : int(e * sr)]
            if len(a) >= int(win * sr * 0.5):
                emb = using_speaker_encoder()(a, sr)
                best_sid, best_sim = None, -1.0
                for sid, c in centroids.items():
                    sim = float(
                        np.dot(emb, c)
                        / (np.linalg.norm(emb) * np.linalg.norm(c) + 1e-8)
                    )
                    if sim > best_sim:
                        best_sid, best_sim = sid, sim
                out_labels.append(best_sid if best_sid is not None else -1)
            else:
                out_labels.append(-1)
        else:
            out_labels.append(-1)
        times.append(t)
        t += smooth_step

    print('out-labels', len(out_labels))
    refined: List[Segment] = []
    cur_sid, s_t = None, None
    for t, sid in list(zip(times, out_labels)) + [(max_t, -999)]:
        if sid != cur_sid:
            if cur_sid is not None and s_t is not None:
                refined.append(Segment(s_t, t, cur_sid))
            cur_sid, s_t = sid, t

    print('refined', len(refined))
    final: List[Segment] = []
    for r in refined:
        if r.spk is None or r.spk < 0:
            continue
        for sm in speech_mask:
            a = max(r.start, sm.start)
            b = min(r.end, sm.end)
            if b - a >= 0.05:
                final.append(Segment(a, b, r.spk))

    # 合并相邻同人
    final = merge_adjacent(final, gap=0.05)
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
    cluster_cos: float = 0.35,
    merge_gap: float = .5,
    merge_maxturn: float = 10.0,
    merge_mincos: float = 0.37,
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

    labels = cluster_by_threshold(embs, cos_thr=cluster_cos)
    for s, lab in zip(speech2, labels):
        s.spk = int(lab)
    print('cluster', labels)

    # 4) 防粘连合并
    speech3 = conservative_merge(
        y,
        sr,
        speech2,
        labels,
        max_gap=merge_gap,
        max_turn=merge_maxturn,
        min_cos=merge_mincos,
    )

    # 5) 帧级重分配
    if reseg:
        speech4 = frame_reassign(y, sr, speech, speech3, smooth_step=0.10, win=1.0)
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
