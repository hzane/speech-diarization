#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, math, json, warnings, pathlib, csv
from typing import List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln

import torch
from scipy.ndimage import binary_opening, binary_closing
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import typer
from rich import print
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================
# I/O & 预处理
# =========================
def load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, s = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if s != sr:
        y = librosa.resample(y, orig_sr=s, target_sr=sr, res_type="kaiser_fast")
        s = sr
    y = y - np.mean(y)
    y = librosa.effects.preemphasis(y, coef=0.97)
    return y.astype(np.float32), s

def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float = -18.0):
    try:
        meter = pyln.Meter(sr)
        l = meter.integrated_loudness(y)
        y = pyln.normalize.loudness(y, l, target_lufs)
    except Exception:
        pass
    return np.clip(y, -0.99, 0.99)

# =========================
# Silero：VAD 概率（原始前向）
# =========================
def frame_audio(y: np.ndarray, sr: int, win_ms: float = 30.0, hop_ms: float = 10.0):
    win = int(round(win_ms/1000.0 * sr))
    hop = int(round(hop_ms/1000.0 * sr))
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = y[idx]  # [n, win]
    return frames, hop

@torch.no_grad()
def silero_probs_raw(y: np.ndarray, sr: int = 16000,
                     win_ms: float = 30.0, hop_ms: float = 10.0,
                     batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    model = torch.hub.load('snakers4/silero-vad', 'silero_vad',
                           force_reload=False, onnx=False).to(device).eval()
    frames, hop = frame_audio(y, sr, win_ms, hop_ms)
    N = frames.shape[0]
    probs = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = min(i + batch_size, N)
        fb = torch.from_numpy(frames[i:j]).to(device)
        pb = model(fb, sr)  # [B] 概率
        probs[i:j] = pb.detach().float().cpu().numpy()
        i = j
    return probs  # 对应 hop_ms 时间步

def hysteresis_binarize(probs: np.ndarray, on: float = 0.6, off: float = 0.4) -> np.ndarray:
    mask = np.zeros_like(probs, dtype=bool)
    talking = False
    for i, p in enumerate(probs):
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
                     min_speech_ms: float = 250.0, min_gap_ms: float = 100.0):
    segs = []
    hop_s = hop_ms / 1000.0
    in_seg, s = False, 0
    arr = mask.tolist() + [False]
    for i, v in enumerate(arr):
        if v and not in_seg:
            in_seg, s = True, i
        elif not v and in_seg:
            in_seg = False
            st, ed = s*hop_s, i*hop_s
            if (ed - st)*1000.0 >= min_speech_ms:
                segs.append([st, ed])
    # 合并相邻小缝
    merged = []
    for st, ed in segs:
        if not merged:
            merged.append([st, ed]); continue
        lst, led = merged[-1]
        if (st - led)*1000.0 <= min_gap_ms:
            merged[-1][1] = ed
        else:
            merged.append([st, ed])
    return merged

# =========================
# 说话人嵌入（ECAPA / 3D-Speaker 可选）
# =========================
class SpeakerEncoder:
    def __init__(self, backend="speechbrain-ecapa", device="cpu", ali_model_id: Optional[str]=None):
        self.backend = backend
        self.device = device
        self.sr = 16000

        if backend == "speechbrain-ecapa":
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device}
            )
            self.kind = "sb"
        elif backend in ("ali-eres2netv2", "ali-campp"):
            from modelscope.hub.snapshot_download import snapshot_download
            import onnxruntime as ort
            mid = ali_model_id or ("iic/speech_eres2netv2_sv_zh-cn_16k-common" if backend=="ali-eres2netv2"
                                   else "iic/speech_campplus_sv_zh-cn_16k-common")
            model_dir = snapshot_download(mid)
            onnx_path = None
            for r,_,fs in os.walk(model_dir):
                for f in fs:
                    if f.endswith(".onnx"):
                        onnx_path = os.path.join(r,f); break
                if onnx_path: break
            if onnx_path is None:
                raise RuntimeError("未找到 .onnx；请检查 modelscope 模型或改用 speechbrain-ecapa")
            self._ort = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._in = self._ort.get_inputs()[0].name
            self._out = self._ort.get_outputs()[0].name
            self.kind = "ali"
        else:
            raise ValueError("backend 必须是 speechbrain-ecapa / ali-eres2netv2 / ali-campp")

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr, res_type="kaiser_fast")
            sr = self.sr
        if self.kind == "sb":
            import torch
            wav = torch.from_numpy(y).float().unsqueeze(0)
            with torch.no_grad():
                e = self.model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
            return e.astype(np.float32)
        elif self.kind == "ali":
            x = y.astype(np.float32)
            if len(self._ort.get_inputs()[0].shape) == 2:  # [B,T]
                x = x[None, :]
            out = self._ort.run([self._out], {self._in: x})[0]
            if out.ndim == 2: out = out[0]
            return out.astype(np.float32)

# =========================
# 嵌入常用处理
# =========================
def pad_with_context(y: np.ndarray, sr: int, start: float, end: float, ctx: float=0.2):
    s = max(0, int((start-ctx)*sr))
    e = min(len(y), int((end+ctx)*sr))
    return y[s:e]

def whiten_l2(embs: np.ndarray) -> np.ndarray:
    X = embs - embs.mean(0, keepdims=True)
    C = np.cov(X.T)
    U,S,VT = np.linalg.svd(C, full_matrices=False)
    W = U @ np.diag(1.0/np.sqrt(S+1e-6)) @ U.T
    Xw = X @ W
    Xw /= (np.linalg.norm(Xw, axis=1, keepdims=True)+1e-9)
    return Xw

def asnorm_scores(query_embs: np.ndarray, ref_centers: np.ndarray,
                  cohort_embs: np.ndarray, topk: int = 200) -> np.ndarray:
    def l2n(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)
    Q = l2n(query_embs); R = l2n(ref_centers); C = l2n(cohort_embs)
    raw = Q @ R.T
    qc = np.sort(Q @ C.T, axis=1)[:, -min(topk, C.shape[0]):]
    rc = np.sort(R @ C.T, axis=1)[:, -min(topk, C.shape[0]):]
    q_mu, q_sigma = qc.mean(axis=1, keepdims=True), qc.std(axis=1, keepdims=True)+1e-6
    r_mu, r_sigma = rc.mean(axis=1, keepdims=True).T, rc.std(axis=1, keepdims=True).T+1e-6
    zn_q = (raw - q_mu) / q_sigma
    zn_r = (raw - r_mu) / r_sigma
    score = 0.5*(zn_q + zn_r)
    return score

# =========================
# 聚类 & VBx-like 重分割
# =========================
def cluster_embeddings(embs: np.ndarray, method="hdbscan", cos_thr: float=0.68):
    if method == "hdbscan":
        D = 1 - cosine_similarity(embs)
        clu = hdbscan.HDBSCAN(min_cluster_size=6, min_samples=3, metric='precomputed')
        labels = clu.fit_predict(D)
    elif method == "agglo":
        D = 1 - cosine_similarity(embs)
        try:
            clu = AgglomerativeClustering(n_clusters=None, linkage="average",
                                          metric="precomputed", distance_threshold=1-cos_thr)
        except TypeError:
            clu = AgglomerativeClustering(n_clusters=None, linkage="average",
                                          affinity="precomputed", distance_threshold=1-cos_thr)
        labels = clu.fit_predict(D)
    else:
        raise ValueError("method 必须是 hdbscan 或 agglo")
    return labels

def viterbi_hmm(scores: np.ndarray, alpha: float=0.995) -> np.ndarray:
    T, K = scores.shape
    eps = 1e-8
    logA = np.full((K, K), np.log((1-alpha)/(K-1)+eps), dtype=np.float32)
    for k in range(K): logA[k,k] = np.log(alpha+eps)
    dp = np.full((T,K), -1e9, dtype=np.float32)
    ptr = np.zeros((T,K), dtype=np.int32)
    dp[0] = scores[0]
    for t in range(1, T):
        prev = dp[t-1][:,None] + logA
        ptr[t] = np.argmax(prev, axis=0)
        dp[t]  = prev[ptr[t], np.arange(K)] + scores[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T-2, -1, -1):
        path[t] = ptr[t+1, path[t+1]]
    return path

# =========================
# 导出与可视化
# =========================
def save_json(out_path: str, segments: List[Dict], speakers: List[str]):
    data = {"segments": segments, "speakers": speakers}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_srt(out_path: str, segments: List[Dict]):
    def fmt(ts: float) -> str:
        h = int(ts // 3600); ts -= h*3600
        m = int(ts // 60);   ts -= m*60
        s = int(ts); ms = int(round((ts - s)*1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{seg['speaker']}\n\n")

def save_csv(out_path: str, segments: List[Dict]):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["start","end","speaker"])
        w.writeheader()
        for seg in segments:
            w.writerow({"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"]})

def plot_diagnostics(out_dir: str, seg_embs: np.ndarray, labels: np.ndarray,
                     adj_sims: np.ndarray, nonadj_sims: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    # 相似度矩阵
    S = cosine_similarity(seg_embs)
    plt.figure(figsize=(6,5))
    plt.imshow(S, vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(); plt.title("Cosine similarity between segments")
    plt.xlabel("segment"); plt.ylabel("segment")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"sim_matrix.png"), dpi=150); plt.close()

    # 直方图
    plt.figure(figsize=(6,4))
    plt.hist(adj_sims, bins=60, range=(-1,1), alpha=0.6, label="adjacent")
    plt.hist(nonadj_sims, bins=60, range=(-1,1), alpha=0.6, label="non-adjacent")
    plt.legend(); plt.title("Similarity distributions")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"sim_hists.png"), dpi=150); plt.close()

# =========================
# 主流程
# =========================
app = typer.Typer(add_completion=False)

@app.command()
def main(
    audio_path: str = typer.Argument(..., help="音频路径（wav/mp3/flac/m4a/ogg 等）"),
    out_dir: str = typer.Option("out", help="输出目录"),
    device: str = typer.Option("cpu", help="cpu/cuda"),
    # 编码器
    backend: str = typer.Option("speechbrain-ecapa", help="speechbrain-ecapa / ali-eres2netv2 / ali-campp"),
    ali_model_id: Optional[str] = typer.Option(None, help="自定义阿里 modelscope 模型ID"),
    # VAD
    vad_on: float = 0.6, vad_off: float = 0.4,
    win_ms: float = 30.0, hop_ms: float = 10.0,
    open_ms: float = 80.0, close_ms: float = 40.0,
    min_speech_ms: float = 250.0, min_gap_ms: float = 100.0,
    # 嵌入与聚类
    pad_ctx: float = 0.2,
    whiten: int = 1,
    asnorm: int = 1,
    cluster: str = "hdbscan",   # 或 "agglo"
    cos_thr: float = 0.68,      # 仅 agglo 用
    # VBx-like
    use_vbx: int = 1,
    alpha: float = 0.995,
    # 可视化
    save_plots: int = 1
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[bold]Audio:[/bold] {audio_path}")

    # 读入 & 归一
    y, sr = load_audio(audio_path, sr=16000)
    y = loudness_normalize(y, sr)

    # VAD
    probs = silero_probs_raw(y, sr, win_ms, hop_ms, device=device)
    mask0 = hysteresis_binarize(probs, on=vad_on, off=vad_off)
    mask  = morph_open_close(mask0, hop_ms, open_ms=open_ms, close_ms=close_ms)
    segs  = mask_to_segments(mask, hop_ms, min_speech_ms=min_speech_ms, min_gap_ms=min_gap_ms)

    if len(segs) == 0:
        print("[red]未检测到语音片段，请调整 VAD/形态学参数。[/red]")
        raise SystemExit(1)
    print(f"[green]Segments:[/green] {len(segs)}")

    # 嵌入
    enc = SpeakerEncoder(backend=backend, device=device, ali_model_id=ali_model_id)
    emb_list = []
    for st, ed in segs:
        a = pad_with_context(y, sr, st, ed, ctx=pad_ctx)
        if len(a) < int(0.4*sr):
            # 对极短片段复制扩展
            rep = int(math.ceil(0.4*sr/len(a)))
            a = np.tile(a, rep)[:int(0.4*sr)]
        e = enc.embed(a, sr)
        emb_list.append(e)
    embs = np.vstack(emb_list)
    if whiten: embs = whiten_l2(embs)

    # 诊断：相邻 vs 非邻接
    S = cosine_similarity(embs)
    adj_sims = []
    for i in range(len(segs)-1):
        adj_sims.append(S[i, i+1])
    adj_sims = np.array(adj_sims) if adj_sims else np.array([0.0])

    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(segs), size=min(2000, len(segs)*4))
    idys = rng.integers(0, len(segs), size=min(2000, len(segs)*4))
    nonadj = [S[i,j] for i,j in zip(idxs, idys) if abs(i-j) > 3]
    nonadj_sims = np.array(nonadj) if nonadj else np.array([0.0])

    # 聚类
    if cluster == "hdbscan":
        labels = cluster_embeddings(embs, method="hdbscan")
    else:
        labels = cluster_embeddings(embs, method="agglo", cos_thr=cos_thr)

    # 类中心
    uniq = sorted([u for u in np.unique(labels) if u != -1])
    if not uniq:
        # 若全是噪声，强行用所有段做一个类中心，避免后续崩
        uniq = [0]; labels = np.zeros(len(labels), dtype=int)
    centers = []
    for k in uniq:
        m = embs[labels==k].mean(0)
        m /= (np.linalg.norm(m)+1e-9)
        centers.append(m)
    centers = np.vstack(centers)  # [K, D]

    # AS-Norm 分数（段→中心）
    scores = embs @ centers.T
    if asnorm:
        # cohort：同条音频的全部段嵌入
        scores = asnorm_scores(embs, centers, embs, topk=min(200, len(embs)))

    # VBx-like（HMM+Viterbi）：把聚类中心当“说话人”
    if use_vbx:
        path = viterbi_hmm(scores, alpha=alpha)  # [N段] 这里按“段”为步长，足够稳；需要更细可以改成滑窗帧级
        final_labels = path
    else:
        final_labels = np.argmax(scores, axis=1)

    # 片段合并（同说话人、间隙≤0.10s）
    def merge_segments(segs, labs, gap=0.10):
        out = []
        cur_lab, s, e = labs[0], segs[0][0], segs[0][1]
        for (st, ed), lb in zip(segs[1:], labs[1:]):
            if lb == cur_lab and st - e <= gap:
                e = ed
            else:
                out.append([s, e, int(cur_lab)])
                cur_lab, s, e = lb, st, ed
        out.append([s, e, int(cur_lab)])
        return out

    merged = merge_segments(segs, final_labels, gap=min_gap_ms/1000.0)

    # 导出
    spk_map = {k: f"SPK_{i}" for i,k in enumerate(uniq)}  # HDBSCAN 可能跳号，映射到 0..K-1
    segments = [{"start": round(s,3), "end": round(e,3), "speaker": spk_map.get(lb, f"SPK_{lb}")} for s,e,lb in merged]
    speakers = sorted(list(set(seg["speaker"] for seg in segments)))
    json_path = os.path.join(out_dir, "diarization.json")
    srt_path  = os.path.join(out_dir, "diarization.srt")
    csv_path  = os.path.join(out_dir, "diarization.csv")
    save_json(json_path, segments, speakers)
    save_srt(srt_path, segments)
    save_csv(csv_path, segments)
    print(f"[blue]Saved[/blue] JSON→ {json_path} | SRT→ {srt_path} | CSV→ {csv_path}")
    print(f"[green]Speakers detected:[/green] {speakers}")

    if save_plots:
        plot_diagnostics(out_dir, embs, labels, adj_sims, nonadj_sims)
        print(f"[blue]Saved plots[/blue] → {os.path.join(out_dir,'sim_matrix.png')} / sim_hists.png")
    # 简要报告
    def stats(x): return (float(np.mean(x)), float(np.std(x)))
    m1,s1 = stats(adj_sims); m2,s2 = stats(nonadj_sims)
    print(f"[bold]adjacent cos[/bold] μ={m1:.3f} σ={s1:.3f}   |   [bold]non-adj cos[/bold] μ={m2:.3f} σ={s2:.3f}")
    print("[dim]提示：若两分布重叠大→考虑提高 open_ms / 更强嵌入 / 开启 asnorm / 用 HDBSCAN。[/dim]")

if __name__ == "__main__":
    main()
