import numpy as np
import numba
import torch
import librosa
from scipy.ndimage import binary_closing, binary_opening
from functools import lru_cache


def frame_audio(y: np.ndarray, sr: int, win_ms: float = 30.0, hop_ms: float = 10.0):
    win = int(round(win_ms / 1000.0 * sr))
    hop = int(round(hop_ms / 1000.0 * sr))

    # 它默认会将数据填充到中心，我们可以通过 center=False 和 pad_mode='constant' 来模拟原始行为
    frames = librosa.util.frame(y, frame_length=win, hop_length=hop)

    return frames.T  # 转置以匹配原始输出形状 [n, win]


class SileroVAD:
    def __init__(self, device: str = "cpu"):
        self.model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            onnx=False,
            skip_validation=True,
        )  # type: ignore
        self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def probs(
        self,
        y: np.ndarray,
        sr: int = 16000,
        win_ms: float = 30.0,
        hop_ms: float = 10.0,
        batch_size: int = 1024,
    ) -> np.ndarray:
        frames = frame_audio(y, sr, win_ms, hop_ms)
        num_frames = frames.shape[0]
        probs = np.zeros(num_frames, dtype=np.float32)

        for i in range(0, num_frames, batch_size):
            batch = frames[i : i + batch_size]
            fb = torch.from_numpy(batch).to(self.device)
            pb = self.model(fb, sr)
            probs[i : i + len(pb)] = pb.cpu().numpy()

        return probs  # 对应 hop_ms 时间步


@lru_cache(maxsize=1)
def using_silero_vad():
    return SileroVAD()



@numba.jit(nopython=True)  # nopython=True 确保生成最高性能的代码
def hysteresis_binarize(
    probs: np.ndarray, on: float = 0.6, off: float = 0.4
) -> np.ndarray:
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


def morph_open_close(
    mask: np.ndarray, hop_ms: float, open_ms: float = 80.0, close_ms: float = 40.0
) -> np.ndarray:
    out = mask.copy()
    if open_ms > 0:
        se_open = np.ones(max(1, int(round(open_ms / hop_ms))), dtype=bool)
        out = binary_opening(out, structure=se_open)
    if close_ms > 0:
        se_close = np.ones(max(1, int(round(close_ms / hop_ms))), dtype=bool)
        out = binary_closing(out, structure=se_close)
    return out


def mask_to_segments(
    mask: np.ndarray,
    hop_ms: float,
    min_speech_ms: float = 250.0,
    min_gap_ms: float = 100.0,
    speech_pad_ms :float = 80.,
) -> list[tuple[float, float]]:
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
    speech_pad_frames = round(speech_pad_ms / hop_ms)
    total_frames = len(mask)

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

    final_segments = []
    for s, e in merged_segments:
        s_padded = max(s - speech_pad_frames, 0)
        e_padded = min(e + speech_pad_frames, total_frames)
        s_time = round(s_padded*hop_s, 3)
        e_time = round(e_padded*hop_s, 3)
        final_segments.append((s_time, e_time))

    return final_segments



def silero_vad_segments(
        y:np.ndarray,
        sr:int = 16000,
    on_threshold: float = 0.6,
    off_threshold: float = 0.4,
    min_speech_ms: float = 250.,
    min_silence_ms: float = 100.,
    speech_pad_ms: float = 40,
    win_ms: float = 30.0,
    hop_ms: float = 10.0,
    morph_open_ms: float = 80.0,
    morph_close_ms: float = 40.0,
    batch_size: int = 512,
):
    vad = using_silero_vad()
    probs = vad.probs(y, sr, win_ms=win_ms, hop_ms = hop_ms, batch_size = batch_size)
    mask = hysteresis_binarize(probs, on = on_threshold, off = off_threshold)
    mask = morph_open_close(mask, hop_ms, open_ms = morph_open_ms, close_ms = morph_close_ms)
    segments = mask_to_segments(mask, hop_ms, min_speech_ms = min_speech_ms, min_gap_ms = min_silence_ms, speech_pad_ms = speech_pad_ms)

    return segments