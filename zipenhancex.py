import torch
import numpy as np
import math
import torchaudio
from tqdm.rich import tqdm

from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from pathlib import Path

# from functools import lru_cache



@torch.inference_mode()
def zip_enhance_imp(model, wav:torch.Tensor, sr:int = 16000, batch_size:int = 80) :
    assert 16000  == sr, "Only 16000 Hz is supported for now"

    device = 'cuda'
    window_s = 2

    T = wav.size(1)
    L = sr * window_s
    hop = int(round(L * 0.75))

    n_frames, T_pad = _calc_frames_cover_all(T, L, hop)
    if T_pad > T:
        wav = torch.nn.functional.pad(wav, (0, (T_pad - T)))

    # patches: [n_frames, L]
    patches = wav.unfold(dimension=-1, size=L, step=hop).squeeze(0) # ->[n_frames, L]

    W = _sqrt_hann(L, device=device, dtype=torch.float32)  # [L]

    outs = []
    for i in tqdm(range(0, n_frames, batch_size), total=round(n_frames / batch_size)):
        batch = patches[i : i + batch_size].to(device)  # [B, L]
        y_wav = model(dict(noisy=batch))['wav_l2'] # [B, L]
        outs.append(y_wav.cpu())

    y_frames = torch.cat(outs, dim = 0).to(device) # [n_frames, L]
    assert y_frames.dim() == 2 and y_frames.size(1) == L
    y_frames = y_frames * W

    # 目标波形与权重累计
    y = torch.zeros(T_pad, device = device)
    den = torch.zeros(T_pad, device = device)

    # 每帧写回的全局索引：[n_frames, L]
    frame_starts = torch.arange(n_frames, device=device)*hop
    idx = frame_starts.unsqueeze(1) + torch.arange(L, device=device)

    # 叠加数值与权重（与 Fold+den 相同效果）
    y.scatter_add_(0, idx.reshape(-1), y_frames.reshape(-1))
    den.scatter_add_(0, idx.reshape(-1), W.repeat(n_frames))
    # 归一化并裁回原长
    y = y / den.clamp_min(1e-8)
    y = y.unsqueeze(0)[:, :T]

    peak = y.abs().max()
    if torch.isfinite(peak) and float(peak) > 1.0:
        y = y.mul(0.99).div(peak).contiguous()
    return y.cpu(), sr


# def ans_audio_norm(x: torch.Tensor) -> torch.Tensor:
#     # 确保输入是浮点类型，以进行精确计算
#     x = x.float()

#     # 第一步：基于整体 RMS 的标准化
#     rms = torch.mean(x**2) ** 0.5
#     scalar = 10 ** (-25 / 20) / (rms + 1e-8)  # 增加一个小的 epsilon 防止除以零
#     x = x * scalar

#     # 第二步：基于高能量部分的 RMS 的精细调整
#     pow_x = x**2
#     avg_pow_x = torch.mean(pow_x)
#     # 选择功率大于平均功率的部分
#     gt_avg_pow_x = pow_x[pow_x > avg_pow_x]

#     # 如果没有大于平均功率的部分（例如，在静音音频中），则跳过第二步
#     if gt_avg_pow_x.numel() > 0:
#         rmsx = torch.mean(gt_avg_pow_x) ** 0.5
#         scalarx = 10 ** (-25 / 20) / (rmsx + 1e-8)  # 增加一个小的 epsilon 防止除以零
#         x = x * scalarx

#     return x


# def tensor_to_wav_bytes(
#     waveform: torch.Tensor, sample_rate: int, clamp: bool = True
# ) -> bytes:
#     assert waveform.ndim == 2 and waveform.shape[0] == 1, "需要形状 [1, T] 的单声道张量"
#     x = waveform.squeeze(0).detach().cpu()
#     if clamp:
#         x = x.clamp_(-1.0, 1.0)

#     enc = AudioEncoder(samples=x, sample_rate=sample_rate)

#     # 编码为 WAV（可选指定输出声道/采样率；此处保持单声道与原采样率）
#     encoded_u8 = enc.to_tensor(
#         format="wav",
#         num_channels=1,
#         sample_rate=sample_rate,
#     )  # -> torch.uint8, shape [N_bytes]

#     return encoded_u8.cpu().contiguous().numpy().tobytes()


# def pcm16_bytes_to_tensor(pcm: bytes, num_channels=1):
#     dtype = torch.int16
#     arr = torch.frombuffer(pcm, dtype=dtype)
#     # reshape 为 [channels, frames] —— 假设是 interleaved PCM
#     if num_channels > 1:
#         arr = arr.reshape(-1, num_channels).t()  # [channels, samples]
#     else:
#         arr = arr.unsqueeze(0)  # [1, samples]

#     arr = arr.float() / 32768.0
#     return arr




def ans_read_audio(apath: str |Path | tuple[torch.Tensor, int], target_sr: int = 16000)->tuple[torch.Tensor, int]:
    assert target_sr == 16000, "Only 16000 Hz is supported for now"

    if isinstance(apath, str|Path):
        wavform, sr = torchaudio.load_with_torchcodec(str(apath))
    else:
        wavform, sr = apath

    if wavform.dim() == 1:
        wavform = wavform.unsqueeze(0)

    if wavform.size(0) >1:
        wavform = wavform.mean(dim = 0, keepdim=True)

    if sr != target_sr:
        wavform = torchaudio.functional.resample(wavform, orig_freq=sr, new_freq = target_sr)
        sr = target_sr

    peak = wavform.abs().max()
    if peak > 1.0:
        wavform = wavform / peak

    return wavform, sr


def using_zipenhancer(device: str):
    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model="iic/speech_zipenhancer_ans_multiloss_16k_base",
        device=device,
    )
    return ans.model.to(device)


def zip_enhance(
    input: str | Path | tuple[torch.Tensor | np.ndarray, int],
    model,
    batch_size:int = 16,
):
    x, sr = ans_read_audio(input)

    wav, sr = zip_enhance_imp(model, x, sr, batch_size = batch_size)
    return wav, sr


def expand_audios(root:str):
    exts = ['.wav', '.flac', '.ogg', '.opus', '.mp3', '.m4a', '.aac', '.mka']
    proot = Path(root)
    if proot.is_file():
        proot = proot.resolve()
        return [proot.relative_to(proot.parent)], proot.parent

    audios = [p.relative_to(proot) for p in proot.rglob('*.*') if p.is_file() and p.suffix.lower() in exts]
    return audios, proot


def _sqrt_hann(L: int, device="cpu", dtype=torch.float32):
    w = torch.hann_window(L, periodic=False, device=device, dtype=dtype)
    return torch.sqrt(torch.clamp(w, min=0))


def _calc_frames_cover_all(T: int, L: int, hop: int):
    # 覆盖到最后一个样本：ceil 而不是 floor
    if T <= 0:
        return 0, 0
    if T <= L:
        return 1, L  # 只需一帧，pad到L
    # or assert T%L==0
    n = int(math.ceil((T - L) / hop)) + 1
    T_pad = (n - 1) * hop + L
    return n, T_pad



def main(root:str):
    audios, proot = expand_audios(root)
    print(proot, len(audios))
    target_root = proot.with_stem(f'{proot.stem}-zipenhanced')

    ans = using_zipenhancer('cuda')
    for apath in audios:
        tpath = target_root.joinpath(apath).with_suffix('.flac')
        if tpath.exists():
            continue
        tpath.parent.mkdir(parents=True, exist_ok=True)

        wav, sr = zip_enhance(proot/apath, ans)
        torchaudio.save_with_torchcodec(tpath, wav, sr)

if __name__ == '__main__':
    from jsonargparse import auto_cli
    auto_cli(main)
