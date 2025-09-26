import torch
import torchaudio
import numpy as np
from modelscope.pipelines.audio.ans_pipeline import ANSZipEnhancerPipeline
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pathlib import Path

# from functools import lru_cache
from torchcodec.encoders import AudioEncoder


def ans_audio_norm(x: torch.Tensor) -> torch.Tensor:
    # 确保输入是浮点类型，以进行精确计算
    x = x.float()

    # 第一步：基于整体 RMS 的标准化
    rms = torch.mean(x**2) ** 0.5
    scalar = 10 ** (-25 / 20) / (rms + 1e-8)  # 增加一个小的 epsilon 防止除以零
    x = x * scalar

    # 第二步：基于高能量部分的 RMS 的精细调整
    pow_x = x**2
    avg_pow_x = torch.mean(pow_x)
    # 选择功率大于平均功率的部分
    gt_avg_pow_x = pow_x[pow_x > avg_pow_x]

    # 如果没有大于平均功率的部分（例如，在静音音频中），则跳过第二步
    if gt_avg_pow_x.numel() > 0:
        rmsx = torch.mean(gt_avg_pow_x) ** 0.5
        scalarx = 10 ** (-25 / 20) / (rmsx + 1e-8)  # 增加一个小的 epsilon 防止除以零
        x = x * scalarx

    return x


def tensor_to_wav_bytes(
    waveform: torch.Tensor, sample_rate: int, clamp: bool = True
) -> bytes:
    assert waveform.ndim == 2 and waveform.shape[0] == 1, "需要形状 [1, T] 的单声道张量"
    x = waveform.squeeze(0).detach().cpu()
    if clamp:
        x = x.clamp_(-1.0, 1.0)

    enc = AudioEncoder(samples=x, sample_rate=sample_rate)

    # 编码为 WAV（可选指定输出声道/采样率；此处保持单声道与原采样率）
    encoded_u8 = enc.to_tensor(
        format="wav",
        num_channels=1,
        sample_rate=sample_rate,
    )  # -> torch.uint8, shape [N_bytes]

    return encoded_u8.cpu().contiguous().numpy().tobytes()


def pcm16_bytes_to_tensor(pcm: bytes, num_channels=1):
    dtype = torch.int16
    arr = torch.frombuffer(pcm, dtype=dtype)
    # reshape 为 [channels, frames] —— 假设是 interleaved PCM
    if num_channels > 1:
        arr = arr.reshape(-1, num_channels).t()  # [channels, samples]
    else:
        arr = arr.unsqueeze(0)  # [1, samples]

    arr = arr.float() / 32768.0
    return arr


def normalize_loudness(
    wav: torch.Tensor,
    sample_rate: int,
    loudness_headroom_db: float = -16,
    energy_floor: float = 2e-3,
):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.
    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        output (torch.Tensor): Loudness normalized output data.
    """
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    input_loudness_db = torchaudio.functional.loudness(wav, sample_rate)
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output


def ans_read_audio(apath: str |Path | tuple[torch.Tensor|np.ndarray, int])->str|bytes:
    if isinstance(apath, str|Path):
        return str(apath)
    wavform, sr = apath
    if isinstance(wavform, np.ndarray):
        wavform = torch.from_numpy(wavform)
    if wavform.ndim == 1:
        wavform = wavform.unsqueeze(0)

    if wavform.shape[0] > 1:
        wavform = wavform.mean(dim=0, keepdim=True)

    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wavform = resampler(wavform)
        sr = target_sr

    return tensor_to_wav_bytes(wavform, sr, clamp=False)


def using_zipenhancer(device: str):
    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model="iic/speech_zipenhancer_ans_multiloss_16k_base",
        device=device,
    )
    return ans  # , ans.SAMPLE_RATE


def zip_enhance(
    input: str | Path | tuple[torch.Tensor | np.ndarray, int],
    ans: ANSZipEnhancerPipeline,
):
    x = ans_read_audio(input)

    pcm = ans(x)["output_pcm"]
    wav = torch.frombuffer(pcm, dtype=torch.int16).clone().unsqueeze(0)
    wav = wav.float() / 32768
    return wav, 16000


def expand_audios(root:str):
    exts = ['.wav', '.flac', '.ogg', '.opus', '.mp3', '.m4a', '.aac', '.mka']
    proot = Path(root)
    if proot.is_file():
        proot = proot.resolve()
        return [proot.relative_to(proot.parent)], proot.parent

    audios = [p.relative_to(proot) for p in proot.rglob('*.*') if p.is_file() and p.suffix.lower() in exts]
    return audios, proot


def main(root:str):
    audios, proot = expand_audios(root)
    print(proot, len(audios))

    ans = using_zipenhancer('cuda')
    for apath in audios:
        wav, sr = zip_enhance(proot/apath, ans)
        tpath = proot.joinpath(apath).with_stem(f'{apath.stem}-ze').with_suffix('.flac')
        torchaudio.save_with_torchcodec(tpath, wav, sr, format='flac', bits_per_sample=16)

if __name__ == '__main__':
    from jsonargparse import auto_cli
    auto_cli(main)