import numpy as np
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from gtcrn import GTCRN
from rich.progress import track


def load_gtcrn(checkpoint_path: str|Path, device: str='cpu') -> torch.nn.Module:
    """加载 GTCRN 模型并加载权重"""
    model = GTCRN().to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    return model


def gtcrn_read_audio(
    audio: str | Path | tuple[np.ndarray, int], sr: int
) -> tuple[torch.Tensor, int]:
    if isinstance(audio, tuple):
        samples, osr = audio
        samples = torch.from_numpy(samples)
        if not torch.is_floating_point(samples):
            samples = samples.float() / 32768
    else:
        samples, osr = torchaudio.load_with_torchcodec(audio)

    if osr != sr:
        samples = torchaudio.functional.resample(samples, osr, sr)
    if samples.shape[0] > 1:
        samples = samples.mean(dim=0, keepdim=True)
    return samples, sr # [1, T], sample-rate



class AudioEnhancer(nn.Module):
    def __init__(self, model:nn.Module,n_fft: int = 512, hop_length: int = 256, win_length: int = 512):
        super().__init__()
        self.model = model

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # 预先计算窗函数并注册为 buffer，它会自动跟随模型移动到 CPU/GPU
        window = torch.hann_window(self.win_length).pow(0.5)
        self.register_buffer('window', window)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        处理单个 [T] 或批量 [B, T] 的波形。
        """
        # --- 稳健地处理输入维度 ---
        is_batch = waveform.dim() == 2
        if not is_batch:
            waveform = waveform.unsqueeze(0) # [T] -> [1, T]

        device = self.window.device
        waveform = waveform.to(device) # type: ignore
        original_length = waveform.shape[-1]

        # --- STFT ---
        stft_complex = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window, # type: ignore
            center=True,
            return_complex=True,
        )

        # --- 模型推理 ---
        # [B, F, T] (complex) -> [B, F, T, 2] (real)
        stft_real = torch.view_as_real(stft_complex)

        with torch.no_grad():
            enhanced_real = self.model(stft_real) # [B, F, T, 2]

        # --- iSTFT ---
        # [B, F, T, 2] (real) -> [B, F, T] (complex)
        enhanced_complex = torch.view_as_complex(enhanced_real.contiguous())

        enhanced_waveform = torch.istft(
            enhanced_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=original_length,
        )

        # 如果原始输入不是批量，则移除批次维度
        if not is_batch:
            enhanced_waveform = enhanced_waveform.squeeze(0)

        return enhanced_waveform.cpu()

    def enhance_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_size_s: float = 360.0,
        overlap_s: float = 1
    ) -> torch.Tensor:
        """
        使用封装好的 AudioEnhancer 对象，通过循环逐块处理长音频。
        """
        assert chunk_size_s > overlap_s, "Chunk size must be greater than overlap size"

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        num_samples = waveform.shape[-1]
        chunk_size = int(chunk_size_s * sample_rate)

        if num_samples <= chunk_size:
            return self(waveform.squeeze(0))

        stride = int((chunk_size_s - overlap_s) * sample_rate)

        unfold = nn.Unfold(kernel_size=(1, chunk_size), stride=(1, stride))
        fold = nn.Fold(output_size=(1, num_samples), kernel_size=(1, chunk_size), stride=(1, stride))

        window = torch.hann_window(chunk_size, device=waveform.device)

        ones_waveform = torch.ones_like(waveform)
        weight_chunks_unfolded = unfold(ones_waveform.unsqueeze(1))
        normalization_weight = fold(weight_chunks_unfolded)
        normalization_weight = torch.clamp(normalization_weight, min=1e-8)

        waveform_chunks_unfolded = unfold(waveform.unsqueeze(1))
        waveform_chunks = waveform_chunks_unfolded.squeeze(0).transpose(0, 1)
        num_chunks = waveform_chunks.shape[0]

        enhanced_chunks_list = []
        print(f"Processing audio in {num_chunks} chunks...")
        for chunk in track(waveform_chunks, description="Enhancing chunks"):
            # 对象内部处理了所有 STFT, 模型推理, iSTFT 和设备移动
            enhanced_chunk = self(chunk)

            enhanced_chunk *= window.to(enhanced_chunk.device)
            enhanced_chunks_list.append(enhanced_chunk)

        enhanced_chunks = torch.stack(enhanced_chunks_list, dim=0)
        enhanced_chunks_for_fold = enhanced_chunks.transpose(0, 1).unsqueeze(0)

        enhanced_waveform = fold(enhanced_chunks_for_fold)
        enhanced_waveform /= normalization_weight

        return enhanced_waveform.squeeze(0).squeeze(0)


def using_gtcrn(model_path:str|Path|None=None, device:str|int = 0):
    model_path = model_path or Path(__file__).parent/'models.gtcrn/model_trained_on_dns3.tar'

    core = load_gtcrn(model_path)
    enhancer = AudioEnhancer(model=core).to(device)
    return enhancer


if __name__ == '__main__':
    filepath = '/data.d/bilix/bilix/yangli.clearvoiced/yt9zJk0VMYWRk-1821-3_S3_ROCK_ROAST.flac'

    mix, sr = gtcrn_read_audio(filepath, 16000)

    xenhancer = using_gtcrn()
    enh = xenhancer.enhance_audio(mix, sr)
    print(enh.shape)
    # enh = enhance_audio_chunked(enhancer, mix, sr)
    torchaudio.save_with_torchcodec("x.flac", enh, sr)
