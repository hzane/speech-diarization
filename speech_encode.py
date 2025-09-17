import torch
import numpy as np
import onnxruntime as ort
from torchaudio.transforms import MelSpectrogram
from speechbrain.inference.classifiers import EncoderClassifier
from functools import lru_cache


def fbank_batch(wavs: np.ndarray, # [B, n_samples]
                sr: int = 16000, n_mels: int = 80, mean_nor: bool = True,)->np.ndarray:
    assert wavs.ndim == 2
    # Kaldi fbank 默认使用dither和移除直流分量
    win_length = int(sr*0.025) #
    hop_length = int(sr*0.010) # from kaldi default options

    mel_spectrogram = MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=win_length, # 通常n_fft等于窗口长度
        win_length=win_length,
        hop_length=hop_length,
        f_min=20.0,
        f_max=sr/2 - 100, # kaldi
        power = 2.0,
    ).to('cuda')
    with torch.inference_mode():
        feat = mel_spectrogram(
            torch.from_numpy(wavs).cuda(),
        )  # [B, n_mels, T]
        # 对数变换，使其成为log-mel-spectrogram (即 fbank)
        feat = torch.log(feat+1e-6)
        feat = feat.transpose(1, 2)

        if mean_nor:
            feat = feat - feat.mean(1, keepdim=True)

    return feat.cpu().numpy()  # [B, T, n_mels]



@lru_cache(maxsize=1)
def using_eres2netv2_encoder():
    # iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
    onnx_path = "models/iic-speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common.onnx"
    session = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    return session


def eres2netv2_encode_batch(wavs: np.ndarray, sr: int = 16000, num_mels:int = 80) -> np.ndarray:
    assert wavs.ndim == 2
    session = using_eres2netv2_encoder()

    features = fbank_batch(wavs, sr = sr, n_mels=num_mels, mean_nor=True)
    y = session.run(None, dict(feature=features))[0]

    return y # [B, 192] #type: ignore



@lru_cache(maxsize=1)
def using_ecapa_encoder(device: str = "cuda")->EncoderClassifier:
    encoder = EncoderClassifier.from_hparams(
        source="LanceaKing/spkrec-ecapa-cnceleb",
        run_opts={"device": device}
    )
    return encoder # type: ignore


def ecapa_encode_batch(wavs: np.ndarray)->np.ndarray:
    encoder = using_ecapa_encoder()
    with torch.inference_mode():
        x = torch.from_numpy(wavs).float()
        y = encoder.encode_batch(x).squeeze(1).cpu().numpy()
    return y # [B, 192]


if __name__ == '__main__':
    import librosa
    apath = 'some.mp3'
    wav, sr = librosa.load(apath, sr = 16000, mono=True)
    batch = frame_audio(wav, sr, win_ms=30.0, hop_ms=10.0)