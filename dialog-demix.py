import torch
import torchaudio
from tqdm.rich import tqdm
from demucs.states import load_model
from demucs.apply import apply_model
from pathlib import Path


def download_models():
    model_list = [
        "97d170e1-a778de4a.th",
        "97d170e1-dbb4db15.th",
        "97d170e1-e41a5468.th",
    ]
    root = Path(__file__).parent.resolve() / "models"
    root.mkdir(parents=True, exist_ok=True)

    outs = []
    for model_name in model_list:
        model_path = root / model_name
        if not model_path.exists():
            remote_url = (
                "https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/"
                + model_name
            )
            torch.hub.download_url_to_file(remote_url, model_path)
        outs.append(str(model_path))
    return outs


class Demucs4DialogEffectMusic:
    def __init__(self, device):
        self.device = device

        self.model_list = [
            "97d170e1-a778de4a.th",
            "97d170e1-dbb4db15.th",
            "97d170e1-e41a5468.th",
        ]

        self.models = []
        models = download_models()
        for model_path in models:
            model = load_model(model_path)
            model.to(device)
            self.models.append(model)

    @property
    def instruments(self):
        return ["music", "effect", "dialog"]

    def separate(
        self,
        wav: torch.Tensor,  # [2, T]
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: shape [3, 2, T], [music, effect, dialog]
        """
        assert wav.dim() == 2 and wav.size(0) == 2, "input wav must be [2, T]"
        assert sample_rate == 44100, "sample rate must be 44100"

        wav = wav.unsqueeze(0).to(self.device)

        outs = []
        for model in self.models:
            out = apply_model(model, wav, shifts=1, overlap=0.8)[0].cpu()
            outs.append(out)
        dnr_demucs = torch.stack(outs).mean(dim=0)

        return dnr_demucs


def demucs_read_audio(audio: str | tuple[torch.Tensor, int]):
    if isinstance(audio, tuple):
        wav, sr = audio
    else:
        wav, sr = torchaudio.load_with_torchcodec(audio)
    if sr != 44100:
        wav = torchaudio.resample(wav, sr, 44100)
        sr = 44100
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)
    if wav.size(0) > 2:
        wav = wav[:2]  # only first two channels

    return wav, sr


def expand_audios(input: str):
    root = Path(input)
    if root.is_file():
        root = root.resolve()
        return [str(root)], root.parent
    exts = [".wav", ".flac", ".mp3", ".webm", ".m4a", ".opus", ".ogg"]
    audios = [
        str(f.relative_to(root))
        for f in root.rglob("*.*")
        if f.is_file() and f.suffix.lower() in exts
    ]
    return audios, root


def separate_dialog(input: str, device: str | int, output: str | None = None):
    audios, root = expand_audios(input)
    troot = Path(output) if output else root.with_stem(f"{root.stem}-dialog")

    model = Demucs4DialogEffectMusic(device)

    for audiopath in tqdm(audios):
        audio, sr = demucs_read_audio(str(root / audiopath))
        stems = model.separate(audio, sr)
        for instrum, stem in zip(model.instruments, stems):
            tpath = (troot / instrum / audiopath).with_suffix(".flac")
            tpath.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save_with_torchcodec(str(tpath), stem, sr)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(separate_dialog)
