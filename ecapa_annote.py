import torch
from pyannote.audio.core.model import Model
from speech_encode import using_ecapa_encoder, using_eres2netv2_encoder, eres2netv2_encode_batch


class ECAPAEncoder(Model):
    def __init__(self, device:str|int = 0):
        super().__init__()
        self.model = using_ecapa_encoder(device)

        self.dimension = 192 # self.model.hparams.embedding_dim

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        参数:
            waveforms (torch.Tensor): 形状为 (batch_size, num_samples) 的波形张量。

        返回:
            torch.Tensor: 形状为 (batch_size, self.dimension) 的嵌入张量。
        """
        # 我们需要用 squeeze(1) 去掉中间多余的维度以符合 pyannote 的要求。
        return self.model.encode_batch(waveforms).squeeze(1)


class ERes2NetV2Encoder(Model):
    def __init__(self, device:str|int = 0):
        super().__init__()
        self.model = using_eres2netv2_encoder()
        self.dimension = 192 # self.model.hparams.embedding_dim

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        y = eres2netv2_encode_batch(waveforms.cpu().numpy())
        return torch.from_numpy(y).to(waveforms.device)
