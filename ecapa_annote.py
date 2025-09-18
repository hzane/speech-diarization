import torch
from pyannote.audio.core.model import Model
from speechbrain.inference.classifiers import EncoderClassifier

class SpeechBrainECAPA(Model):
    def __init__(self, device:str = 'cuda'):
        super().__init__()

        self.model = EncoderClassifier.from_hparams(
            source = "LanceaKing/spkrec-ecapa-cnceleb",
            run_opts={"device": device}
        )

        self.dimension = 192 # self.model.hparams.embedding_dim

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        参数:
            waveforms (torch.Tensor): 形状为 (batch_size, num_samples) 的波形张量。

        返回:
            torch.Tensor: 形状为 (batch_size, self.dimension) 的嵌入张量。
        """
        # 使用 SpeechBrain 模型的 encode_batch 方法进行编码
        # SpeechBrain 返回的形状是 (batch_size, 1, dimension)，
        # 我们需要用 squeeze(1) 去掉中间多余的维度以符合 pyannote 的要求。
        return self.model.encode_batch(waveforms).squeeze(1)