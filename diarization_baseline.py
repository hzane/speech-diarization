import torch
import os
import warnings
import torchaudio
from dataclasses import dataclass, field, asdict
from dacite import from_dict, Config as DaciteConfig
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pyannote.audio.pipelines.utils.hook import ProgressHook

from pathlib import Path
from rich.progress import track
from ecapa_annote import ERes2NetV2Encoder, ECAPAEncoder
from functools import lru_cache
from collections import defaultdict
# from zipenhancer_pipe import using_zipenhancer, zip_enhance


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 忽略pyannote的一些警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.*")


@dataclass(frozen=True)
class DiarizationParameters:
    min_stem_s: float = 3.
    max_segment_s: float = 20.0  # 每条输出音轨的最大目标时长
    same_speaker_gap_s: float = 3.0  # 邻接同说话人可合并最大间隔
    # 可合并的两个连续片段之间的最大静音秒数, 触发填充的最小静音间隔
    max_gap_s: float = 1.2
    fade_ms: float = 20.0  # 在每个片段的开头和结尾应用的淡入淡出时长（毫秒）
    min_speech_duration_s: float = 0.2  # 短于该值的语音段会被去除
    min_silence_duration_s: float = 0.1  # 短于该值的静音会被“填平”
    clustering_threshold: float = 0.7  # 余弦相似度阈值（越低越“爱合并”）
    min_speakers: int = 2
    max_speakers: int = 6


def extract_speaker_stems(
    audio: str | Path | dict,
    segments: list[tuple[float, float, str | int]],
    root: str | Path,
    max_segment_s: float,
    max_gap_s: float,
    fade_ms: float,
    min_stem_s: float,
) -> dict[str | int, list[str]]:
    """
    为每个说话人导出合并后的音轨，并确保每条音轨的时长不超过 max_segment_s。
    Args:
        audio (str | Path | dict[waveform, sample_rate]): 原始音频文件路径。
        segments (list): 包含 (start, end, speaker) 的片段列表。
        root (str | Path): 保存输出文件的根目录。
        max_segment_s (float): 每条输出音轨的最大目标时长（秒）。
        max_gap_s (float): 两个连续片段之间的最大静音秒数。
        fade_ms (float): 在每个片段的开头和结尾应用的淡入淡出时长（毫秒）。

    Returns:
        dict[str | int, list[str]]: 一个字典，映射说话人ID到其生成的文件路径列表。
    """
    if isinstance(audio, str|Path):
        audio_path = Path(audio)
        y, sr = torchaudio.load(str(audio))
        num_channels, _ = y.shape
    else:
        y, sr = audio['waveform'], audio['sample_rate']
        num_channels = y.shape[0]
        audio_path = Path(root)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    speaker_segments = defaultdict(list)
    for start, end, speaker in segments:
        speaker_segments[speaker].append((start, end))
    for speaker in speaker_segments:
        speaker_segments[speaker].sort()

    output_files = defaultdict(list)

    for speaker, segs in track(
        speaker_segments.items(), description="正在导出限时音轨..."
    ):

        def _save_track(chunks_to_save: list):
            if not chunks_to_save:
                return
            duration = sum(c.shape[1] for c in chunks_to_save) / sr
            if duration < min_stem_s:
                return

            final_waveform = torch.cat(chunks_to_save, dim=1)
            out_filename = (
                f"{speaker}/{audio_path.stem}-{len(output_files[speaker]):03d}.flac"
            )
            out_path = root / out_filename
            torchaudio.save(
                str(out_path), final_waveform, sr, format="flac", bits_per_sample=16
            )
            output_files[speaker].append(str(out_path.absolute()))

        # 初始化状态变量
        current_track_chunks = []
        current_track_duration_s = 0.0
        last_end_s = 0.0

        fade_samples = int(round(fade_ms / 1000.0 * sr))
        fade_tf = torchaudio.transforms.Fade(
            fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape="linear"
        )

        for i, (start_s, end_s) in enumerate(segs):
            speech_duration_s = end_s - start_s

            # 计算将要添加的静音时长
            silence_duration_s = 0.0
            if i > 0:
                actual_gap_s = start_s - last_end_s
                silence_duration_s = min(actual_gap_s, max_gap_s)

            # 如果当前块非空，并且加上新片段会超长，则先保存当前块
            if (
                current_track_duration_s + silence_duration_s + speech_duration_s
                > max_segment_s
            ):
                _save_track(current_track_chunks)

                # 重置状态，开始新的音轨
                current_track_chunks = []
                current_track_duration_s = 0.0
                # 新的音轨开头不应添加静音
                silence_duration_s = 0.0

            if silence_duration_s > 0:
                silence_samples = int(silence_duration_s * sr)
                silence_tensor = torch.zeros(
                    (num_channels, silence_samples), dtype=y.dtype
                )
                current_track_chunks.append(silence_tensor)
                current_track_duration_s += silence_duration_s

            # 添加语音块
            start_samples = int(start_s * sr)
            end_samples = int(end_s * sr)
            speech_chunk = y[:, start_samples:end_samples]
            if speech_chunk.shape[1] >= fade_samples * 2:
                speech_chunk = fade_tf(speech_chunk)

            current_track_chunks.append(speech_chunk)
            current_track_duration_s += speech_duration_s
            last_end_s = end_s

        # --- 循环结束后，保存最后一个正在构建的音轨 ---
        _save_track(current_track_chunks)

    return dict(output_files)


@lru_cache(maxsize=1)
def using_speaker_diarization_cnceleb(
    device: str | int,
    min_speech_duration_s: float,  # 短于该值的语音段会被去除
    min_silence_duration_s: float,  # 短于该值的静音会被“填平”（避免把同一人说话切得太碎）
    clustering_threshold: float,  # 余弦相似度阈值（越低越“爱合并”）
):
    diarizer = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN"),
    )
    diarizer.embedding = ERes2NetV2Encoder()  # ECAPAEncoder(0)
    diarizer.to(torch.device(device))
    diarizer.segmentation.threshold = 0.5
    diarizer.segmentation.min_duration_on = min_speech_duration_s
    diarizer.segmentation.min_duration_off = min_silence_duration_s

    diarizer.clustering.threshold = clustering_threshold
    return diarizer


# 假设 using_speaker_diarization_cnceleb 和 ProgressHook 已经定义
# from your_module import using_speaker_diarization_cnceleb, ProgressHook


def merge_same_speaker(
    segments: list[tuple[float, float, int | str]],
    max_gap_s: float,
    max_segment_s: float,
):
    if not segments:
        return []

    merged: list[tuple[float, float, int | str]] = [segments[0]]

    for next_start, next_end, next_spk in segments[1:]:
        cur_start, cur_end, cur_spk = merged[-1]
        gap = next_start - cur_end
        if (
            cur_end - cur_start >= max_segment_s
            or next_spk != cur_spk
            or gap > max_gap_s
        ):
            merged.append((next_start, next_end, next_spk))
            continue

        # Merge if overlapping or gap within threshold
        cur_end = max(cur_end, next_end)
        merged[-1] = (cur_start, cur_end, cur_spk)

    return merged


def adjust_segment_boundaries(
    segments, padding: float
) -> list[tuple[float, float, int | str]]:
    if len(segments) < 2:
        return segments

    adjusted = list(segments)
    for i in range(len(adjusted) - 1):
        current_start, current_end, current_spk = adjusted[i]
        next_start, next_end, next_spk = adjusted[i + 1]

        # 静音间隙大于我们设定的阈值
        gap = next_start - current_end
        if gap >= padding:
            adjusted[i] = (current_start, current_end + padding, current_spk)
            adjusted[i + 1] = (max(0, next_start - padding), next_end, next_spk)

    return adjusted


def diarize_audio(
    audio_filepath: str | Path|dict,
    min_speech_duration_s: float,
    min_silence_duration_s: float,
    min_speakers: int,
    max_speakers: int,
    rttm_filepath: str | Path | None = None,
) -> list[tuple[float, float, str | int]]:
    diarize = using_speaker_diarization_cnceleb(
        device=0,
        min_speech_duration_s=min_speech_duration_s,
        min_silence_duration_s=min_silence_duration_s,
        clustering_threshold=0.70,
    )

    with ProgressHook() as hook:
        diarization: Annotation = diarize(
            audio_filepath,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hook=hook,
        )

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):  # type:ignore
        segments.append((turn.start, turn.end, speaker))

    if rttm_filepath:
        with open(rttm_filepath, "w") as f:
            diarization.write_rttm(f)
    return segments


# 假设 diarization 是 pyannote.audio.Annotation 类型的对象
# from pyannote.core import Annotation, Segment, Timeline


def expand_audios(root: Path):
    if root.is_file():
        root = root.resolve()
        return [root], root.parent

    exts = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"]
    audios = [p for p in root.rglob("*.*") if p.is_file() and p.suffix.lower() in exts]
    return audios, root


class Diarizer:
    def __init__(self, hparams: DiarizationParameters):
        self.hparams = hparams
        # self.ans = using_zipenhancer('cuda')

    def diarize(self, apath: str | Path| dict, rttm_filepath: str | Path|None):
        segments = diarize_audio(
            apath,
            self.hparams.min_speech_duration_s,
            min_silence_duration_s=self.hparams.min_silence_duration_s,
            min_speakers=self.hparams.min_speakers,
            max_speakers=self.hparams.max_speakers,
            rttm_filepath=rttm_filepath,
        )
        segments = sorted(segments)
        return segments

    def merge_segments(self, segments: list[tuple[float, float, int | str]]):
        segments = merge_same_speaker(
            segments, self.hparams.max_gap_s, self.hparams.max_segment_s
        )
        return segments

    def pad_segment(self, segments):
        segments = adjust_segment_boundaries(
            segments, padding=self.hparams.fade_ms * 2 / 1000
        )
        return segments

    def extract_speaker(
        self, segments, audio_path: str | Path | dict, root: str | Path
    ) -> dict:
        info = extract_speaker_stems(
            audio_path,
            segments,
            root,
            self.hparams.max_segment_s,
            self.hparams.max_gap_s,
            self.hparams.fade_ms,
            self.hparams.min_stem_s,
        )
        return info

    def __call__(self, audio_path:str|Path, root:str|Path, with_rttm:bool = False):
        rttm_filepath = Path(audio_path).with_suffix(".rttm") if with_rttm else None
        segments = self.diarize(audio_path, rttm_filepath)
        segments = self.merge_segments(segments)
        segments = self.pad_segment(segments)

        # wav, sr = zip_enhance(audio_path, self.ans)
        # audio = dict(waveform=wav, sample_rate=sr)
        info = self.extract_speaker(segments, audio_path, root)
        return segments, info


def main(
    root: str,
    min_speakers: int = 2,
    max_speakers: int = 6,
    max_segment_s: float = 20.0,
    min_silence_duration_s: float = 0.1,
    min_speech_duration_s: float = 0.35,
    same_speaker_gap_s: float = 1.5,
    max_gap_s:float = 3.0,
    fade_ms: float = 30.0,
):
    args = locals()
    args.pop('root')

    hparams = from_dict(
        data_class=DiarizationParameters, data=args, config=DaciteConfig(strict=True)
    )
    diarizer = Diarizer(hparams)
    audios, aroot = expand_audios(Path(root))
    print(aroot, len(audios))

    for apath in track(audios, description="diarization", disable=True):
        troot = root / apath.with_name(f"{apath.stem}-speakers")
        segments, info = diarizer(apath, troot, True)

        print(apath, len(segments))


def mainx(
    root: str,
    min_speakers: int = 1,
    max_speakers: int = 6,
    min_silence_s: float = 0.1,
    max_segment_s: float = 15.0,
    min_speech_s: float = 0.25,
    same_speaker_gap_s: float = 1.5,
    fade_ms: float = 30.0,
):
    audios, aroot = expand_audios(Path(root))
    print(aroot, len(audios))

    for apath in track(audios, description="diarization", disable=True):
        troot = root / apath.with_name(f"{apath.stem}-speakers")
        rttm_filepath = apath.with_suffix(".rttm")

        segments = diarize_audio(
            apath,
            min_speech_duration_s=min_speech_s,
            min_silence_duration_s=min_silence_s,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            rttm_filepath=rttm_filepath,
        )
        segments = sorted(segments)
        segments = merge_same_speaker(
            segments,  # type:ignore
            max_gap_s=same_speaker_gap_s,
            max_segment_s=max_segment_s,
        )

        segments = adjust_segment_boundaries(segments, padding=fade_ms / 1000)

        print(apath, len(segments))
        extract_speaker_stems(
            apath,
            segments,
            troot,
            max_segment_s=max_segment_s,
            max_gap_s=same_speaker_gap_s * 2,
            fade_ms=fade_ms,
            min_stem_s = 2.
        )


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main)

'''
def create_visualization(waveform: np.ndarray, sr: int, diarization: Annotation, vad_segments):
    """创建包含波形、VAD和说话人日记的可视化图表。"""
    fig, ax = plt.subplots(figsize=(15, 4))
    fig.tight_layout(pad=2.0)

    # 1. 绘制音频波形
    time_axis = np.arange(len(waveform)) / sr
    ax.plot(time_axis, waveform, color='gray', alpha=0.7, linewidth=0.5, label="Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Diarization & VAD Visualization")
    ax.grid(True, linestyle='--', alpha=0.5)

    # 2. 绘制VAD（语音活动检测）区域
    # 我们用浅灰色背景条来表示VAD
    for segment in vad_segments.itersegments():
        ax.axvspan(segment.start, segment.end, color='silver', alpha=0.3, ymin=0, ymax=1, label='_nolegend_')

    # 3. 绘制说话人日记结果
    speakers = sorted(diarization.labels()) # type: ignore
    colors = plt.cm.get_cmap('tab10', len(speakers))

    for i, speaker in enumerate(speakers):
        speaker_turns = diarization.label_timeline(speaker)
        for turn in speaker_turns:  # type: ignore
            ax.axvspan(turn.start, turn.end, color=colors(i), alpha=0.6, ymin=0.8, ymax=0.95, label=f'Speaker {speaker}' if turn == speaker_turns[0] else '_nolegend_')
            ax.text((turn.start + turn.end) / 2, 0.5, speaker, ha='center', va='center', fontsize=9, color='white', bbox=dict(facecolor=colors(i), alpha=0.8, boxstyle='round,pad=0.2'))

    ax.legend(loc='upper right')
    ax.set_xlim(0, time_axis[-1])
    return fig
'''


'''
import matplotlib.pyplot as plt
def diarize_audio(
    audio_file_path: str,
    min_speakers: int,
    max_speakers: int,
    progress=gr.Progress(track_tqdm=True)
):
    """
    核心处理函数：加载模型、执行日记、生成输出。
    """
    progress(0, desc="加载模型...")
    pipeline = get_pipeline()

    progress(0.3, desc="正在执行说话人日记...")
    try:
        diarization = pipeline(
            audio_file_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
    except Exception as e:
        raise gr.Error(f"处理音频时出错: {e}")

    speech_activity = diarization.support()


    # --- 准备输出 ---
    progress(0.7, desc="生成结果和可视化图表...")

    # 1. 创建DataFrame
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker
        })
    df = pd.DataFrame(segments)

    # 2. 创建可视化图表
    waveform, sr = librosa.load(audio_file_path, sr=None, mono=True)
    # 归一化以获得更好的绘图效果
    waveform = waveform / np.max(np.abs(waveform))

    viz_plot = create_visualization(waveform, sr, diarization, speech_activity)

    progress(1.0, desc="完成！")
    return viz_plot, df
'''


"""
if __name__ == "__main__":
    import gradio as gr

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                # 输入控件
                audio_input = gr.Audio(sources=["upload"], type="filepath", label="音频输入")
                with gr.Accordion("高级超参数", open=False):
                    min_speakers_slider = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="最少说话人数 (min_speakers)")
                    max_speakers_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="最多说话人数 (max_speakers)")

                submit_button = gr.Button("Diarize Audio", variant="primary")

            with gr.Column(scale=2):
                plot_output = gr.Plot(label="Diarization Timeline")
                dataframe_output = gr.DataFrame(headers=["start", "end", "speaker"], label="Segments")

        # 绑定事件
        submit_button.click(
            fn=diarize_audio,
            inputs=[audio_input, min_speakers_slider, max_speakers_slider],
            outputs=[plot_output, dataframe_output]
        )
    demo.launch(debug=True, share=False)
"""

'''
def extract_speaker_stems(
    audio_path: str | Path,
    segments: list[tuple[float, float, str | int]],
    root: str | Path,
    fade_ms: float = 10.0,
):
    """
    基于分段结果（[{start, end, speaker}, ...]）将每个说话人导出为独立音轨（与原音频同长度）。
    - 提供短交叉淡入淡出以避免段边界咔嗒声。
    """
    y, sr = torchaudio.load(str(audio_path))
    speakers = sorted(list({seg[-1] for seg in segments}))

    _, n = y.shape
    stems = {spk: torch.zeros_like(y) for spk in speakers}  # type: ignore

    fade = int(round(fade_ms / 1000.0 * sr))
    fade_tf = torchaudio.transforms.Fade(
        fade_in_len=fade,
        fade_out_len=fade,
        fade_shape="linear",
    )

    for seg in segments:
        spk = seg[-1]
        s = max(0, int(round(seg[0] * sr)))
        e = min(n, int(round(seg[1] * sr)))
        if e <= s:
            continue
        # 复制原音到对应轨，使用 torchaudio Fade 进行段内淡入/淡出
        chunk = y[:, s:e]
        seg_len = e - s
        if seg_len >= fade:
            # torchaudio expects shape (channels, time)
            chunk = fade_tf(chunk)

        stems[spk][:, s:e] += chunk  # type:ignore

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    out_paths = {}
    for spk, wav in stems.items():
        out_path = root / f"{root.stem}-{spk}.flac"

        torchaudio.save(str(out_path), wav, sr, format="flac", bits_per_sample=16)
        out_paths[spk] = out_path
    return out_paths
'''
