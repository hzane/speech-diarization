import torch
import os
import warnings
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pyannote.audio.pipelines.utils.hook import ProgressHook

from pathlib import Path
from rich.progress import track
from ecapa_annote import ERes2NetV2Encoder, ECAPAEncoder
from functools import lru_cache
from collections import defaultdict


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 忽略pyannote的一些警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.*")


def extract_speaker_stems_with_silence_control(
    audio_path: str | Path,
    segments: list[tuple[float, float, str | int]],
    root: str | Path,
    max_silence_s: float = 2.5,  # 新增：分段之间的最大静音秒数
    fade_ms: float = 20.0,  # 稍长的淡入淡出以平滑拼接
):
    """
    为每个说话人导出独立的、合并后的音轨，并控制片段间的最大静音时长。

    Args:
        audio_path (str | Path): 原始音频文件路径。
        segments (list): 包含 (start, end, speaker) 的片段列表。
        root (str | Path): 保存输出文件的根目录。
        max_silence_s (float): 两个连续片段之间的最大静音秒数。
                               如果原始静音更短，则保留原始静音。
        fade_ms (float): 在每个片段的开头和结尾应用的淡入淡出时长（毫秒）。
    """
    audio_path = Path(audio_path)
    y, sr = torchaudio.load(str(audio_path))
    num_channels, num_samples = y.shape

    speaker_segments = defaultdict(list)
    for start, end, speaker in segments:
        speaker_segments[speaker].append((start, end))

    for speaker in speaker_segments:
        speaker_segments[speaker].sort()

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out_paths = {}

    # 为每个说话人独立处理并生成新音轨
    for speaker, segs in track(
        speaker_segments.items(), description="正在导出说话人音轨..."
    ):
        final_chunks = []

        fade_samples = int(round(fade_ms / 1000.0 * sr))
        fade_tf = torchaudio.transforms.Fade(
            fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape="linear"
        )

        last_end_s = 0.0

        for i, (start_s, end_s) in enumerate(segs):
            # 添加静音
            if i > 0:
                actual_gap_s = start_s - last_end_s
                silence_duration_s = min(actual_gap_s, max_silence_s)

                if silence_duration_s > 0:
                    silence_samples = int(silence_duration_s * sr)
                    silence_tensor = torch.zeros(
                        (num_channels, silence_samples), dtype=y.dtype
                    )
                    final_chunks.append(silence_tensor)

            # 提取并处理语音片段
            start_samples = int(start_s * sr)
            end_samples = int(end_s * sr)
            speech_chunk = y[:, start_samples:end_samples]

            if speech_chunk.shape[1] >= fade_samples * 2:
                speech_chunk = fade_tf(speech_chunk)

            final_chunks.append(speech_chunk)

            last_end_s = end_s

        if not final_chunks:
            continue  # 如果该说话人没有任何有效片段，则跳过

        final_waveform = torch.cat(final_chunks, dim=1)

        out_path = root / f"{audio_path.stem}-{speaker}.flac"

        torchaudio.save(
            str(out_path), final_waveform, sr, format="flac", bits_per_sample=16
        )
        out_paths[speaker] = out_path

    return out_paths


@lru_cache(maxsize=1)
def using_speaker_diarization_cnceleb(
    device: str | int = 0,
    min_speech_duration_s: float = 0.2,
    min_silence_duration_s: float = 0.1,
    clustering_threshold: float = 0.65,
):
    diarizer = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN"),
    )
    diarizer.embedding = ERes2NetV2Encoder()  # ECAPAEncoder(0)
    diarizer.to(torch.device(device))
    diarizer.segmentation.threshold = 0.5
    diarizer.segmentation.min_duration_on = (
        min_speech_duration_s  # 短于该值的语音段会被去除
    )
    diarizer.segmentation.min_duration_off = (
        min_silence_duration_s  # 短于该值的静音会被“填平”（避免把同一人说话切得太碎）
    )
    diarizer.clustering.threshold = (
        clustering_threshold  # 余弦相似度阈值（越低越“爱合并”）
    )
    return diarizer


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


def diarize_audio(
    audio_filepath: str,
    min_speakers: int = 2,
    max_speakers: int = 6,
    rttm_filepath: str | None = None,
) -> list[tuple[float, float, str | int]]:
    diarize = using_speaker_diarization_cnceleb(
        device=0,
        min_speech_duration_s=0.25,
        min_silence_duration_s=0.1,
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


def main(
    root: str,
    min_speakers: int = 2,
    max_speakers: int = 6,
    max_silence_s: float = 1.5,
    fade_ms: float = 10.0,
):
    audios, aroot = expand_audios(Path(root))
    print(aroot, len(audios))

    troot = aroot.with_stem(f"{aroot.stem}-speakers")

    for apath in track(audios, description="diarization", disable=True):
        segments = diarize_audio(
            str(apath),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            rttm_filepath=str(apath.with_suffix(".rttm")),
        )
        print(apath, len(segments))
        extract_speaker_stems_with_silence_control(
            apath, segments, troot, max_silence_s=max_silence_s, fade_ms=fade_ms
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
