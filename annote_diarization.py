import torch
import os
import pandas as pd
import numpy as np
import warnings
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pyannote.audio.pipelines.utils.hook import ProgressHook

from pathlib import Path
from rich.progress import track
from ecapa_annote import SpeechBrainECAPA
from functools import lru_cache


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 忽略pyannote的一些警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.*")


@lru_cache(maxsize=1)
def using_speaker_diarization_cnceleb(device:str='cuda'):
    diarizer = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN"),
        )
    diarizer.embedding = SpeechBrainECAPA()
    diarizer.to(torch.device(device))
    return diarizer


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


def export_speaker_stems(y: np.ndarray, sr: int, segments: list, out_dir: str,
                         fade_ms: float = 10.0, prefix: str = "stem_"):
    """
    基于分段结果（[{start, end, speaker}, ...]）将每个说话人导出为独立音轨（与原音频同长度）。
    - 重叠段会在多个轨里同时出现（若要“分离”重叠，见下方 TSE 方案）。
    - 提供短交叉淡入淡出以避免段边界咔嗒声。
    """
    speakers = sorted(list({seg["speaker"] for seg in segments}))
    n = len(y)
    stems = {spk: np.zeros_like(y, dtype=np.float32) for spk in speakers}

    fade = int(round(fade_ms / 1000.0 * sr))
    win_in  = np.linspace(0.0, 1.0, num=fade, dtype=np.float32) if fade > 0 else None
    win_out = np.linspace(1.0, 0.0, num=fade, dtype=np.float32) if fade > 0 else None

    for seg in segments:
        spk = seg["speaker"]
        s = max(0, int(round(seg["start"] * sr)))
        e = min(n, int(round(seg["end"]   * sr)))
        if e <= s: continue
        # 复制原音到对应轨
        stems[spk][s:e] += y[s:e]

        # 边界淡入/淡出，减少咔嗒声（不改变对齐）
        if fade > 0:
            if s+fade <= e:
                stems[spk][s:s+fade] *= win_in
            if e-fade >= s:
                stems[spk][e-fade:e] *= win_out

    # 写盘
    os.makedirs(out_dir, exist_ok=True)
    out_paths = {}
    for spk, wav in stems.items():
        # 防止叠加导致少数峰值>1
        mx = np.max(np.abs(wav)) + 1e-9
        if mx > 1.0:
            wav = wav / mx * 0.99
        out_path = os.path.join(out_dir, f"{prefix}{spk}.wav")
        sf.write(out_path, wav, sr)
        out_paths[spk] = out_path
    return out_paths


def diarize_audiox(
    audio_file_path: str,
    min_speakers: int,
    max_speakers: int,
):
    diarize = using_speaker_diarization_cnceleb()

    with ProgressHook() as hook:
        diarization = diarize(
            audio_file_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hook = hook,
        )

    speech_activity = diarization.support()

    # 1. 创建DataFrame
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker
        })
    df = pd.DataFrame(segments)

    return df


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




# 假设 diarization 是 pyannote.audio.Annotation 类型的对象
# from pyannote.core import Annotation, Segment, Timeline

def split_audio_by_diarization(
    diarization,
    audio_path: str,
    output_dir: str,
    fade_duration_ms: int = 10,
    output_format: str = "flac",          # 新增：输出格式 ('flac', 'wav', 'mp3' 等)
    flac_compression_level: int = 8,    # 新增：FLAC 压缩级别 (0-8)
    bits_per_sample: int = 16,            # 新增：输出文件的位深度
    create_manifest: bool = True
):
    """
    根据 pyannote 的 diarization 结果，将音频切分并保存为指定格式。

    Args:
        diarization: pyannote.audio.Annotation 对象。
        audio_path (str): 原始音频文件路径。
        output_dir (str): 保存切分后音频的根目录。
        fade_duration_ms (int): 淡入淡出时长（毫秒）。
        output_format (str): 输出音频的格式，如 'flac' 或 'wav'。
        flac_compression_level (int): FLAC 格式的压缩级别，范围从 0 (最快) 到 8 (最小)。
        bits_per_sample (int): 输出文件的位深度 (例如 16 或 24)。
        create_manifest (bool): 是否创建 manifest 文件。
    """
    # --- 1. 参数验证 ---
    supported_formats = ["flac", "wav"]
    if output_format not in supported_formats:
        raise ValueError(f"不支持的输出格式: '{output_format}'. 请选择: {supported_formats}")

    print(f"正在加载原始音频文件: {audio_path}...")
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"错误：无法加载音频文件。 {e}")
        return

    # --- 2. 初始化 Fade transform ---
    fade_transform = None
    fade_in_len = 0
    fade_out_len = 0
    if fade_duration_ms > 0:
        fade_samples = int(fade_duration_ms / 1000.0 * sample_rate)
        fade_in_len, fade_out_len = fade_samples, fade_samples
        fade_transform = torchaudio.transforms.Fade(
            fade_in_len=fade_in_len, fade_out_len=fade_out_len, fade_shape='linear'
        )
        print(f"已启用淡入淡出效果，时长: {fade_duration_ms}ms。")

    # --- 3. 准备输出 ---
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"音频将被切分并保存到: {output_path.absolute()} (格式: {output_format.upper()})")

    speaker_counts = {}
    manifest_data = []

    # --- 4. 循环切分和保存 ---
    for turn, _, speaker in track(diarization.itertracks(yield_label=True), description="正在切分音频..."):
        # 切分
        start_sample = int(turn.start * sample_rate)
        end_sample = int(turn.end * sample_rate)
        audio_chunk = waveform[:, start_sample:end_sample]

        if audio_chunk.shape[1] == 0:
            continue

        # 应用淡入淡出
        if fade_transform and audio_chunk.shape[1] >= (fade_in_len + fade_out_len):
            audio_chunk = fade_transform(audio_chunk)

        # 准备输出路径和文件名
        speaker_dir = output_path / speaker
        speaker_dir.mkdir(exist_ok=True)
        count = speaker_counts.get(speaker, 0)

        # 使用新的 output_format 来确定文件扩展名
        output_filename = f"{speaker}_{count:03d}.{output_format}"
        full_output_path = speaker_dir / output_filename

        # --- 核心修改：根据格式保存文件 ---
        if output_format == "flac":
            torchaudio.save(
                full_output_path, audio_chunk, sample_rate,
                format="flac",
                compression=flac_compression_level,
                bits_per_sample=bits_per_sample
            )
        elif output_format == "wav":
            torchaudio.save(
                full_output_path, audio_chunk, sample_rate,
                format="wav",
                bits_per_sample=bits_per_sample
            )

        # 更新计数器和 manifest
        speaker_counts[speaker] = count + 1
        if create_manifest:
            manifest_data.append({
                "output_path": str(full_output_path.absolute()),
                "speaker": speaker,
                "original_start_s": turn.start,
                "original_end_s": turn.end
            })

    # --- 5. 写入 manifest 文件 ---
    if create_manifest and manifest_data:
        manifest_path = output_path / "manifest.csv"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("output_path,speaker,original_start_s,original_end_s\n")
            for item in manifest_data:
                f.write(f"{item['output_path']},{item['speaker']},{item['original_start_s']:.3f},{item['original_end_s']:.3f}\n")
        print(f"Manifest 文件已创建: {manifest_path.absolute()}")

    print("\n切分完成！")
    for speaker, count in speaker_counts.items():
        print(f"  - 为 {speaker} 创建了 {count} 个 {output_format.upper()} 文件。")


if __name__ == "__main__x":
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