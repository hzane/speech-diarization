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


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 忽略pyannote的一些警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.*")


@lru_cache(maxsize=1)
def using_speaker_diarization_cnceleb(device: str | int = 0):
    diarizer = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN"),
    )
    diarizer.embedding = ERes2NetV2Encoder()  # ECAPAEncoder(0)
    diarizer.to(torch.device(device))
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


# ---- 静音压缩相关工具 ----
def _merge_union(segments: list[tuple[float, float, str | int]]):
    """将多说话人片段合并成时间上的并集 [(start, end), ...]（单位：秒）。"""
    if not segments:
        return []
    ivals = sorted([(float(s), float(e)) for s, e, _ in segments])
    merged: list[tuple[float, float]] = []
    cur_s, cur_e = ivals[0]
    for s, e in ivals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _append_with_xfade(
    out: torch.Tensor | None,
    piece: torch.Tensor,
    xfade: int,
) -> torch.Tensor:
    """将 piece 追加到 out，使用线性交叉淡化，输入形状均为 (C, T)。"""
    if out is None:
        return piece.clone()
    if xfade <= 0:
        return torch.cat([out, piece], dim=1)

    c = min(xfade, out.shape[1], piece.shape[1])
    if c == 0:
        return torch.cat([out, piece], dim=1)

    win_in = torch.linspace(0.0, 1.0, steps=c, dtype=out.dtype, device=out.device)
    win_out = 1.0 - win_in

    cross = out[:, -c:] * win_out + piece[:, :c] * win_in
    return torch.cat([out[:, :-c], cross, piece[:, c:]], dim=1)


def compress_long_silences(
    y: torch.Tensor,
    sr: int,
    speech_union: list[tuple[float, float]],
    max_silence_s: float = 0.5,
    crossfade_ms: float = 10.0,
) -> tuple[torch.Tensor, list[tuple[float, float, float]]]:
    """
    根据讲话并集，压缩相邻讲话之间过长的静音（> max_silence_s），并用交叉淡化拼接。

    返回：
      - 压缩后的波形 y2 (C, T')
      - pieces: 列表[(orig_start, orig_end, new_start)]，用于时间重映射
    """
    if not speech_union:
        return y, [(0.0, y.shape[1] / sr, 0.0)]

    xfade = int(round(crossfade_ms / 1000.0 * sr))
    n = y.shape[1]
    pieces: list[tuple[int, int]] = []  # 样本级 orig 区间

    prev_end = 0
    for us, ue in speech_union:
        us_i = max(0, int(round(us * sr)))
        ue_i = min(n, int(round(ue * sr)))
        if ue_i <= us_i:
            continue

        # 保留紧邻讲话前的静音 up to max_silence
        gap = us_i - prev_end
        if gap > 0:
            keep = min(gap, int(round(max_silence_s * sr)))
            if keep > 0:
                pieces.append((us_i - keep, us_i))

        # 保留讲话区间
        pieces.append((us_i, ue_i))
        prev_end = ue_i

    # 处理尾部静音
    tail = n - prev_end
    if tail > 0:
        keep = min(tail, int(round(max_silence_s * sr)))
        if keep > 0:
            pieces.append((n - keep, n))

    # 逐段拼接并记录新时间轴
    out: torch.Tensor | None = None
    new_pieces: list[tuple[float, float, float]] = []
    cur_new = 0  # 样本级
    for a, b in pieces:
        p = y[:, a:b]
        out = _append_with_xfade(out, p, xfade)
        # 交叉淡化生效后，新追加长度为 p.shape[1] - overlap，其中 overlap≈xfade
        # 但记录的 new_start 以 piece 拼接前的 cur_new 为准即可
        new_start_s = cur_new / sr
        new_pieces.append((a / sr, b / sr, new_start_s))
        if out is None:
            cur_new = 0
        else:
            # 实际追加长度：第一块直接长度，其它块额外增加 len(p)-c
            # 简化：用 out 的总长度追踪
            cur_new = out.shape[1]

    return out if out is not None else y[:, :0], new_pieces


def remap_segments(
    segments: list[tuple[float, float, str | int]],
    pieces: list[tuple[float, float, float]],
) -> list[tuple[float, float, str | int]]:
    """根据 pieces 将 (start,end,speaker) 重映射到压缩后的时间轴。
    假设每个讲话段完全位于某个 piece(讲话片段) 内。
    """
    remapped: list[tuple[float, float, str | int]] = []
    for s, e, spk in segments:
        # 找到覆盖 start 的 piece
        ps = next((p for p in pieces if p[0] <= s < p[1]), None)
        pe = next((p for p in pieces if p[0] < e <= p[1]), None)
        if ps is None or pe is None:
            # 某些边界误差时回退：不变
            remapped.append((s, e, spk))
            continue
        new_s = ps[2] + (s - ps[0])
        new_e = pe[2] + (e - pe[0])
        remapped.append((new_s, new_e, spk))
    return remapped


def extract_speaker_stems_with_silence_compression(
    audio_path: str | Path,
    segments: list[tuple[float, float, str | int]],
    root: str | Path,
    fade_ms: float = 10.0,
    max_silence_s: float = 0.5,
    crossfade_ms: float = 10.0,
):
    """
    在导出说话人轨道前，先按“讲话并集”压缩过长静音（> max_silence_s），
    并对拼接处做交叉淡化（crossfade_ms）。随后按新时间轴导出各说话人音轨。
    """
    y, sr = torchaudio.load(str(audio_path))
    union = _merge_union(segments)
    y2, pieces = compress_long_silences(y, sr, union, max_silence_s, crossfade_ms)
    remapped = remap_segments(segments, pieces)

    speakers = sorted(list({seg[-1] for seg in remapped}))
    _, n2 = y2.shape
    stems = {spk: torch.zeros_like(y2) for spk in speakers}  # type: ignore

    fade = int(round(fade_ms / 1000.0 * sr))
    fade_tf = torchaudio.transforms.Fade(
        fade_in_len=fade, fade_out_len=fade, fade_shape="linear"
    )

    for s, e, spk in remapped:
        s_i = max(0, int(round(s * sr)))
        e_i = min(n2, int(round(e * sr)))
        if e_i <= s_i:
            continue
        chunk = y2[:, s_i:e_i]
        if (e_i - s_i) >= fade and fade > 0:
            chunk = fade_tf(chunk)
        stems[spk][:, s_i:e_i] += chunk  # type: ignore

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out_paths = {}
    for spk, wav in stems.items():
        out_path = root / f"{Path(audio_path).stem}-compressed-{spk}.flac"
        torchaudio.save(str(out_path), wav, sr, format="flac", bits_per_sample=16)
        out_paths[spk] = out_path
    return out_paths, remapped


def diarize_audio(
    audio_filepath: str,
    min_speakers: int = 2,
    max_speakers: int = 6,
    rttm_filepath: str | None = None,
) -> list[tuple[float, float, str | int]]:
    diarize = using_speaker_diarization_cnceleb()

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


def split_audio_by_diarization(
    diarization: list[tuple[float, float, int | str]],
    audio_path: str | Path,
    root: str | Path,
    fade_duration_ms: int = 10,
    flac_compression_level: int = 8,  # 新增：FLAC 压缩级别 (0-8)
    bits_per_sample: int = 16,  # 新增：输出文件的位深度
):
    """
    将音频切分并保存为指定格式。

    Args:
        diarization: pyannote.audio.Annotation 对象。
        audio_path (str): 原始音频文件路径。
        root (str): 保存切分后音频的根目录。
        fade_duration_ms (int): 淡入淡出时长（毫秒）。
        output_format (str): 输出音频的格式，如 'flac' 或 'wav'。
        flac_compression_level (int): FLAC 格式的压缩级别，范围从 0 (最快) 到 8 (最小)。
        bits_per_sample (int): 输出文件的位深度 (例如 16 或 24)。
        create_manifest (bool): 是否创建 manifest 文件。
    """
    audio_path, root = Path(audio_path), Path(root)

    waveform, sample_rate = torchaudio.load(str(audio_path))

    fade_transform = None
    fade_in_len = 0
    fade_out_len = 0

    fade_samples = int(fade_duration_ms / 1000.0 * sample_rate)
    fade_in_len, fade_out_len = fade_samples, fade_samples
    fade_transform = torchaudio.transforms.Fade(
        fade_in_len=fade_in_len, fade_out_len=fade_out_len, fade_shape="linear"
    )

    for start, end, speaker in track(diarization, description="正在切分音频..."):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        if start_sample >= end_sample:
            continue
        audio_chunk = waveform[:, start_sample:end_sample]

        if audio_chunk.shape[1] >= (fade_in_len + fade_out_len):
            audio_chunk = fade_transform(audio_chunk)

        # 准备输出路径和文件名
        target_path = (
            root
            / f"{audio_path.stem}-{speaker}/{audio_path.stem}-{speaker}-{start_sample}.flac"
        )
        target_path.parent.mkdir(exist_ok=True, parents=True)

        torchaudio.save(
            str(target_path),
            audio_chunk,
            sample_rate,
            format="flac",
            compression=flac_compression_level,
            bits_per_sample=bits_per_sample,
        )


def expand_audios(root: Path):
    if root.is_file():
        root = root.resolve()
        return [root], root.parent

    exts = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"]
    audios = [p for p in root.rglob("*.*") if p.is_file() and p.suffix.lower() in exts]
    return audios, root


def merge_adjacent_skip_shorts(
    segments: list[tuple[float, float, int | str]], gap: float = 1.0, min_speech_s: float = 0.25,
) -> list[tuple[float, float, int | str]]:

    def too_short(s, e, spk)->bool:
        return (e-s)<min_speech_s

    # 辅助函数：合并属于同一个说话人的相邻片段
    segments = [s for s in segments if not too_short(*s)]
    if not segments:
        return []
    merged = [segments[0]]
    for next_s, next_e, next_spk in segments[1:]:
        last_s, last_e, last_spk = merged[-1]
        if next_spk == last_spk and (next_s - last_e) <= gap:
            merged[-1] = (last_s, next_e, last_spk)
        else:
            merged.append((next_s, next_e, next_spk))
    return merged


def main(
    root: str,
    min_speakers: int = 2,
    max_speakers: int = 6,
    min_adjacent_gap_s: float = 1.5,
):
    audios, aroot = expand_audios(Path(root))
    troot = aroot.with_stem(f"{aroot.stem}-speakers")

    for apath in track(audios, description="diarization", disable=True):
        segments = diarize_audio(
            str(apath),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            rttm_filepath=str(apath.with_suffix(".rttm")),
        )
        print(apath, len(segments))

        segments = merge_adjacent_skip_shorts(segments, gap=min_adjacent_gap_s)
        split_audio_by_diarization(segments, apath, troot)
    ...


if __name__ == "__main__":
    from jsonargparse import auto_cli

    "/data.d/bilix/bilix/yangliv-30s/ytt65VoeVxDwU-2157-5_ROCK_ROAST5_vocals_5_dialog_0.wav"
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
