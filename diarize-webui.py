import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr
from anti_stick_diarize import diarize

BACKENDS = ["speechbrain-ecapa", "ali-eres2netv2", "ali-campp", "pyannote"]

# 固定颜色表（避免每次随机不同）
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def gradio_numpy_audio(audio_input: tuple[int, np.ndarray]):
    sr, y = audio_input
    if y.ndim ==2 :
        y = y[:, 0]

    y = y.astype(np.float32)/32768
    return y, sr


def run_diarize(
    audio,
    vad_thr,
    min_speech,
    min_silence,
    speech_pad,
    scd_win,
    scd_step,
    scd_thr,
    cluster_cos,
    merge_gap,
    merge_maxturn,
    merge_mincos,
    reseg,
):
    y, sr = gradio_numpy_audio(audio)
    segs = diarize(
        wav_path=(y, sr),
        sr=16000,
        target_lufs=-18.0,
        vad_thr=vad_thr,
        min_speech=min_speech,
        min_silence=min_silence,
        speech_pad=speech_pad,
        morph_bridge_ms=80.0,
        scd_win=scd_win,
        scd_step=scd_step,
        scd_thr=scd_thr,
        cluster_cos=cluster_cos,
        merge_gap=merge_gap,
        merge_maxturn=merge_maxturn,
        merge_mincos=merge_mincos,
        reseg=1 if reseg else 0,
    )

    rows = []
    for i, s in enumerate(segs, 1):
        rows.append(
            {
                "idx": i,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "dur": round(s.end - s.start, 3),
                "speaker": f"SPK_{s.spk}",
            }
        )
    df = pd.DataFrame(rows)
    print('dataframe', df.shape)

    # 绘图（波形 + 着色）
    # 注意：这里用 librosa 的波形作背景，着色使用 axvspan
    t = np.arange(len(y)) / sr
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.plot(t, y, linewidth=0.6)
    if len(segs) > 0:
        max_spk = max(s.spk for s in segs if s.spk is not None)
        for s in segs:
            color = COLORS[(s.spk or 0) % len(COLORS)]
            ax.axvspan(s.start, s.end, alpha=0.25, color=color)
    ax.set_xlim(0, max(1e-3, t[-1]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform with diarization spans")
    plt.tight_layout()
    print('here')
    return fig, df


def build_ui():
    with gr.Blocks(title="Anti‑Sticky Diarization Viewer") as demo:
        gr.Markdown(
            "## 抗粘连分割 · 交互式可视化\n上传音频，选择编码器后端，实时查看分段效果。"
        )
        with gr.Row():
            audio = gr.Audio(sources=["upload"], type="numpy", label="音频")
        with gr.Accordion("参数（可选调优）", open=False):
            with gr.Row():
                vad_thr = gr.Slider(0.3, 0.9, 0.55, step=0.01, label="VAD 阈值")
                min_speech = gr.Slider(
                    0.05, 0.6, 0.15, step=0.01, label="最短语音段(s)"
                )
                min_silence = gr.Slider(0.03, 0.5, 0.10, step=0.01, label="最短静音(s)")
                speech_pad = gr.Slider(
                    0.0, 0.2, 0.04, step=0.01, label="语音边缘填充(s)"
                )
            with gr.Row():
                scd_win = gr.Slider(0.4, 1.5, 0.8, step=0.05, label="SCD 滑窗(s)")
                scd_step = gr.Slider(0.05, 0.6, 0.2, step=0.01, label="SCD 步长(s)")
                scd_thr = gr.Slider(0.3, 2, 1.20, step=0.01, label="SCD 峰值阈(z)")
                reseg = gr.Checkbox(value=True, label="帧级重分配（更细边界）")
            with gr.Row():
                cluster_cos = gr.Slider(
                    0.15, 0.9, 0.65, step=0.01, label="聚类相似阈(余弦)"
                )
                merge_gap = gr.Slider(
                    0.01, 10.5, .75, step=0.01, label="合并允许间隙(s)"
                )
                merge_maxturn = gr.Slider(
                    2.0, 30.0, 10.0, step=0.5, label="同人最长连续(s)"
                )
                merge_mincos = gr.Slider(
                    0.1, 0.99, 0.75, step=0.01, label="跨缝最小相似度"
                )

        btn = gr.Button("运行")
        fig = gr.Plot(label="波形与分段")
        table = gr.Dataframe(label="Segments", interactive=False)

        btn.click(
            fn=run_diarize,
            inputs=[
                audio,
                vad_thr,
                min_speech,
                min_silence,
                speech_pad,
                scd_win,
                scd_step,
                scd_thr,
                cluster_cos,
                merge_gap,
                merge_maxturn,
                merge_mincos,
                reseg,
            ],
            outputs=[fig, table],
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
