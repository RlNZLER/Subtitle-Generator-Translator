#!/usr/bin/env python3
import os, json, shutil, subprocess, tempfile
from pathlib import Path
import gradio as gr
from faster_whisper import WhisperModel

# ---------- helpers ----------
def srt_ts(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def vtt_ts(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    hh = f"{h:02}:" if h > 0 else ""
    return f"{hh}{m:02}:{s:02}.{ms:03}"

def write_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            speaker = f"{seg.get('speaker')}:\n" if seg.get("speaker") else ""
            f.write(f"{i}\n{srt_ts(seg['start'])} --> {srt_ts(seg['end'])}\n{speaker}{seg['text'].strip()}\n\n")

def write_vtt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            if seg.get("speaker"):
                f.write(f"{vtt_ts(seg['start'])} --> {vtt_ts(seg['end'])}\n<v {seg['speaker']}>{seg['text'].strip()}</v>\n\n")
            else:
                f.write(f"{vtt_ts(seg['start'])} --> {vtt_ts(seg['end'])}\n{seg['text'].strip()}\n\n")

def write_ass(segments, path):
    palette = ["&H00FF00&","&H0000FF&","&HFF0000&","&H00FFFF&","&HFF00FF&","&HFFFF00&"]  # BGR
    spk_color, idx = {}, 0
    def ass_ts(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); cs = int((t - int(t)) * 100)
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"
    with open(path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,40,40,40,1\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        for seg in segments:
            spk = seg.get("speaker")
            txt = seg["text"].strip().replace("\n"," ")
            if spk:
                if spk not in spk_color:
                    spk_color[spk] = palette[idx % len(palette)]; idx += 1
                color = spk_color[spk]
                text = f"{{\\c{color}}}[{spk}] {txt}"
            else:
                text = txt
            f.write(f"Dialogue: 0,{ass_ts(seg['start'])},{ass_ts(seg['end'])},Default,,0,0,0,,{text}\n")

def extract_audio(input_path: Path, out_format="mp3") -> Path:
    assert out_format in {"mp3","wav","flac"}
    out_path = input_path.with_suffix(f".{out_format}")
    if out_format == "mp3":
        codec = ["-vn","-acodec","libmp3lame","-q:a","2"]
    elif out_format == "wav":
        codec = ["-vn","-acodec","pcm_s16le"]
    else:
        codec = ["-vn","-acodec","flac"]
    cmd = ["ffmpeg","-y","-i",str(input_path),*codec,str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_path

# ---------- diarization (optional) ----------
def try_diarize(media_path: Path):
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return None, "No HUGGINGFACE_TOKEN set → skipping diarization."
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        diar = pipeline(str(media_path))
        turns = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
        turns.sort(key=lambda x: (x["start"], x["end"]))
        remap, k = {}, 1
        for t in turns:
            if t["speaker"] not in remap:
                remap[t["speaker"]] = f"Speaker {k}"; k += 1
            t["speaker"] = remap[t["speaker"]]
        return turns, None
    except Exception as e:
        return None, f"Diarization error: {e}"

def label_segment(seg, turns):
    if not turns: return None
    s0, s1 = seg["start"], seg["end"]
    best, overlap = None, 0.0
    for t in turns:
        o = max(0.0, min(s1, t["end"]) - max(s0, t["start"]))
        if o > overlap:
            best, overlap = t["speaker"], o
    return best

# ---------- core ----------
def transcribe_ui(
    file_obj,
    model_size,
    device,
    compute_type,
    beam_size,
    vad_filter,
    words,
    language,
    translate,
    diarize,
    save_audio_fmt
):
    if file_obj is None:
        return "Please upload an audio/video file.", None, None, None, None, None, None

    media_path = Path(file_obj.name)
    workdir = Path(tempfile.mkdtemp(prefix="fwui_"))
    local_path = workdir / media_path.name
    shutil.copy(media_path, local_path)

    # Init whisper
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Transcribe
    segments_gen, info = model.transcribe(
        str(local_path),
        beam_size=int(beam_size),
        vad_filter=vad_filter,
        language=(language or None) if language != "" else None,
        word_timestamps=words,
        condition_on_previous_text=True,
        task="translate" if translate else "transcribe",
    )

    segments = []
    for seg in segments_gen:
        item = {"start": float(seg.start), "end": float(seg.end), "text": seg.text}
        if words and seg.words:
            item["words"] = [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in seg.words]
        segments.append(item)

    # Optional diarization
    turns, dia_msg = (None, None)
    if diarize:
        turns, dia_msg = try_diarize(local_path)
        if turns:
            for s in segments:
                s["speaker"] = label_segment(s, turns)

    # Write outputs
    base = workdir / local_path.stem
    txt_path  = base.with_suffix(".txt")
    srt_path  = base.with_suffix(".srt")
    vtt_path  = base.with_suffix(".vtt")
    ass_path  = base.with_suffix(".ass")
    json_path = base.with_suffix(".json")

    txt_path.write_text(
        "\n".join((s.get("speaker", "") + (": " if s.get("speaker") else "") + s["text"].strip()) for s in segments),
        encoding="utf-8"
    )
    write_srt(segments, srt_path)
    write_vtt(segments, vtt_path)
    write_ass(segments, ass_path)
    meta = {
        "language": info.language,
        "language_probability": float(info.language_probability or 0),
        "model": model_size,
        "translate": bool(translate),
        "segments": segments,
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional audio extraction
    audio_out = None
    if save_audio_fmt in {"mp3","wav","flac"}:
        try:
            audio_out = extract_audio(local_path, save_audio_fmt)
        except Exception as e:
            dia_msg = (dia_msg or "") + f"\nAudio extraction failed: {e}"

    # Summary message
    msg = [
        f"Detected language: {info.language} (p={getattr(info,'language_probability',0):.2f})",
        f"Segments: {len(segments)}",
        f"Model: {model_size}, Device: {device}, Compute: {compute_type}"
    ]
    if translate:
        msg.append("Task: translate → English")
    if vad_filter:
        msg.append("VAD: on (skip silences)")
    if diarize:
        msg.append("Diarization: on" if turns else "Diarization: requested but not applied.")
    if dia_msg:
        msg.append(dia_msg.strip())
    msg = "\n".join(msg)

    return (
        msg,
        str(txt_path),
        str(srt_path),
        str(vtt_path),
        str(ass_path),
        str(json_path),
        (str(audio_out) if audio_out else None)
    )

# ---------- UI ----------
MODEL_CHOICES = ["small","medium","large-v3"]
DEVICE_CHOICES = ["auto","cuda","cpu"]
COMPUTE_CHOICES = ["float16","int8_float16","int8","auto"]

with gr.Blocks(title="Whisper Transcriber (GUI)") as demo:
    gr.Markdown("## 🎙️ Whisper Transcriber\nUpload audio/video, choose options, and download subtitles/transcripts.")
    with gr.Row():
        inp = gr.File(file_count="single", file_types=["audio","video"], label="Audio/Video file")
    with gr.Row():
        model = gr.Dropdown(MODEL_CHOICES, value="small", label="Model")
        device = gr.Dropdown(DEVICE_CHOICES, value="cuda", label="Device")
        ctype  = gr.Dropdown(COMPUTE_CHOICES, value="float16", label="Compute type")
        beam   = gr.Slider(1, 8, value=5, step=1, label="Beam size")
    with gr.Row():
        vad    = gr.Checkbox(True, label="Skip silences (VAD)")
        words  = gr.Checkbox(False, label="Word timestamps")
        translate = gr.Checkbox(False, label="Translate to English")
        diar   = gr.Checkbox(False, label="Speaker diarization (requires HF token)")
        lang   = gr.Textbox(value="", label='Force language (e.g., "en", "ar") — leave blank for auto')
    with gr.Row():
        save_audio = gr.Radio(choices=["none","mp3","wav","flac"], value="none", label="Also save audio extracted from video?")
    go = gr.Button("Transcribe", variant="primary")

    out_msg  = gr.Textbox(label="Status", lines=6)
    out_txt  = gr.File(label="Transcript (.txt)")
    out_srt  = gr.File(label="Subtitles (.srt)")   # works in VLC (same basename as video)
    out_vtt  = gr.File(label="WebVTT (.vtt)")
    out_ass  = gr.File(label="Styled subtitles (.ass, per-speaker colors)")
    out_json = gr.File(label="Metadata (.json)")
    out_audio= gr.File(label="Extracted audio (optional)")

    go.click(
        transcribe_ui,
        inputs=[inp, model, device, ctype, beam, vad, words, lang, translate, diar, save_audio],
        outputs=[out_msg, out_txt, out_srt, out_vtt, out_ass, out_json, out_audio]
    )

if __name__ == "__main__":
    # Share=False keeps it local. Open the printed URL in your browser.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
