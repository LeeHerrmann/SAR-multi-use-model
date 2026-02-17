"""
SAR Drone Detection - All-in-One Demo
  • Webcam / image  → instant detection
  • Video upload    → processed with colour-coded boxes
Confidence colours:  🔴 Red < 40%  |  🟡 Yellow 40-70%  |  🟢 Green ≥ 70%
"""

import gradio as gr
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# ── model ──────────────────────────────────────────────────────────────
print("Loading model...")
model = YOLO("best.pt")
print("Ready!\n")


# ── shared helper ───────────────────────────────────────────────────────
def draw_boxes(frame, boxes):
    """Draw colour-coded bounding boxes on a frame (BGR numpy array)."""
    out = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu())

        if conf >= 0.70:
            color, label = (0, 255, 0),   f"HIGH  {conf:.0%}"   # green
        elif conf >= 0.40:
            color, label = (0, 255, 255), f"MED   {conf:.0%}"   # yellow
        else:
            color, label = (0, 0, 255),   f"LOW   {conf:.0%}"   # red

        # box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        # label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


# ── tab 1: instant (webcam / image) ────────────────────────────────────
def detect_image(image, conf_thresh):
    if image is None:
        return None, "Upload an image or use the webcam."

    results = model.predict(image, conf=conf_thresh, verbose=False, imgsz=640)

    # image comes in as RGB numpy → convert to BGR for cv2 drawing
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_bgr = draw_boxes(bgr, results[0].boxes)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    n     = len(results[0].boxes)
    high  = sum(1 for b in results[0].boxes if float(b.conf[0]) >= 0.70)
    med   = sum(1 for b in results[0].boxes if 0.40 <= float(b.conf[0]) < 0.70)
    low   = sum(1 for b in results[0].boxes if float(b.conf[0]) < 0.40)

    report = (
        f"**Detected: {n} person(s)**\n\n"
        f"🟢 High confidence (≥70%): {high}\n\n"
        f"🟡 Medium (40–70%): {med}\n\n"
        f"🔴 Low (<40%): {low}"
    )
    return annotated_rgb, report


# ── tab 2: video upload ─────────────────────────────────────────────────
def detect_video(video_path, conf_thresh, skip, progress=gr.Progress()):
    if video_path is None:
        return None, "Upload a video first."

    cap    = cv2.VideoCapture(video_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    out      = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx   = 0
    stats       = {"high": 0, "med": 0, "low": 0}
    last_frame  = None   # reuse previous annotated frame for skipped frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        progress(frame_idx / max(total, 1), desc=f"Frame {frame_idx}/{total}")

        if frame_idx % skip == 0 or last_frame is None:
            results    = model.predict(frame, conf=conf_thresh,
                                       verbose=False, imgsz=640)
            annotated  = draw_boxes(frame, results[0].boxes)

            for b in results[0].boxes:
                c = float(b.conf[0])
                if   c >= 0.70: stats["high"] += 1
                elif c >= 0.40: stats["med"]  += 1
                else:           stats["low"]  += 1

            last_frame = annotated
        else:
            annotated = last_frame   # fast: reuse without re-detecting

        out.write(annotated)

    cap.release()
    out.release()

    total_det = stats["high"] + stats["med"] + stats["low"]
    report = (
        f"**Frames processed: {frame_idx}**\n\n"
        f"🟢 High confidence: {stats['high']}\n\n"
        f"🟡 Medium confidence: {stats['med']}\n\n"
        f"🔴 Low confidence: {stats['low']}\n\n"
        f"**Total detections: {total_det}**"
    )
    return out_path, report


# ── interface ───────────────────────────────────────────────────────────
CSS = """
.header {
    text-align: center;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 30px; border-radius: 14px;
    color: white; margin-bottom: 20px;
}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(),
               title="SAR Drone Detection") as app:

    gr.HTML("""
    <div class="header">
        <h1 style="margin:0; font-size:40px;">🚁 SAR Drone Detection</h1>
        <p style="margin:8px 0 0; font-size:18px; opacity:.9;">
            Colour-coded confidence  ·  🔴 Low  🟡 Medium  🟢 High
        </p>
    </div>
    """)

    with gr.Tabs():

        # ── TAB 1 ──────────────────────────────────────────────────────
        with gr.Tab("📸 Image / Webcam"):
            gr.Markdown("Upload a photo **or** use your webcam — results appear instantly.")

            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(
                        sources=["upload", "webcam"],
                        label="Input",
                        type="numpy"
                    )
                    img_conf = gr.Slider(0.05, 0.50, value=0.25, step=0.05,
                                         label="Confidence threshold")
                    img_btn  = gr.Button("🔍 Detect", variant="primary")

                with gr.Column():
                    img_out    = gr.Image(label="Result")
                    img_report = gr.Markdown()

            # auto-fire on webcam stream, manual button for uploads
            img_input.change(detect_image,
                             [img_input, img_conf],
                             [img_out, img_report])
            img_btn.click(detect_image,
                          [img_input, img_conf],
                          [img_out, img_report])

        # ── TAB 2 ──────────────────────────────────────────────────────
        with gr.Tab("🎬 Video"):
            gr.Markdown("Upload drone footage — every frame is annotated with colour-coded boxes.")

            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload video")
                    vid_conf  = gr.Slider(0.05, 0.50, value=0.25, step=0.05,
                                          label="Confidence threshold")
                    vid_skip  = gr.Slider(1, 6, value=2, step=1,
                                          label="Process every N-th frame  (1 = all, higher = faster)")
                    vid_btn   = gr.Button("🎬 Process Video", variant="primary")

                with gr.Column():
                    vid_out    = gr.Video(label="Result")
                    vid_report = gr.Markdown()

            vid_btn.click(detect_video,
                          [vid_input, vid_conf, vid_skip],
                          [vid_out, vid_report])

app.launch(share=True)
