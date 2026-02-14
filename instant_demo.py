"""
INSTANT Real-Time Demo
Use webcam for immediate results - perfect for presentations!
"""

import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np

# Load model ONCE
print("⚡ Loading model for real-time detection...")
model = YOLO("best.pt")
print("✅ Ready for instant detection!")

def detect_instant(image):
    """INSTANT detection on single frame - NO WAITING!"""

    if image is None:
        return None, "Waiting for image..."

    # Detect on single image - INSTANT!
    results = model.predict(
        image,
        conf=0.25,
        verbose=False,
        imgsz=416  # Fast!
    )

    # Draw boxes
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Count
    num_people = len(results[0].boxes)

    status = f"**Found {num_people} person(s)!**" if num_people > 0 else "No people detected"

    return annotated, status

# TWO MODES: Webcam (instant) + Video (fast)

def process_video_fast(video):
    """Fast video processing"""
    if video is None:
        return None, "Upload video"

    import tempfile
    output = tempfile.mktemp(suffix='.mp4')

    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FAST: Small output
    ow, oh = w//2, h//2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps//2, (ow, oh))

    count = 0
    total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1

        # SPEED: Skip frames
        if count % 4 != 0:
            continue

        # SPEED: Small frame
        small = cv2.resize(frame, (640, 360))
        results = model.predict(small, conf=0.3, verbose=False, imgsz=320)

        annotated = results[0].plot()
        annotated = cv2.resize(annotated, (w, h))

        num = len(results[0].boxes)
        total += num

        cv2.putText(annotated, f"People: {num}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        small_out = cv2.resize(annotated, (ow, oh))
        out.write(small_out)

    cap.release()
    out.release()

    return output, f"**Found {total} people!**"

# CREATE INTERFACE WITH TABS
with gr.Blocks(title="⚡ Real-Time Detection") as demo:

    gr.Markdown("# ⚡ Real-Time Drone Person Detection")
    gr.Markdown("### Choose your demo mode:")

    with gr.Tabs():

        # WEBCAM TAB - INSTANT!
        with gr.Tab("📸 Instant Demo (Webcam/Image)"):
            gr.Markdown("### Upload image or use webcam for INSTANT results!")

            with gr.Row():
                webcam_input = gr.Image(
                    sources=["webcam", "upload"],
                    label="📷 Webcam or Upload Image",
                    type="numpy"
                )
                instant_output = gr.Image(label="⚡ Instant Results")

            instant_status = gr.Markdown("**Ready!** Upload image or use webcam")

            # Auto-detect on change
            webcam_input.change(
                fn=detect_instant,
                inputs=webcam_input,
                outputs=[instant_output, instant_status]
            )

        # FAST VIDEO TAB
        with gr.Tab("🎬 Fast Video"):
            gr.Markdown("### Upload short video for fast processing")

            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video (keep it short!)")
                    video_btn = gr.Button("⚡ Process Fast", variant="primary")

                with gr.Column():
                    video_output = gr.Video(label="Results")
                    video_status = gr.Markdown()

            video_btn.click(
                fn=process_video_fast,
                inputs=video_input,
                outputs=[video_output, video_status]
            )

    gr.Markdown("""
---
**💡 Tips for Best Demo:**
- **Instant Mode:** Use webcam or upload single frame - NO WAITING!
- **Fast Video:** Keep videos under 30 seconds for quick demo
- **Speed optimizations:** Frame skipping + small resolution = FAST! ⚡
    """)

demo.launch(share=True)