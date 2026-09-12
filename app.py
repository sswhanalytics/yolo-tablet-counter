"""
Tablet Counter - Streamlit + YOLO live camera app
--------------------------------------------------
Features:
- Continuous live camera feed (via streamlit-webrtc)
- Real-time YOLO inference with bounding boxes drawn on each frame
- Live tablet count display
- Snapshot button to capture the current annotated frame
- Snapshot gallery with download buttons
 
Run with:
    streamlit run app.py
"""

import threading
import time
from datetime import datetime
from io import BytesIO

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Page Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="YOLO Tablet Counter", page_icon="💊", layout="wide")
st.title("💊 YOLO Live Tablet Counter")
st.caption("Detect and count tablets in real time.")

# --------------------------------------------------------------------------
# Sidebar Setting Controls
# --------------------------------------------------------------------------
st.sidebar.header("Settings")

MODEL_PATH = st.sidebar.text_input("Model Path", value="best.pt")
CONF_THRESH = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
IOU_THRESH = st.sidebar.slider("IoU threshold (NMS)", 0.0, 1.0, 0.45, 0.01)
DOT_RADIUS = st.sidebar.slider("Dot radius", 2, 15, 6)

@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(path: str):
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
    st.sidebar.success(f"Model loaded: {MODEL_PATH}")
except Exception as e:
    st.sidebar.error(f"Could not load model: {e}")
    st.stop()
    
# --------------------------------------------------------------------------
# Shared state between the WebRTC background thread and the main thread.
# streamlit-webrtc runs frame processing in its own thread, so we can't rely
# on st.session_state there directly — use a small thread-safe holder.
# --------------------------------------------------------------------------

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_count = 0
        
if "shared_state" not in st.session_state:
    st.session_state.shared_state = SharedState()
if "snapshots" not in st.session_state:
    st.session_state.snapshots = []   # list of (timestamp_str, PIL.Image, count)
 
shared_state: SharedState = st.session_state.shared_state

# --------------------------------------------------------------------------
# Video processing callback
# --------------------------------------------------------------------------
def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
 
    results = model.predict(
        img,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False,
    )[0]
 
    count = 0
    if results.boxes is not None:
        count = len(results.boxes)
        for idx, box in enumerate(results.boxes, start=1):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
            # Solid dot at the tablet's center, with a thin white outline
            # so it stays visible against tablets of any color.
            cv2.circle(img, (cx, cy), DOT_RADIUS, (0, 0, 255), -1)
            cv2.circle(img, (cx, cy), DOT_RADIUS, (255, 255, 255), 1)
 
    # Overlay running count on the top-left corner
    banner = f"Tablets detected: {count}"
    cv2.rectangle(img, (0, 0), (330, 40), (0, 0, 0), -1)
    cv2.putText(
        img, banner, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )
 
    with shared_state.lock:
        shared_state.latest_frame = img.copy()
        shared_state.latest_count = count
 
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --------------------------------------------------------------------------
# Layout: video stream + live stats + snapshot controls
# --------------------------------------------------------------------------
col_video, col_stats = st.columns([3, 1])
 
with col_video:
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
 
    webrtc_ctx = webrtc_streamer(
        key="tablet-counter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=process_frame,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
 
with col_stats:
    st.subheader("Live count")
    count_placeholder = st.empty()
 
    st.divider()
    snapshot_clicked = st.button("📸 Take snapshot", use_container_width=True, type="primary")
    clear_clicked = st.button("🗑️ Clear snapshots", use_container_width=True)
 
# Live-updating count display while the stream is running
if webrtc_ctx.state.playing:
    while webrtc_ctx.state.playing:
        with shared_state.lock:
            current_count = shared_state.latest_count
        count_placeholder.metric("Tablets detected", current_count)
 
        if snapshot_clicked:
            with shared_state.lock:
                frame = shared_state.latest_frame
                cnt = shared_state.latest_count
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.snapshots.insert(0, (ts, pil_img, cnt))
                st.toast(f"Snapshot saved ({cnt} tablets)")
            snapshot_clicked = False  # avoid duplicate saves in the loop
        time.sleep(0.15)
        break  # rerun via Streamlit's own refresh cycle instead of a tight loop
else:
    count_placeholder.metric("Tablets detected", 0)
 
if clear_clicked:
    st.session_state.snapshots = []
    st.rerun()
 
# --------------------------------------------------------------------------
# Snapshot gallery
# --------------------------------------------------------------------------
st.divider()
st.subheader("📁 Snapshots")
 
if not st.session_state.snapshots:
    st.info("No snapshots yet. Click 'Take snapshot' while the camera is running.")
else:
    n_cols = 4
    rows = [
        st.session_state.snapshots[i : i + n_cols]
        for i in range(0, len(st.session_state.snapshots), n_cols)
    ]
    for row in rows:
        cols = st.columns(n_cols)
        for col, (ts, img, cnt) in zip(cols, row):
            with col:
                st.image(img, caption=f"{ts} — {cnt} tablets", use_container_width=True)
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    "Download",
                    data=buf.getvalue(),
                    file_name=f"tablet_snapshot_{ts.replace(':', '-').replace(' ', '_')}.png",
                    mime="image/png",
                    key=f"dl_{ts}_{cnt}_{id(img)}",
                )