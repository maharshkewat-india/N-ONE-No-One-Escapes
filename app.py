import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from deepface import DeepFace
except Exception:  # pragma: no cover - optional dependency path
    DeepFace = None

st.set_page_config(
    page_title="PROJECT N-ONE | Advanced AI Surveillance Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background: linear-gradient(135deg, #050816 0%, #111827 100%); color: #f9fafb; }
    .stSidebar { background-color: #0f172a; }
    div.stButton > button:first-child {
        background-color: #22c55e;
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 8px;
    }
    div.stButton > button:hover {
        filter: brightness(1.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT_DIR = Path(__file__).resolve().parent
REG_DIR = ROOT_DIR / "registered_faces"
LOG_DIR = ROOT_DIR / "detection_logs"
CSV_LOG_PATH = LOG_DIR / "system_audit_logs.csv"
TEMP_FRAME_PATH = ROOT_DIR / "temp_current_frame.jpg"
TEMP_VIDEO_PATH = ROOT_DIR / "temp_video_upload.mp4"


def initialize_directories() -> None:
    REG_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)


def initialize_log_file() -> None:
    if not CSV_LOG_PATH.exists():
        pd.DataFrame(columns=["Timestamp", "Mode", "Subject_Name", "Role", "Event_Status"]).to_csv(
            CSV_LOG_PATH,
            index=False,
        )


def log_event(mode: str, name: str, role: str, status: str) -> None:
    row = pd.DataFrame(
        [[time.strftime("%Y-%m-%d %H:%M:%S"), mode, name, role, status]],
        columns=["Timestamp", "Mode", "Subject_Name", "Role", "Event_Status"],
    )
    row.to_csv(CSV_LOG_PATH, mode="a", header=False, index=False)


def configure_session_state() -> None:
    defaults = {
        "authenticated": False,
        "role": "Guest",
        "streaming": False,
        "last_frame": None,
        "last_status": "Idle",
        "last_detection": "No detections yet",
        "last_detection_details": "No details available",
        "last_logged_event": "",
        "last_match_snapshot": None,
        "last_match_caption": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_registered_profiles() -> None:
    if not REG_DIR.exists():
        return
    for image_file in REG_DIR.glob("*"):
        if image_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image_file.unlink()
    get_registered_profiles.clear()


def clear_audit_log() -> None:
    pd.DataFrame(columns=["Timestamp", "Mode", "Subject_Name", "Role", "Event_Status"]).to_csv(
        CSV_LOG_PATH,
        index=False,
    )


@st.cache_data
def get_registered_profiles() -> list[dict]:
    profiles: list[dict] = []
    if not REG_DIR.exists():
        return profiles
    for image_file in sorted(REG_DIR.glob("*")):
        if image_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        name = image_file.stem.split("_", 1)[-1]
        role_type = "Lost" if image_file.stem.startswith("Lost") else "Member"
        profile_face = cv2.imread(str(image_file))
        profile_face = extract_face_crop(profile_face)
        profile_face = standardize_face(profile_face)
        descriptor = compute_orb_descriptors(profile_face)
        profiles.append(
            {
                "name": name,
                "role": role_type,
                "path": str(image_file),
                "orb_descriptor": descriptor,
            }
        )
    return profiles


def save_registered_profile(name: str, role_label: str, uploaded_file) -> bool:
    if not name or not uploaded_file:
        return False
    prefix = "Lost" if "Lost" in role_label else "Member"
    safe_name = "_".join(name.split())
    destination = REG_DIR / f"{prefix}_{safe_name}.jpg"
    with Image.open(uploaded_file) as img:
        img = img.convert("RGB")
        img.save(destination)
    get_registered_profiles.clear()
    return True


@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def compute_face_histogram(image_bgr) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist)
    return hist

@st.cache_resource
def create_orb_detector():
    return cv2.ORB_create(nfeatures=500)


def compute_orb_descriptors(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = create_orb_detector()
    _, descriptors = orb.detectAndCompute(gray, None)
    return descriptors


def compare_orb_descriptors(left_desc, right_desc):
    if left_desc is None or right_desc is None:
        return float("inf"), 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(left_desc, right_desc)
    if not matches:
        return float("inf"), 0
    matches = sorted(matches, key=lambda m: m.distance)
    top_matches = matches[:30]
    avg_distance = sum(m.distance for m in top_matches) / len(top_matches)
    return avg_distance, len(matches)


def extract_faces(frame):
    face_cascade = load_face_cascade()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(48, 48))
    return faces


def extract_face_crop(image_bgr):
    faces = extract_faces(image_bgr)
    if len(faces) == 0:
        return image_bgr
    x, y, w, h = faces[0]
    return image_bgr[y : y + h, x : x + w]


def standardize_face(image_bgr, size=(224, 224)) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    face = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(face, cv2.COLOR_RGB2BGR)


def fallback_identity_match(frame, profiles: list[dict]) -> tuple[bool, str, str]:
    if len(profiles) == 0:
        return False, "", ""
    best_name = ""
    best_role = ""
    best_score = float("inf")
    frame_face = standardize_face(frame)
    frame_descriptor = compute_orb_descriptors(frame_face)
    for profile in profiles:
        try:
            profile_descriptor = profile.get("orb_descriptor")
            score, matches = compare_orb_descriptors(frame_descriptor, profile_descriptor)
            if matches >= 12 and score < best_score:
                best_score = score
                best_name = profile["name"]
                best_role = profile["role"]
        except Exception:
            continue
    if best_name and best_score < 60:
        return True, best_name, best_role
    return False, "", ""


def deepface_identity_match(frame, profiles: list[dict]) -> tuple[bool, str, str]:
    if DeepFace is None or len(profiles) == 0:
        return False, "", ""
    frame_face = standardize_face(frame)
    temp_frame_path = TEMP_FRAME_PATH
    cv2.imwrite(str(temp_frame_path), frame_face)
    try:
        results = DeepFace.find(
            img_path=str(temp_frame_path),
            db_path=str(REG_DIR),
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv",
            silent=True,
        )
    except Exception:
        return False, "", ""
    if results and len(results) > 0 and not results[0].empty:
        result_row = results[0].iloc[0]
        identity_path = str(result_row.get("identity", ""))
        identity_name = Path(identity_path).stem.split("_", 1)[-1]
        role_type = "Lost" if Path(identity_path).stem.startswith("Lost") else "Member"
        return True, identity_name, role_type
    return False, "", ""


def capture_match_snapshot(face_roi, name: str, role: str, detection_summary: str):
    if face_roi is None or face_roi.size == 0:
        return None
    snapshot = face_roi.copy()
    color = (0, 255, 0) if role == "Member" else (0, 0, 255)
    cv2.rectangle(snapshot, (0, 0), (snapshot.shape[1] - 1, snapshot.shape[0] - 1), color, 2)
    cv2.putText(
        snapshot,
        f"{role}: {name}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)


def match_identity(frame, profiles: list[dict]) -> tuple[bool, str, str]:
    if DeepFace is not None:
        try:
            matched, name, role = deepface_identity_match(frame, profiles)
            if matched:
                return matched, name, role
        except Exception:
            pass
    return fallback_identity_match(frame, profiles)


def check_weapon_contours(frame) -> tuple[bool, list[tuple[int, int, int, int]]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    threat_found = False
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 900 < area < 18000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / max(h, 1)
            if aspect_ratio > 2.5 or aspect_ratio < 0.35:
                boxes.append((x, y, w, h))
                threat_found = True
    return threat_found, boxes


def process_frame(frame, mode: str, profiles: list[dict]) -> tuple[np.ndarray, str]:
    annotated_frame = frame.copy()
    detection_summary = "No detections yet"
    faces = extract_faces(frame)

    if "3. Threat" in mode:
        threat_detected, boxes = check_weapon_contours(annotated_frame)
        if threat_detected:
            for x, y, w, h in boxes:
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    annotated_frame,
                    "THREAT ALERT",
                    (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            detection_summary = "Threat contour heuristic alert"
            st.session_state.last_status = detection_summary
            return annotated_frame, detection_summary

    if profiles and len(faces) > 0:
        st.session_state.last_status = "Matching... please wait"
        face_found = False
        for (x, y, w, h) in faces:
            face_roi = frame[y : y + h, x : x + w]
            matched, name, role = match_identity(face_roi, profiles)
            if matched:
                face_found = True
                color = (0, 255, 0) if role == "Member" else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{role}: {name}",
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                snapshot_image = capture_match_snapshot(face_roi, name, role, detection_summary)
                st.session_state.last_match_snapshot = snapshot_image
                st.session_state.last_match_caption = f"{role}: {name}"
                if "1. Lost Person" in mode and role == "Lost":
                    detection_summary = f"Lost person target located: {name}"
                elif "2. Member Attendance" in mode and role == "Member":
                    detection_summary = f"Known member acknowledged: {name}"
                else:
                    detection_summary = f"Identity matched: {name}"
                break
        if not face_found and len(faces) > 0:
            detection_summary = "Face detected but no known match"
            st.session_state.last_status = detection_summary

    if detection_summary and detection_summary != "No detections yet":
        st.session_state.last_status = detection_summary

    return annotated_frame, detection_summary


def render_sidebar() -> None:
    st.sidebar.title("🔐 N-ONE Security Gate")
    st.sidebar.caption("RBAC access control for surveillance operations")

    if st.session_state.authenticated:
        if st.sidebar.button("🔒 Logout System", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.role = "Guest"
            st.session_state.streaming = False
            st.rerun()
        st.sidebar.markdown(f"**Current Role:** {st.session_state.role}")
    else:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login", use_container_width=True):
            if username == "admin" and password == "admin123":
                st.session_state.authenticated = True
                st.session_state.role = "Administrator"
                st.session_state.streaming = False
                st.rerun()
            elif username == "operator" and password == "op123":
                st.session_state.authenticated = True
                st.session_state.role = "Operator"
                st.session_state.streaming = False
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials. Try admin/admin123 or operator/op123.")
        st.sidebar.warning("Authentication required to start surveillance.")

    st.sidebar.markdown("---")
    if st.session_state.role == "Administrator":
        st.sidebar.header("👤 Registration Module")
        reg_name = st.sidebar.text_input("Full Name / Identifier", key="reg_name")
        reg_role = st.sidebar.selectbox(
            "Classification Role",
            ["Lost Person / Victim", "Registered Member / Staff"],
            key="reg_role",
        )
        uploaded_photo = st.sidebar.file_uploader("Upload Clear Face Photo", type=["jpg", "jpeg", "png"], key="reg_photo")
        if st.sidebar.button("💾 Save Profile", use_container_width=True):
            if save_registered_profile(reg_name, reg_role, uploaded_photo):
                st.sidebar.success(f"Registered profile for {reg_name}")
                st.session_state.last_status = "Profile saved"
            else:
                st.sidebar.error("Provide both a name and a photo before saving.")
        if st.sidebar.button("🧹 Clear Registered Profiles", use_container_width=True):
            clear_registered_profiles()
            st.sidebar.success("All registered profiles deleted.")
            st.session_state.last_status = "Registered profiles cleared"
        if st.sidebar.button("📄 Reset Audit Log", use_container_width=True):
            clear_audit_log()
            st.sidebar.success("Audit log reset.")
            st.session_state.last_status = "Audit log reset"
    else:
        st.sidebar.info("Registration is restricted to administrators.")


def render_main_ui() -> None:
    st.title("🎯 N-ONE COMMAND CENTER")
    st.markdown("### Mission Control for AI Surveillance, Target Tracking, and Threat Detection")
    st.markdown("---")

    st.subheader("⚙️ Operational Console")
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        mode = st.selectbox(
            "Select Active Surveillance Mode",
            [
                "1. Lost Person Search (Target Alert Mode)",
                "2. Member Attendance Logger (Known vs Unknown)",
                "3. Threat & Weapon Detection Mode",
            ],
            key="selected_mode",
        )
    with col_ctrl2:
        source_type = st.radio(
            "Select Video Input Source",
            ["Laptop Webcam", "Recorded Video File", "IP Camera Stream"],
            horizontal=True,
            key="input_source",
        )

    video_target = None
    uploaded_video = None
    rtsp_url = ""
    if source_type == "Laptop Webcam":
        video_target = 0
    elif source_type == "Recorded Video File":
        uploaded_video = st.file_uploader("Upload Video File (.mp4 / .avi / .mov)", type=["mp4", "avi", "mov"], key="video_upload")
        if uploaded_video is not None:
            TEMP_VIDEO_PATH.write_bytes(uploaded_video.getbuffer())
            video_target = str(TEMP_VIDEO_PATH)
            st.markdown("**Uploaded video preview**")
            st.video(str(TEMP_VIDEO_PATH))
    elif source_type == "IP Camera Stream":
        rtsp_url = st.text_input(
            "Enter RTSP / HTTP Camera Stream URL",
            value="",
            placeholder="rtsp://admin:12345@192.168.1.100:554/h264Preview_01_main",
            key="rtsp_url",
        )
        if rtsp_url:
            video_target = rtsp_url

    st.markdown("---")

    profiles = get_registered_profiles()
    log_df = pd.read_csv(CSV_LOG_PATH) if CSV_LOG_PATH.exists() else pd.DataFrame(columns=["Timestamp", "Mode", "Subject_Name", "Role", "Event_Status"])
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Active RBAC Role", st.session_state.role)
    col_metric2.metric("Registered Profiles", len(profiles))
    col_metric3.metric("Audit Log Entries", len(log_df))
    with st.expander("Registered profile details", expanded=False):
        if profiles:
            for profile in profiles:
                st.write(f"- **{profile['role']}**: {profile['name']}")
        else:
            st.info("No registered faces available. Add profiles from the admin sidebar.")

    st.markdown("---")
    button_label = "🟢 Surveillance: On" if st.session_state.streaming else "🔴 Surveillance: Off"
    if st.button(button_label, use_container_width=True):
        st.session_state.streaming = not st.session_state.streaming
        st.session_state.last_status = "Streaming started" if st.session_state.streaming else "Streaming stopped"
        st.rerun()

    st.markdown("---")
    status_area = st.empty()
    status_text = st.session_state.last_status or "Idle"
    if "Matching" in status_text:
        status_area.info(status_text)
    elif "Threat" in status_text or "alert" in status_text.lower():
        status_area.error(status_text)
    elif status_text in {"Streaming started", "Streaming stopped", "Idle", "Profile saved", "Registered profiles cleared", "Audit log reset"}:
        status_area.info(status_text)
    else:
        status_area.success(status_text)

    with st.expander("📝 Last Detection Details", expanded=True):
        st.write(st.session_state.last_detection_details)

    frame_placeholder = st.empty()
    snapshot_placeholder = st.empty()

    if st.session_state.last_frame is None:
        frame_placeholder.info("Camera feed will appear here after the stream starts.")
    else:
        frame_placeholder.image(st.session_state.last_frame, channels="RGB", use_container_width=True)

    if st.session_state.last_match_snapshot is not None:
        snapshot_placeholder.markdown("### 📸 Match Snapshot")
        snapshot_placeholder.image(st.session_state.last_match_snapshot, caption=st.session_state.last_match_caption, channels="RGB", use_container_width=True)

    if st.session_state.authenticated and st.session_state.streaming and video_target is not None:
        cap = cv2.VideoCapture(video_target)
        if not cap.isOpened():
            st.error("Unable to open the requested source. Check the URL, permissions, or webcam connection.")
            st.session_state.streaming = False
        else:
            while st.session_state.streaming and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.session_state.streaming = False
                    st.session_state.last_status = "Stream disconnected"
                    break

                annotated_frame, detection_summary = process_frame(frame, mode, profiles)
                st.session_state.last_detection = detection_summary
                st.session_state.last_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(st.session_state.last_frame, channels="RGB", use_container_width=True)

                if st.session_state.last_match_snapshot is not None:
                    snapshot_placeholder.markdown("### 📸 Match Snapshot")
                    snapshot_placeholder.image(st.session_state.last_match_snapshot, caption=st.session_state.last_match_caption, channels="RGB", use_container_width=True)

                event_key = f"{mode}|{detection_summary}"
                if detection_summary and event_key != st.session_state.last_logged_event:
                    if "1. Lost Person" in mode and "Lost person target located" in detection_summary:
                        log_event(mode, detection_summary.split(":", 1)[-1].strip(), "Lost Victim", "Target Located")
                        st.session_state.last_logged_event = event_key
                    elif "2. Member Attendance" in mode and "Known member acknowledged" in detection_summary:
                        log_event(mode, detection_summary.split(":", 1)[-1].strip(), "Staff/Member", "Checked In")
                        st.session_state.last_logged_event = event_key
                    elif "3. Threat" in mode and "Threat" in detection_summary:
                        log_event(mode, "Unknown Weapon", "Threat", "Weapon Pattern Detected")
                        st.session_state.last_logged_event = event_key

                time.sleep(0.08)
            cap.release()

    st.markdown("---")
    st.subheader("📊 Structured Audit & Detection Logs")
    if CSV_LOG_PATH.exists():
        log_df = pd.read_csv(CSV_LOG_PATH)
        st.dataframe(log_df, use_container_width=True)
        st.download_button(
            label="📥 Download Audit Logs (CSV)",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="n_one_surveillance_logs.csv",
            mime="text/csv",
        )

    st.caption(f"Status: {st.session_state.last_status} | {st.session_state.last_detection}")


def main() -> None:
    initialize_directories()
    initialize_log_file()
    configure_session_state()
    render_sidebar()
    if not st.session_state.authenticated:
        st.warning("🔒 System locked. Authenticate from the sidebar to access the surveillance console.")
        return
    render_main_ui()


if __name__ == "__main__":
    main()