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
UNKNOWN_DIR = ROOT_DIR / "unknown_faces"
UNKNOWN_DB_PATH = ROOT_DIR / "unknown_person_db.csv"
UNKNOWN_SIGHTING_LOG_PATH = ROOT_DIR / "unknown_sighting_log.csv"

# In-memory cache for face encodings and tracking
KNOWN_FACE_ENCODINGS = []
UNKNOWN_FACE_ENCODINGS = []
FACE_CANDIDATES = {}  # { 'encoding': [timestamp, sightings] }
MAX_CANDIDATE_AGE = 5  # seconds
MIN_CANDIDATE_SIGHTINGS = 3 # times seen


def initialize_directories() -> None:
    REG_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    UNKNOWN_DIR.mkdir(exist_ok=True)


def initialize_log_file() -> None:
    if not CSV_LOG_PATH.exists():
        pd.DataFrame(columns=["Timestamp", "Mode", "Subject_Name", "Role", "Event_Status"]).to_csv(
            CSV_LOG_PATH,
            index=False,
        )
    if not UNKNOWN_DB_PATH.exists():
        pd.DataFrame(
            columns=["unknown_id", "image_path", "first_seen_timestamp", "last_seen_timestamp", "last_known_location", "assigned_name"]
        ).to_csv(UNKNOWN_DB_PATH, index=False)
    else:
        try:
            df = pd.read_csv(UNKNOWN_DB_PATH)
            if "assigned_name" not in df.columns:
                df["assigned_name"] = ""
                df.to_csv(UNKNOWN_DB_PATH, index=False)
        except pd.errors.EmptyDataError:
             pd.DataFrame(
                columns=["unknown_id", "image_path", "first_seen_timestamp", "last_seen_timestamp", "last_known_location", "assigned_name"]
            ).to_csv(UNKNOWN_DB_PATH, index=False)

    if not UNKNOWN_SIGHTING_LOG_PATH.exists():
        pd.DataFrame(columns=["sighting_id", "unknown_id", "timestamp", "location"]).to_csv(
            UNKNOWN_SIGHTING_LOG_PATH, index=False
        )


def get_next_unknown_id() -> str:
    if not UNKNOWN_DB_PATH.exists() or pd.read_csv(UNKNOWN_DB_PATH).empty:
        return "unknown_001"
    db_df = pd.read_csv(UNKNOWN_DB_PATH)
    if db_df.empty:
        return "unknown_001"
    last_id = db_df["unknown_id"].max()
    if not isinstance(last_id, str):
         return "unknown_001"
    last_num = int(last_id.split("_")[-1])
    return f"unknown_{last_num + 1:03d}"


@st.cache_resource
def load_all_face_encodings():
    """Loads all known and unknown face encodings into memory."""
    if DeepFace is None:
        return

    # Load known faces
    KNOWN_FACE_ENCODINGS.clear()
    for img_path in REG_DIR.glob("*.jpg"):
        try:
            name = img_path.stem.split("_", 1)[-1]
            role = "Lost" if img_path.stem.startswith("Lost") else "Member"
            embedding = DeepFace.represent(
                img_path=str(img_path), model_name="Facenet", enforce_detection=False
            )
            if embedding and len(embedding) > 0:
                 KNOWN_FACE_ENCODINGS.append({"name": name, "role": role, "encoding": embedding[0]["embedding"]})
        except Exception as e:
            st.warning(f"Could not process {img_path.name}: {e}")

    # Load unknown faces
    UNKNOWN_FACE_ENCODINGS.clear()
    for img_path in UNKNOWN_DIR.glob("*.jpg"):
        try:
            unknown_id = img_path.stem
            embedding = DeepFace.represent(
                img_path=str(img_path), model_name="Facenet", enforce_detection=False
            )
            if embedding and len(embedding) > 0:
                UNKNOWN_FACE_ENCODINGS.append({"id": unknown_id, "encoding": embedding[0]["embedding"]})
        except Exception:
            # st.warning(f"Could not load encoding for {img_path.name}: {e}")
            pass # Avoid spamming warnings if an image is corrupted


def find_face_in_cache(face_encoding: list, cache: list, tolerance=0.40) -> dict | None:
    """Finds a face in a given cache of encodings."""
    for entry in cache:
        distance = np.linalg.norm(np.array(face_encoding) - np.array(entry["encoding"]))
        if distance < tolerance:
            return entry
    return None


def register_new_unknown(face_roi: np.ndarray, face_encoding: list, location: str) -> str:
    """Registers a new unknown person and updates the in-memory cache."""
    unknown_id = get_next_unknown_id()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    image_path = UNKNOWN_DIR / f"{unknown_id}.jpg"
    cv2.imwrite(str(image_path), face_roi)

    new_person_df = pd.DataFrame(
        [[unknown_id, str(image_path), timestamp, timestamp, location, ""]],
        columns=["unknown_id", "image_path", "first_seen_timestamp", "last_seen_timestamp", "last_known_location", "assigned_name"],
    )
    header = not UNKNOWN_DB_PATH.exists() or UNKNOWN_DB_PATH.stat().st_size == 0
    new_person_df.to_csv(UNKNOWN_DB_PATH, mode="a", header=header, index=False)

    log_sighting(unknown_id, timestamp, location)

    # Update in-memory cache
    UNKNOWN_FACE_ENCODINGS.append({"id": unknown_id, "encoding": face_encoding})
    
    return unknown_id


def log_sighting(unknown_id: str, timestamp: str, location: str) -> None:
    sighting_df = pd.read_csv(UNKNOWN_SIGHTING_LOG_PATH)
    sighting_id = len(sighting_df) + 1
    new_sighting_df = pd.DataFrame(
        [[sighting_id, unknown_id, timestamp, location]], columns=["sighting_id", "unknown_id", "timestamp", "location"]
    )
    new_sighting_df.to_csv(UNKNOWN_SIGHTING_LOG_PATH, mode="a", header=False, index=False)


def update_unknown_sighting(unknown_id: str, location: str) -> str:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        db_df = pd.read_csv(UNKNOWN_DB_PATH)
        db_df.loc[db_df["unknown_id"] == unknown_id, "last_seen_timestamp"] = timestamp
        db_df.loc[db_df["unknown_id"] == unknown_id, "last_known_location"] = location
        db_df.to_csv(UNKNOWN_DB_PATH, index=False)
        log_sighting(unknown_id, timestamp, location)
        last_seen = db_df.loc[db_df["unknown_id"] == unknown_id, "first_seen_timestamp"].iloc[0]
        return last_seen
    except (FileNotFoundError, pd.errors.EmptyDataError, IndexError):
        return "N/A"


def tag_unknown_person(unknown_id: str, assigned_name: str) -> bool:
    if not UNKNOWN_DB_PATH.exists():
        return False
    db_df = pd.read_csv(UNKNOWN_DB_PATH)
    if unknown_id not in db_df["unknown_id"].values:
        return False
    db_df.loc[db_df["unknown_id"] == unknown_id, "assigned_name"] = assigned_name
    db_df.to_csv(UNKNOWN_DB_PATH, index=False)
    return True


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
        "active_alerts": []
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_registered_profiles() -> None:
    if not REG_DIR.exists():
        return
    for image_file in REG_DIR.glob("*"):
        if image_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image_file.unlink()
    load_all_face_encodings.clear()


def clear_audit_log() -> None:
    pd.DataFrame(columns=["Timestamp", "Mode", "Subject_Name", "Role", "Event_Status"]).to_csv(
        CSV_LOG_PATH,
        index=False,
    )


def save_registered_profile(name: str, role_label: str, uploaded_file) -> bool:
    if not name or not uploaded_file:
        return False
    prefix = "Lost" if "Lost" in role_label else "Member"
    safe_name = "_".join(name.split())
    destination = REG_DIR / f"{prefix}_{safe_name}.jpg"
    with Image.open(uploaded_file) as img:
        img = img.convert("RGB")
        img.save(destination)
    load_all_face_encodings.clear()
    return True


@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def capture_match_snapshot(face_roi, name: str, role: str):
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


def process_frame(frame, mode: str) -> tuple[np.ndarray, str]:
    annotated_frame = frame.copy()
    detection_summary = "No detections"

    # --- Threat Detection (Always On) ---
    if "3. Threat" in mode:
        threat_detected, boxes = check_weapon_contours(annotated_frame)
        if threat_detected:
            for x, y, w, h in boxes:
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "THREAT ALERT", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            detection_summary = "Threat contour heuristic alert"
            st.session_state.last_status = detection_summary
            alert_id = f"threat-{time.time()}"
            new_alert = {"id": alert_id, "type": "Threat", "message": "Potential weapon detected."}
            if not any(a['type'] == 'Threat' for a in st.session_state.active_alerts):
                 st.session_state.active_alerts.append(new_alert)
            return annotated_frame, detection_summary

    # --- Face Detection & Recognition ---
    try:
        face_objs = DeepFace.represent(frame, enforce_detection=False, detector_backend='opencv')
    except Exception as e:
        # This can happen if no faces are found and deepface errors out
        return annotated_frame, "No faces detected in frame."

    if not face_objs:
        return annotated_frame, "No faces detected"

    st.session_state.last_status = f"Found {len(face_objs)} face(s), matching..."
    
    face_found_in_frame = False
    for face_obj in face_objs:
        x, y, w, h = face_obj['facial_area'].values()
        face_roi = frame[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue
            
        face_encoding = face_obj["embedding"]
        face_found_in_frame = True

        # 1. Check against KNOWN faces
        known_match = find_face_in_cache(face_encoding, KNOWN_FACE_ENCODINGS)
        if known_match:
            name, role = known_match["name"], known_match["role"]
            color = (0, 255, 0) if role == "Member" else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated_frame, f"{role}: {name}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            detection_summary = f"Matched: {role} {name}"
            
            st.session_state.last_match_snapshot = capture_match_snapshot(face_roi, name, role)
            st.session_state.last_match_caption = f"{role}: {name}"
            if "1. Lost Person" in mode and role == "Lost":
                alert_id = f"lost-{name}-{time.time()}"
                new_alert = {"id": alert_id, "type": "Lost Person", "message": f"Lost person '{name}' has been located."}
                if not any(a.get('message') == new_alert.get('message') for a in st.session_state.active_alerts):
                    st.session_state.active_alerts.append(new_alert)
            continue

        # 2. Check against UNKNOWN faces (only in attendance mode)
        if "2. Member Attendance" in mode:
            unknown_match = find_face_in_cache(face_encoding, UNKNOWN_FACE_ENCODINGS)
            if unknown_match:
                unknown_id = unknown_match["id"]
                first_seen = update_unknown_sighting(unknown_id, "Main Feed")
                detection_summary = f"Re-identified: ID {unknown_id}"
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
                cv2.putText(annotated_frame, f"ID: {unknown_id}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(annotated_frame, f"First Seen: {first_seen}", (x, max(0, y - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                continue

            # 3. Handle as a new CANDIDATE face
            candidate_match_key = None
            # Find a matching candidate in the temporary buffer
            for candidate_encoding_tuple, (ts, sightings) in FACE_CANDIDATES.items():
                distance = np.linalg.norm(np.array(face_encoding) - np.array(candidate_encoding_tuple))
                if distance < 0.5: # Use a slightly more lenient tolerance for candidates
                    candidate_match_key = candidate_encoding_tuple
                    break
            
            current_time = time.time()
            if candidate_match_key:
                # If a candidate is found, update its timestamp and sightings count
                FACE_CANDIDATES[candidate_match_key][0] = current_time
                FACE_CANDIDATES[candidate_match_key][1] += 1
                sightings = FACE_CANDIDATES[candidate_match_key][1]
                detection_summary = f"Tracking candidate... ({sightings}/{MIN_CANDIDATE_SIGHTINGS})"
                
                # If candidate seen enough times, promote to a new unknown person
                if sightings >= MIN_CANDIDATE_SIGHTINGS:
                    unknown_id = register_new_unknown(face_roi, list(candidate_match_key), "Main Feed")
                    detection_summary = f"New Unknown Registered: {unknown_id}"
                    del FACE_CANDIDATES[candidate_match_key] # Remove from candidates
            else:
                # If no matching candidate, add this new face as a candidate
                FACE_CANDIDATES[tuple(face_encoding)] = [current_time, 1]
                detection_summary = "New face candidate spotted."

            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
            cv2.putText(annotated_frame, "Candidate", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Clean up old candidates that haven't been seen recently
    current_time = time.time()
    expired_candidates = [key for key, (ts, _) in FACE_CANDIDATES.items() if current_time - ts > MAX_CANDIDATE_AGE]
    for key in expired_candidates:
        if key in FACE_CANDIDATES:
             del FACE_CANDIDATES[key]

    if face_found_in_frame:
        st.session_state.last_status = detection_summary
    
    return annotated_frame, detection_summary


def render_sidebar() -> None:
    st.sidebar.title("🔐 N-ONE Security Gate")
    st.sidebar.caption("RBAC access control for surveillance operations")
    # ... (rest of sidebar remains the same)
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
                load_all_face_encodings() # Reload cache
            else:
                st.sidebar.error("Provide both a name and a photo before saving.")
        if st.sidebar.button("🧹 Clear Registered Profiles", use_container_width=True):
            clear_registered_profiles()
            st.sidebar.success("All registered profiles deleted.")
            st.session_state.last_status = "Registered profiles cleared"
            load_all_face_encodings() # Reload cache
        if st.sidebar.button("📄 Reset Audit Log", use_container_width=True):
            clear_audit_log()
            st.sidebar.success("Audit log reset.")
            st.session_state.last_status = "Audit log reset"
    else:
        st.sidebar.info("Registration is restricted to administrators.")


def render_main_ui() -> None:
    st.title("🎯 N-ONE COMMAND CENTER")
    st.markdown("### Mission Control for AI Surveillance, Target Tracking, and Threat Detection")

    if st.session_state.active_alerts:
        st.markdown("<h2 style='color: #ef4444;'>🚨 CRITICAL ALERTS</h2>", unsafe_allow_html=True)
        alerts_to_keep = []
        for alert in st.session_state.active_alerts:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.error(f"**{alert['type']}:** {alert['message']}")
            with col2:
                if st.button(f"Dismiss {alert['id'][-4:]}", key=f"dismiss_{alert['id']}"):
                    pass
                else:
                    alerts_to_keep.append(alert)
        st.session_state.active_alerts = alerts_to_keep
        st.markdown("---")
        
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
    if source_type == "Laptop Webcam":
        video_target = 0
    elif source_type == "Recorded Video File":
        uploaded_video = st.file_uploader("Upload Video File (.mp4 / .avi / .mov)", type=["mp4", "avi", "mov"], key="video_upload")
        if uploaded_video is not None:
            TEMP_VIDEO_PATH.write_bytes(uploaded_video.getbuffer())
            video_target = str(TEMP_VIDEO_PATH)
    elif source_type == "IP Camera Stream":
        rtsp_url = st.text_input("Enter RTSP / HTTP Camera Stream URL", key="rtsp_url")
        if rtsp_url:
            video_target = rtsp_url

    st.markdown("---")

    log_df = pd.read_csv(CSV_LOG_PATH) if CSV_LOG_PATH.exists() else pd.DataFrame()
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Active RBAC Role", st.session_state.role)
    col_metric2.metric("Registered Profiles", len(KNOWN_FACE_ENCODINGS))
    col_metric3.metric("Log Entries", len(log_df))

    st.markdown("---")
    button_label = "🟢 Surveillance: On" if st.session_state.streaming else "🔴 Surveillance: Off"
    if st.button(button_label, use_container_width=True):
        st.session_state.streaming = not st.session_state.streaming
        st.session_state.last_status = "Streaming started" if st.session_state.streaming else "Streaming stopped"
        st.rerun()

    st.markdown("---")
    status_area = st.empty()
    status_text = st.session_state.get('last_status', "Idle")
    if "Matching" in status_text or "candidate" in status_text:
        status_area.info(status_text)
    elif "Threat" in status_text or "alert" in status_text.lower():
        status_area.error(status_text)
    else:
        status_area.success(status_text)
    
    frame_placeholder = st.empty()
    if st.session_state.last_frame is None:
        frame_placeholder.info("Camera feed will appear here after the stream starts.")
    else:
        frame_placeholder.image(st.session_state.last_frame, channels="RGB", use_container_width=True)

    if st.session_state.authenticated and st.session_state.streaming and video_target is not None:
        cap = cv2.VideoCapture(video_target)
        if not cap.isOpened():
            st.error("Unable to open the requested source.")
            st.session_state.streaming = False
        else:
            while st.session_state.streaming and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.session_state.streaming = False
                    st.session_state.last_status = "Stream disconnected"
                    break

                annotated_frame, detection_summary = process_frame(frame, mode)
                st.session_state.last_detection = detection_summary
                st.session_state.last_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(st.session_state.last_frame, channels="RGB", use_container_width=True)

                time.sleep(0.08) # Control frame rate
            cap.release()

    st.markdown("---")
    st.subheader("📊 Structured Audit & Detection Logs")
    # ... (rest of UI remains the same)
    if CSV_LOG_PATH.exists():
        log_df = pd.read_csv(CSV_LOG_PATH)

        with st.expander("🔍 Filter Audit Logs", expanded=False):
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                log_start_date = st.date_input("Start date", None, key="audit_start_date")
                log_end_date = st.date_input("End date", None, key="audit_end_date")
            with col_filter2:
                search_term = st.text_input("Search by keyword (name, role, event)", key="audit_search")

        filtered_log_df = log_df.copy()
        if log_start_date and log_end_date:
            filtered_log_df["Timestamp"] = pd.to_datetime(filtered_log_df["Timestamp"])
            start_date_ts = pd.to_datetime(log_start_date)
            end_date_ts = pd.to_datetime(log_end_date)
            filtered_log_df = filtered_log_df[(filtered_log_df['Timestamp'].dt.date >= start_date_ts.date()) & (filtered_log_df['Timestamp'].dt.date <= end_date_ts.date())]

        if search_term:
            filtered_log_df = filtered_log_df[
                filtered_log_df["Subject_Name"].str.contains(search_term, case=False, na=False) |
                filtered_log_df["Role"].str.contains(search_term, case=False, na=False) |
                filtered_log_df["Event_Status"].str.contains(search_term, case=False, na=False)
            ]

        st.dataframe(filtered_log_df, use_container_width=True)
        st.download_button(
            label="📥 Download Filtered Audit Logs (CSV)",
            data=filtered_log_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_n_one_surveillance_logs.csv",
            mime="text/csv",
        )

    st.subheader("🕵️ Unknown & Re-Identified Person Logs")

    with st.expander("🔍 Filter Unknown Person Logs", expanded=False):
        col_ufilter1, col_ufilter2 = st.columns(2)
        with col_ufilter1:
            unknown_search_term = st.text_input("Search by Unknown ID", key="unknown_search")
        with col_ufilter2:
            sighting_start_date = st.date_input("Sighting start date", None, key="sighting_start_date")
            sighting_end_date = st.date_input("Sighting end date", None, key="sighting_end_date")


    col_unknown1, col_unknown2 = st.columns(2)
    with col_unknown1:
        st.markdown("**Unknown Persons Database**")
        if UNKNOWN_DB_PATH.exists():
            try:
                unknown_df = pd.read_csv(UNKNOWN_DB_PATH)
                if not unknown_df.empty:
                    filtered_unknown_df = unknown_df.copy()

                    if sighting_start_date and sighting_end_date:
                        filtered_unknown_df["last_seen_timestamp"] = pd.to_datetime(
                            filtered_unknown_df["last_seen_timestamp"]
                        )
                        start_date_ts = pd.to_datetime(sighting_start_date)
                        end_date_ts = pd.to_datetime(sighting_end_date)
                        filtered_unknown_df = filtered_unknown_df[
                            (filtered_unknown_df["last_seen_timestamp"].dt.date >= start_date_ts.date())
                            & (filtered_unknown_df["last_seen_timestamp"].dt.date <= end_date_ts.date())
                        ]

                    if unknown_search_term:
                        filtered_unknown_df = filtered_unknown_df[
                            filtered_unknown_df["unknown_id"].str.contains(unknown_search_term, case=False, na=False)
                        ]

                    st.dataframe(filtered_unknown_df, use_container_width=True)
                else:
                    st.info("Unknown person database is currently empty.")
            except pd.errors.EmptyDataError:
                st.info("Unknown person database is currently empty.")
            
    with col_unknown2:
        st.markdown("**Sighting Log for Unknowns**")
        if UNKNOWN_SIGHTING_LOG_PATH.exists():
            sighting_df = pd.read_csv(UNKNOWN_SIGHTING_LOG_PATH)
            filtered_sighting_df = sighting_df.copy()
            
            if sighting_start_date and sighting_end_date:
                filtered_sighting_df["timestamp"] = pd.to_datetime(filtered_sighting_df["timestamp"])
                start_date_ts = pd.to_datetime(sighting_start_date)
                end_date_ts = pd.to_datetime(sighting_end_date)
                filtered_sighting_df = filtered_sighting_df[(filtered_sighting_df['timestamp'].dt.date >= start_date_ts.date()) & (filtered_sighting_df['timestamp'].dt.date <= end_date_ts.date())]

            if unknown_search_term:
                filtered_sighting_df = filtered_sighting_df[filtered_sighting_df["unknown_id"].str.contains(unknown_search_term, case=False, na=False)]
                
            st.dataframe(filtered_sighting_df, use_container_width=True)

    if st.session_state.role == "Administrator":
        with st.expander("✏️ Tag an Unknown Person", expanded=False):
            untagged_ids = []
            if UNKNOWN_DB_PATH.exists():
                try:
                    unknown_df = pd.read_csv(UNKNOWN_DB_PATH)
                    untagged_df = unknown_df[unknown_df["assigned_name"].isnull() | (unknown_df["assigned_name"] == "")]
                    if not untagged_df.empty:
                        untagged_ids = untagged_df["unknown_id"].tolist()
                except pd.errors.EmptyDataError:
                    pass

            if not untagged_ids:
                st.info("No untagged unknown persons available to label.")
            else:
                with st.form("tagging_form"):
                    tag_id = st.selectbox("Select Unknown ID to tag", options=untagged_ids)
                    tag_name = st.text_input("Assign a name/tag (e.g., 'Regular Courier')")
                    submitted = st.form_submit_button("Save Tag")
                    if submitted:
                        if tag_id and tag_name:
                            if tag_unknown_person(tag_id, tag_name):
                                st.success(f"Successfully tagged {tag_id} as '{tag_name}'.")
                                st.rerun()
                            else:
                                st.error(f"Failed to tag {tag_id}. Check if the ID exists.")
                        else:
                            st.warning("Please select an ID and provide a name.")

    with st.expander("🕵️‍♂️ Unknown Persons Gallery", expanded=False):
        render_unknown_gallery()

    st.caption(f"Status: {st.session_state.last_status} | {st.session_state.last_detection}")


def render_unknown_gallery():
    if not UNKNOWN_DB_PATH.exists() or pd.read_csv(UNKNOWN_DB_PATH).empty:
        st.info("No unknown persons have been logged yet.")
        return

    st.markdown("A visual log of all unique unknown individuals detected by the system.")
    unknown_df = pd.read_csv(UNKNOWN_DB_PATH)
    
    num_cols = 5  # Define number of columns for the gallery
    cols = st.columns(num_cols)
    
    for index, row in unknown_df.iterrows():
        col_index = index % num_cols
        with cols[col_index]:
            image_path_str = str(row["image_path"])
            if Path(image_path_str).exists():
                st.image(image_path_str, use_column_width=True)
                st.caption(f"ID: {row['unknown_id']}")
                st.caption(f"First Seen: {row['first_seen_timestamp']}")
            else:
                st.warning(f"ID: {row['unknown_id']}\n(Image not found)")


def main() -> None:
    initialize_directories()
    initialize_log_file()
    configure_session_state()
    load_all_face_encodings()
    render_sidebar()
    if not st.session_state.authenticated:
        st.warning("🔒 System locked. Authenticate from the sidebar to access the surveillance console.")
        return
    render_main_ui()


if __name__ == "__main__":
    main()