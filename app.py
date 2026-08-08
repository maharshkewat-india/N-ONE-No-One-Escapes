import hmac
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.spatial.distance import cosine
from streamlit.errors import StreamlitSecretNotFoundError

from deepface_adapter import DEEPFACE_IMPORT_ERROR, DeepFace


AUTH_SECRET_NAMES = (
    "ADMIN_USERNAME",
    "ADMIN_PASSWORD",
    "OPERATOR_USERNAME",
    "OPERATOR_PASSWORD",
)
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 60


def _read_secret(name: str) -> str | None:
    """Read a secret from the process environment or Streamlit Secrets."""
    environment_value = os.getenv(name)
    if environment_value is not None:
        return environment_value

    try:
        secret_value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None

    return None if secret_value is None else str(secret_value)


def load_auth_credentials() -> dict[str, str]:
    """Load Admin and Operator credentials and fail closed when any are missing."""
    credentials = {name: _read_secret(name) for name in AUTH_SECRET_NAMES}
    missing = [name for name, value in credentials.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing required authentication secrets: " + ", ".join(missing)
        )
    return {name: value for name, value in credentials.items() if value is not None}

if DEEPFACE_IMPORT_ERROR:
    st.warning(
        "DeepFace runtime is not fully available in this environment. "
        f"The app will continue with a safe fallback. Error: {DEEPFACE_IMPORT_ERROR}"
    )

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

# --- Constants and Paths ---
ROOT_DIR = Path(__file__).resolve().parent
REG_DIR = ROOT_DIR / "registered_faces"
LOG_DIR = ROOT_DIR / "detection_logs"
UNKNOWN_DIR = ROOT_DIR / "unknown_faces"
CSV_LOG_PATH = LOG_DIR / "system_audit_logs.csv"
TEMP_VIDEO_PATH = ROOT_DIR / "temp_video_upload.mp4"
UNKNOWN_DB_PATH = ROOT_DIR / "unknown_person_db.csv"
UNKNOWN_SIGHTING_LOG_PATH = ROOT_DIR / "unknown_sighting_log.csv"

# DeepFace Model Configuration
DEEPFACE_MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "SFace"]
DEEPFACE_BACKENDS = ["opencv", "mtcnn", "retinaface", "mediapipe", "dlib", "ssd", "yolov8n", "yunet"]
DEEPFACE_METRICS = ["cosine", "euclidean", "euclidean_l2"]
ATTRIBUTE_TASKS = ["age", "gender", "emotion", "race"]

# Default Configuration
DEEPFACE_MODEL = "Facenet"
DEEPFACE_BACKEND = "opencv"
DEEPFACE_METRIC = "cosine"
COSINE_THRESHOLD = 0.40

# --- In-memory Cache ---
KNOWN_FACE_ENCODINGS = []


def initialize_directories() -> None:
    """Create necessary directories if they don't exist."""
    REG_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    UNKNOWN_DIR.mkdir(exist_ok=True)


def initialize_log_files() -> None:
    """Create log files with headers if they don't exist."""
    if not CSV_LOG_PATH.exists() or CSV_LOG_PATH.stat().st_size == 0:
        pd.DataFrame(columns=["Timestamp", "Mode", "Subject_ID", "Role", "Event_Type", "Details"]).to_csv(
            CSV_LOG_PATH, index=False
        )
    if not UNKNOWN_DB_PATH.exists() or UNKNOWN_DB_PATH.stat().st_size == 0:
        pd.DataFrame(
            columns=["unknown_id", "image_path", "first_seen_timestamp", "last_seen_timestamp", "last_known_location", "assigned_name"]
        ).to_csv(UNKNOWN_DB_PATH, index=False)
    if not UNKNOWN_SIGHTING_LOG_PATH.exists() or UNKNOWN_SIGHTING_LOG_PATH.stat().st_size == 0:
        pd.DataFrame(columns=["sighting_id", "unknown_id", "timestamp", "location"]).to_csv(
            UNKNOWN_SIGHTING_LOG_PATH, index=False
        )


def log_event(mode: str, subject_id: str, role: str, event_type: str, details: str = "") -> None:
    """Logs a single surveillance event to the main audit CSV."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    new_log_entry = pd.DataFrame(
        [[timestamp, mode, subject_id, role, event_type, details]],
        columns=["Timestamp", "Mode", "Subject_ID", "Role", "Event_Type", "Details"],
    )
    new_log_entry.to_csv(CSV_LOG_PATH, mode="a", header=False, index=False)
    st.session_state.last_logged_event = f"{timestamp} - {event_type}: {subject_id}"


def get_next_unknown_id() -> str:
    """Generates the next sequential ID for a new unknown person."""
    if not UNKNOWN_DB_PATH.exists() or pd.read_csv(UNKNOWN_DB_PATH).empty:
        return "unknown_001"
    db_df = pd.read_csv(UNKNOWN_DB_PATH)
    if db_df.empty:
        return "unknown_001"
    last_id = db_df["unknown_id"].max()
    if not isinstance(last_id, str) or not last_id.startswith("unknown_"):
        return "unknown_001"
    last_num = int(last_id.split("_")[-1])
    return f"unknown_{last_num + 1:03d}"


def get_registered_profile_count() -> int:
    """Return the number of saved profiles visible to the dashboard."""
    if KNOWN_FACE_ENCODINGS:
        return len(KNOWN_FACE_ENCODINGS)

    profile_files = [
        img_path
        for img_path in REG_DIR.glob("*.*")
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return len(profile_files)


@st.cache_resource
def load_known_face_encodings():
    """Loads all known face encodings from the registration directory into memory."""
    if DeepFace is None:
        return

    KNOWN_FACE_ENCODINGS.clear()
    for img_path in REG_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        name = img_path.stem.split("_", 1)[-1]
        role = "Lost" if img_path.stem.startswith("Lost") else "Member"
        embedding = None
        try:
            embedding_obj = DeepFace.represent(
                img_path=str(img_path), model_name=DEEPFACE_MODEL, enforce_detection=False
            )
            if embedding_obj and len(embedding_obj) > 0:
                embedding = embedding_obj[0].get("embedding")
        except Exception as e:
            st.warning(f"Could not process {img_path.name}: {e}")

        KNOWN_FACE_ENCODINGS.append({"name": name, "role": role, "encoding": embedding})


def find_face_in_known_cache(face_encoding: list) -> dict | None:
    """Finds a face in the in-memory cache of known encodings using cosine similarity."""
    for entry in KNOWN_FACE_ENCODINGS:
        if entry.get("encoding") is None:
            continue
        distance = cosine(np.array(face_encoding), np.array(entry["encoding"]))
        if distance < COSINE_THRESHOLD:
            return entry
    return None


def register_new_unknown(face_roi: np.ndarray, location: str, mode: str) -> str:
    """Registers a new unknown person, saves their image, logs the event, and updates the DB."""
    unknown_id = get_next_unknown_id()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    image_path = UNKNOWN_DIR / f"{unknown_id}.jpg"
    
    cv2.imwrite(str(image_path), face_roi)

    new_person_df = pd.DataFrame(
        [[unknown_id, str(image_path), timestamp, timestamp, location, ""]],
        columns=["unknown_id", "image_path", "first_seen_timestamp", "last_seen_timestamp", "last_known_location", "assigned_name"],
    )
    new_person_df.to_csv(UNKNOWN_DB_PATH, mode="a", header=False, index=False)

    log_sighting(unknown_id, timestamp, location)
    log_event(mode=mode, subject_id=unknown_id, role="Unknown", event_type="New Unknown Detected")
    
    # Re-build the representations file for the unknown directory
    if (UNKNOWN_DIR / "representations_facenet.pkl").exists():
        (UNKNOWN_DIR / "representations_facenet.pkl").unlink()
    DeepFace.find(img_path=str(image_path), db_path=str(UNKNOWN_DIR), model_name=DEEPFACE_MODEL, distance_metric=DEEPFACE_METRIC)

    return unknown_id


def log_sighting(unknown_id: str, timestamp: str, location: str) -> None:
    """Adds a record to the sighting log for an unknown person."""
    try:
        sighting_id = (pd.read_csv(UNKNOWN_SIGHTING_LOG_PATH)["sighting_id"].max() + 1)
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
        sighting_id = 1

    new_sighting_df = pd.DataFrame(
        [[sighting_id, unknown_id, timestamp, location]], columns=["sighting_id", "unknown_id", "timestamp", "location"]
    )
    new_sighting_df.to_csv(UNKNOWN_SIGHTING_LOG_PATH, mode="a", header=False, index=False)


def update_unknown_sighting(unknown_id: str, location: str, mode: str) -> None:
    """Updates the last seen timestamp/location for an unknown person and logs the event."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        db_df = pd.read_csv(UNKNOWN_DB_PATH)
        db_df.loc[db_df["unknown_id"] == unknown_id, "last_seen_timestamp"] = timestamp
        db_df.loc[db_df["unknown_id"] == unknown_id, "last_known_location"] = location
        db_df.to_csv(UNKNOWN_DB_PATH, index=False)
        log_sighting(unknown_id, timestamp, location)
        log_event(mode=mode, subject_id=unknown_id, role="Unknown", event_type="Re-identified")
    except (FileNotFoundError, pd.errors.EmptyDataError, IndexError):
        pass


def analyze_facial_attributes(frame: np.ndarray, face_objs: list) -> list[dict]:
    """Analyze facial attributes (age, gender, emotion, race) for detected faces."""
    attributes_list = []
    if DeepFace is None:
        return attributes_list
    
    try:
        analysis = DeepFace.analyze(
            img_path=frame, enforce_detection=False, silent=True
        )
        if isinstance(analysis, list):
            attributes_list = analysis
    except Exception as e:
        pass
    
    return attributes_list


def extract_and_align_faces(frame: np.ndarray) -> list[np.ndarray]:
    """Extract and align faces from the frame."""
    aligned_faces = []
    if DeepFace is None:
        return aligned_faces
    
    try:
        extracted = DeepFace.extract_faces(
            img_path=frame, enforce_detection=False, silent=True
        )
        if isinstance(extracted, list):
            aligned_faces = [face.get("face") for face in extracted if "face" in face]
    except Exception as e:
        pass
    
    return aligned_faces


def verify_face_pair(img1_path: str, img2_path: str, model: str = DEEPFACE_MODEL) -> dict:
    """Verify if two face images belong to the same person (1-to-1 matching)."""
    if DeepFace is None:
        return {"verified": False, "distance": 1.0, "threshold": 0.4}
    
    try:
        result = DeepFace.verify(
            img1_path=img1_path, img2_path=img2_path,
            model_name=model, enforce_detection=False, silent=True
        )
        return result
    except Exception as e:
        return {"verified": False, "distance": 1.0, "threshold": 0.4, "error": str(e)}


def detect_spoofing(frame: np.ndarray) -> dict:
    """Detect face spoofing/liveness attacks using anti-spoofing model."""
    if DeepFace is None:
        return {"is_real": True, "confidence": 0.0}
    
    try:
        result = DeepFace.detect_spoofing(
            img_path=frame, detector_backend="opencv", silent=True
        )
        return result
    except Exception as e:
        return {"is_real": True, "confidence": 0.0, "error": str(e)}


def get_face_embeddings(frame: np.ndarray, model: str = DEEPFACE_MODEL) -> list:
    """Get facial embeddings for faces in the frame."""
    if DeepFace is None:
        return []
    
    try:
        embeddings = DeepFace.represent(
            img_path=frame, model_name=model, enforce_detection=False, silent=True
        )
        return embeddings if isinstance(embeddings, list) else []
    except Exception as e:
        return []


def check_weapon_contours(frame) -> tuple[bool, list[tuple[int, int, int, int]]]:
    """Heuristic-based weapon contour detection."""
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


def process_frame(frame: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    """The core processing pipeline for each video frame."""
    annotated_frame = frame.copy()
    detection_summary = "Status: Idle"

    is_face_rec_mode = "Threat" not in mode
    
    # --- 1. Threat Detection (Exclusive Mode) ---
    if not is_face_rec_mode:
        threat_detected, boxes = check_weapon_contours(annotated_frame)
        if threat_detected:
            for x, y, w, h in boxes:
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "THREAT ALERT", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            detection_summary = "Threat contour heuristic alert!"
            log_event(mode=mode, subject_id="System", role="N/A", event_type="Threat Detected", details="Contour analysis matched threat profile.")
        else:
            detection_summary = "No threats detected."
        st.session_state.last_status = detection_summary
        return annotated_frame, detection_summary

    # --- 2. Face Detection & Recognition (For all other modes) ---
    try:
        face_objs = DeepFace.represent(
            img_path=frame, model_name=DEEPFACE_MODEL,
            enforce_detection=False, detector_backend='opencv'
        )
    except Exception as e:
        st.session_state.last_status = f"DeepFace Error: {e}"
        return annotated_frame, f"Error during face representation: {e}"

    if not face_objs:
        st.session_state.last_status = "No faces detected in frame."
        return annotated_frame, "No faces detected in frame."

    detection_summary = f"Found {len(face_objs)} face(s), analyzing..."
    
    for face_obj in face_objs:
        x, y, w, h = face_obj['facial_area'].values()
        face_roi = frame[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue
            
        face_encoding = face_obj["embedding"]

        # Step A: Check against the KNOWN faces in-memory cache
        known_match = find_face_in_known_cache(face_encoding)
        if known_match:
            name, role = known_match["name"], known_match["role"]
            color = (0, 255, 0) if role == "Member" else (255, 0, 0)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            label = f"{role}: {name}"
            cv2.putText(annotated_frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            detection_summary = f"Matched: {label}"
            log_event(mode=mode, subject_id=name, role=role, event_type="Known Face Match")
            continue

        # Step B: If not a known face, check against the UNKNOWN database on disk
        try:
            # Use DeepFace.find against the unknown faces directory. This is faster if DB is large.
            dfs = DeepFace.find(
                img_path=face_roi, db_path=str(UNKNOWN_DIR),
                model_name=DEEPFACE_MODEL, distance_metric=DEEPFACE_METRIC,
                enforce_detection=False, silent=True
            )
            if dfs and not dfs[0].empty:
                match_path = Path(dfs[0].iloc[0]['identity'])
                unknown_id = match_path.stem
                update_unknown_sighting(unknown_id, "Main Feed", mode)
                detection_summary = f"Re-identified: {unknown_id}"
                color = (255, 165, 0) # Orange
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated_frame, f"ID: {unknown_id}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                continue
        except Exception:
             # This can fail if UNKNOWN_DIR is empty, which is fine on first run.
             pass
        
        # Step C: If not known and not previously unknown, register as a NEW UNKNOWN
        unknown_id = register_new_unknown(face_roi, "Main Feed", mode)
        detection_summary = f"New Unknown Registered: {unknown_id}"
        color = (255, 255, 0) # Cyan
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated_frame, f"New ID: {unknown_id}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    st.session_state.last_status = detection_summary
    return annotated_frame, detection_summary

def configure_session_state() -> None:
    defaults = {
        "authenticated": False, "role": "Guest", "streaming": False,
        "last_frame": None, "last_status": "Idle",
        "last_detection": "No detections yet", "last_logged_event": "",
        "active_alerts": [],
        "login_attempts": 0, "login_locked_until": 0.0,
        "selected_model": DEEPFACE_MODEL,
        "selected_backend": DEEPFACE_BACKEND,
        "selected_metric": DEEPFACE_METRIC,
        "enable_attributes": True,
        "enable_spoofing_detection": False,
        "similarity_threshold": COSINE_THRESHOLD,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def save_registered_profile(name: str, role_label: str, uploaded_file) -> bool:
    """Save a face profile. This operation is available to Administrators only."""
    if not name or not uploaded_file:
        return False

    prefix = "Lost" if "Lost" in role_label else "Member"
    safe_name = "".join(
        character if character.isalnum() else "_" for character in name
    ).strip("_")
    if not safe_name:
        return False

    destination = REG_DIR / f"{prefix}_{safe_name}.jpg"
    with Image.open(uploaded_file) as image:
        image.convert("RGB").save(destination)

    load_known_face_encodings.clear()
    load_known_face_encodings()
    return True


def clear_audit_log() -> None:
    """Clear audit records. This operation is available to Administrators only."""
    if CSV_LOG_PATH.exists():
        CSV_LOG_PATH.unlink()
    initialize_log_files()


def clear_registered_profiles() -> None:
    """Clear registered face profiles. This operation is available to Administrators only."""
    for profile_path in REG_DIR.glob("*"):
        if profile_path.is_file():
            profile_path.unlink()
    load_known_face_encodings.clear()


def render_sidebar() -> None:
    st.sidebar.title("🔐 N-ONE Security Gate")
    st.sidebar.caption("RBAC access control for surveillance ops")

    try:
        credentials = load_auth_credentials()
    except RuntimeError:
        st.sidebar.error("Authentication is unavailable: required secrets are not configured.")
        st.sidebar.caption(
            "Configure ADMIN_USERNAME, ADMIN_PASSWORD, OPERATOR_USERNAME, and "
            "OPERATOR_PASSWORD in Streamlit Secrets "
            "or the process environment."
        )
        return
    
    if st.session_state.authenticated:
        if st.sidebar.button("🔒 Logout System", width='stretch'):
            st.session_state.streaming = False
            st.session_state.authenticated = False
            st.session_state.role = "Guest"
            st.rerun()
        st.sidebar.markdown(f"**Current Role:** {st.session_state.role}")
    else:
        st.sidebar.info("Login to access the command center.")
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login", width='stretch'):
            now = time.monotonic()
            locked_until = st.session_state.get("login_locked_until", 0.0)
            if now < locked_until:
                remaining = max(1, int(locked_until - now) + 1)
                st.sidebar.error(f"Too many failed attempts. Try again in {remaining} seconds.")
                return

            authenticated_role = None
            if (
                hmac.compare_digest(username, credentials["ADMIN_USERNAME"])
                and hmac.compare_digest(password, credentials["ADMIN_PASSWORD"])
            ):
                authenticated_role = "Administrator"
            elif (
                hmac.compare_digest(username, credentials["OPERATOR_USERNAME"])
                and hmac.compare_digest(password, credentials["OPERATOR_PASSWORD"])
            ):
                authenticated_role = "Operator"

            if authenticated_role is not None:
                st.session_state.login_attempts = 0
                st.session_state.login_locked_until = 0.0
                st.session_state.authenticated = True
                st.session_state.role = authenticated_role
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
                    st.session_state.login_locked_until = now + LOGIN_LOCKOUT_SECONDS
                    st.session_state.login_attempts = 0
                    st.sidebar.error("Too many failed attempts. Login is locked for 60 seconds.")
                else:
                    st.sidebar.error("Invalid credentials.")

    if st.session_state.role == "Administrator":
        st.sidebar.markdown("---")
        st.sidebar.header("Administrator controls")
        with st.sidebar.form("registration_form"):
            reg_name = st.text_input("Full Name / Identifier")
            reg_role = st.selectbox(
                "Classification Role",
                ["Lost Person / Victim", "Registered Member / Staff"],
            )
            uploaded_photo = st.file_uploader(
                "Upload Clear Face Photo", type=["jpg", "jpeg", "png"]
            )
            submitted = st.form_submit_button("Save Profile")
            if submitted:
                if save_registered_profile(reg_name, reg_role, uploaded_photo):
                    st.sidebar.success(f"Registered profile for {reg_name}")
                    st.session_state.last_status = "Profile saved"
                else:
                    st.sidebar.error("Provide both name and photo.")

        st.sidebar.markdown("---")
        st.sidebar.header("AI Model Configuration")
        with st.sidebar.expander("DeepFace Settings", expanded=False):
            st.session_state.selected_model = st.selectbox(
                "Facial Recognition Model",
                DEEPFACE_MODELS,
                index=(
                    DEEPFACE_MODELS.index(st.session_state.selected_model)
                    if st.session_state.selected_model in DEEPFACE_MODELS
                    else 0
                ),
                help="Choose the facial recognition model",
            )
            st.session_state.selected_backend = st.selectbox(
                "Face Detection Backend",
                DEEPFACE_BACKENDS,
                index=(
                    DEEPFACE_BACKENDS.index(st.session_state.selected_backend)
                    if st.session_state.selected_backend in DEEPFACE_BACKENDS
                    else 0
                ),
                help="Choose the face detection method",
            )
            st.session_state.selected_metric = st.selectbox(
                "Similarity Metric",
                DEEPFACE_METRICS,
                index=(
                    DEEPFACE_METRICS.index(st.session_state.selected_metric)
                    if st.session_state.selected_metric in DEEPFACE_METRICS
                    else 0
                ),
                help="Choose distance metric for comparison",
            )
            st.session_state.similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.similarity_threshold,
                step=0.05,
                help="Lower = more strict matching",
            )

        with st.sidebar.expander("Analysis Features", expanded=False):
            st.session_state.enable_attributes = st.checkbox(
                "Enable Facial Attributes (Age/Gender/Emotion/Race)",
                value=st.session_state.enable_attributes,
                help="Analyze demographic attributes",
            )
            st.session_state.enable_spoofing_detection = st.checkbox(
                "Enable Anti-Spoofing Detection",
                value=st.session_state.enable_spoofing_detection,
                help="Detect face spoofing/liveness attacks",
            )

        st.sidebar.markdown("---")
        st.sidebar.header("Admin Actions")
        if st.sidebar.button(
            "Clear All Registered Profiles", width="stretch", type="primary"
        ):
            clear_registered_profiles()
            st.sidebar.success("All registered profiles deleted.")
        if st.sidebar.button("Reset Full Audit Log", width="stretch", type="primary"):
            clear_audit_log()
            st.sidebar.success("Audit log reset.")

    return



def render_main_ui() -> None:
    st.title("🎯 N-ONE COMMAND CENTER")

    st.subheader("⚙️ Operational Console")
    col_ctrl1, col_ctrl2 = st.columns(2)
    mode = col_ctrl1.selectbox(
        "Select Active Surveillance Mode",
        ["1. Lost Person Search", "2. Member Attendance Logger", "3. Threat Detection Mode"],
        key="selected_mode", help="Select the primary mission for the AI."
    )
    source_type = col_ctrl2.radio(
        "Select Video Input Source",
        ["Laptop Webcam", "Recorded Video File", "IP Camera Stream"],
        horizontal=True, key="input_source"
    )

    video_target = None
    if source_type == "Laptop Webcam":
        video_target = 0
    elif source_type == "Recorded Video File":
        uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
        if uploaded_video:
            TEMP_VIDEO_PATH.write_bytes(uploaded_video.getbuffer())
            video_target = str(TEMP_VIDEO_PATH)
    elif source_type == "IP Camera Stream":
        video_target = st.text_input("Enter RTSP / HTTP Stream URL", placeholder="rtsp://...")

    st.markdown("---")
    log_df = pd.read_csv(CSV_LOG_PATH) if CSV_LOG_PATH.exists() and CSV_LOG_PATH.stat().st_size > 0 else pd.DataFrame()
    unk_df = pd.read_csv(UNKNOWN_DB_PATH) if UNKNOWN_DB_PATH.exists() and UNKNOWN_DB_PATH.stat().st_size > 0 else pd.DataFrame()
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Active Role", st.session_state.role)
    col_m2.metric("Registered Profiles", get_registered_profile_count())
    col_m3.metric("Unknowns Logged", len(unk_df))
    col_m4.metric("Total Log Events", len(log_df))

    button_label = "Stop Surveillance" if st.session_state.streaming else "Start Surveillance"
    if st.button(button_label, width='stretch', type="primary" if not st.session_state.streaming else "secondary"):
        if not st.session_state.streaming and video_target is None:
            st.error("Cannot start stream: no valid video source selected.")
        else:
            st.session_state.streaming = not st.session_state.streaming
            st.rerun()

    st.markdown("---")
    status_area = st.empty()
    frame_placeholder = st.empty()

    if st.session_state.streaming and video_target is not None:
        cap = cv2.VideoCapture(video_target)
        if not cap.isOpened():
            st.error(f"Error: Could not open video source.")
            st.session_state.streaming = False
        else:
            while st.session_state.streaming:
                ret, frame = cap.read()
                if not ret:
                    st.session_state.streaming = False
                    st.warning("Video stream ended or file finished.")
                    cap.release()
                    st.rerun()
                    break
                
                annotated_frame, detection_summary = process_frame(frame, mode)
                
                status_text = st.session_state.get('last_status', "Idle")
                status_area.info(f"Last Status: {status_text}")
                final_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(final_frame, width='stretch')
                
                time.sleep(0.01)

        if cap.isOpened():
            cap.release()
    else:
        frame_placeholder.info("Surveillance feed is offline. Select a source and start the stream.")


def render_facial_analytics_panel() -> None:
    """Render facial analytics and verification tools."""
    st.markdown("---")
    st.subheader("🔬 Facial Analytics & Verification Tools")
    
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(
        ["Face Comparison", "Batch Analysis", "Detection Tuning"]
    )
    
    with analytics_tab1:
        st.write("**1-to-1 Face Verification**")
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_img1 = st.file_uploader("Upload First Face Image", type=["jpg", "jpeg", "png"], key="verify_img1")
        with col2:
            uploaded_img2 = st.file_uploader("Upload Second Face Image", type=["jpg", "jpeg", "png"], key="verify_img2")
        
        if uploaded_img1 and uploaded_img2 and st.button("Verify Match"):
            temp_path1 = ROOT_DIR / "temp_verify_1.jpg"
            temp_path2 = ROOT_DIR / "temp_verify_2.jpg"
            
            with Image.open(uploaded_img1) as img:
                img.convert("RGB").save(temp_path1)
            with Image.open(uploaded_img2) as img:
                img.convert("RGB").save(temp_path2)
            
            result = verify_face_pair(str(temp_path1), str(temp_path2), st.session_state.selected_model)
            
            col_result1, col_result2, col_result3 = st.columns(3)
            col_result1.metric("Match Status", "✅ MATCH" if result.get("verified") else "❌ NO MATCH")
            col_result2.metric("Distance", f"{result.get('distance', 0):.4f}")
            col_result3.metric("Threshold", f"{result.get('threshold', 0):.4f}")
            
            if not result.get("verified"):
                st.warning("Faces do not match with current threshold settings.")
            else:
                st.success("Faces successfully verified as the same person!")
            
            temp_path1.unlink(missing_ok=True)
            temp_path2.unlink(missing_ok=True)
    
    with analytics_tab2:
        st.write("**Batch Facial Attribute Analysis**")
        uploaded_batch = st.file_uploader("Upload Image for Analysis", type=["jpg", "jpeg", "png"], key="batch_analysis")
        
        if uploaded_batch and st.button("Analyze Attributes"):
            temp_image = ROOT_DIR / "temp_analysis.jpg"
            with Image.open(uploaded_batch) as img:
                img.convert("RGB").save(temp_image)
            
            analysis_results = analyze_facial_attributes(cv2.imread(str(temp_image)), [])
            
            if analysis_results:
                for idx, result in enumerate(analysis_results):
                    st.write(f"**Face #{idx + 1}**")
                    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                    col_a1.metric("Age", f"{result.get('age', 'N/A')} yrs")
                    col_a2.metric("Gender", result.get('dominant_gender', 'N/A'))
                    col_a3.metric("Emotion", result.get('dominant_emotion', 'N/A'))
                    col_a4.metric("Race", result.get('dominant_race', 'N/A'))
            else:
                st.info("No faces detected or attribute analysis unavailable.")
            
            temp_image.unlink(missing_ok=True)
    
    with analytics_tab3:
        st.write("**Detection Model Tuning**")
        st.info("Model settings are managed by the platform owner and are read-only for User accounts.")
        
        col_tune1, col_tune2 = st.columns(2)
        with col_tune1:
            st.write("**Current Model Settings:**")
            st.code(f"""
Model: {st.session_state.selected_model}
Backend: {st.session_state.selected_backend}
Metric: {st.session_state.selected_metric}
Threshold: {st.session_state.similarity_threshold:.2f}
            """)
        
        with col_tune2:
            st.write("**Features Enabled:**")
            features = []
            if st.session_state.enable_attributes:
                features.append("✓ Facial Attributes")
            if st.session_state.enable_spoofing_detection:
                features.append("✓ Anti-Spoofing")
            if features:
                st.write("\n".join(features))
            else:
                st.write("No additional features enabled")


def render_log_viewer() -> None:
    st.markdown("---")
    st.subheader("📊 Structured Audit & Detection Logs")
    if CSV_LOG_PATH.exists() and CSV_LOG_PATH.stat().st_size > 0:
        log_df = pd.read_csv(CSV_LOG_PATH).sort_values(by="Timestamp", ascending=False)
        st.dataframe(log_df, width='stretch', height=300)
    else:
        st.info("No audit logs recorded yet.")

    st.subheader("🕵️ Unknown Persons Database")
    if UNKNOWN_DB_PATH.exists() and UNKNOWN_DB_PATH.stat().st_size > 0:
        try:
            unknown_df = pd.read_csv(UNKNOWN_DB_PATH).sort_values(by="last_seen_timestamp", ascending=False)
            st.dataframe(unknown_df, width='stretch', height=300)
        except pd.errors.EmptyDataError:
            st.info("Unknown person database is empty.")
    else:
        st.info("No unknown persons recorded yet.")


def main() -> None:
    """Main function to run the Streamlit application."""
    initialize_directories()
    initialize_log_files()
    configure_session_state()
    
    render_sidebar()
    
    if st.session_state.authenticated:
        load_known_face_encodings()
        render_main_ui()
        render_facial_analytics_panel()
        render_log_viewer()
    else:
        st.warning("🔒 System locked. Please authenticate via the sidebar.")


if __name__ == "__main__":
    main()
