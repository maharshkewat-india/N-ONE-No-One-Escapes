import hmac
import os
import time
from io import BytesIO
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
VICTIM_SIGHTING_LOG_PATH = LOG_DIR / "victim_sighting_log.csv"

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
PROFILE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

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
    if not VICTIM_SIGHTING_LOG_PATH.exists() or VICTIM_SIGHTING_LOG_PATH.stat().st_size == 0:
        pd.DataFrame(columns=["profile_id", "name", "timestamp", "location"]).to_csv(
            VICTIM_SIGHTING_LOG_PATH, index=False
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


def get_profile_category(profile_stem: str) -> str:
    """Normalize current and legacy profile prefixes to the dashboard labels."""
    prefix = profile_stem.split("_", 1)[0].lower()
    return "Victim" if prefix in {"victim", "lost"} else "Staff"


def get_profile_name(profile_stem: str) -> str:
    """Return a readable profile name from a stored filename."""
    _, separator, name = profile_stem.partition("_")
    return (name if separator else profile_stem).replace("_", " ")


def get_registered_profiles_df() -> pd.DataFrame:
    """Build the registered-face inventory used by both authenticated roles."""
    profiles = []
    for image_path in REG_DIR.iterdir() if REG_DIR.exists() else []:
        if not image_path.is_file() or image_path.suffix.lower() not in PROFILE_IMAGE_SUFFIXES:
            continue
        profiles.append(
            {
                "profile_id": image_path.stem,
                "name": get_profile_name(image_path.stem),
                "category": get_profile_category(image_path.stem),
                "file_name": image_path.name,
            }
        )

    return pd.DataFrame(
        profiles,
        columns=["profile_id", "name", "category", "file_name"],
    ).sort_values(["category", "name"], ignore_index=True)


def get_registered_profile_count() -> int:
    """Return the number of saved profiles visible to the dashboard."""
    return len(get_registered_profiles_df())


def get_unknown_face_count() -> int:
    """Return the number of unique unknown faces currently tracked."""
    try:
        unknown_df = pd.read_csv(UNKNOWN_DB_PATH)
        if "unknown_id" in unknown_df.columns:
            unknown_count = int(unknown_df["unknown_id"].dropna().astype(str).nunique())
            if unknown_count:
                return unknown_count
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        pass

    if not UNKNOWN_DIR.exists():
        return 0
    return sum(
        1
        for image_path in UNKNOWN_DIR.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in PROFILE_IMAGE_SUFFIXES
    )


def is_victim_search_mode(mode: str) -> bool:
    """Return whether the selected mode searches for one victim target only."""
    return mode.startswith("1.")


def record_victim_sighting(profile_id: str, name: str, location: str) -> bool:
    """Persist a victim sighting, throttling duplicate frames at one camera."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        sighting_df = pd.read_csv(VICTIM_SIGHTING_LOG_PATH)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        sighting_df = pd.DataFrame(
            columns=["profile_id", "name", "timestamp", "location"]
        )

    matches = sighting_df[sighting_df["profile_id"].astype(str) == str(profile_id)]
    if not matches.empty:
        last = matches.sort_values("timestamp").iloc[-1]
        last_timestamp = pd.to_datetime(last.get("timestamp"), errors="coerce")
        current_timestamp = pd.to_datetime(timestamp)
        if (
            str(last.get("location", "")) == location
            and not pd.isna(last_timestamp)
            and (current_timestamp - last_timestamp).total_seconds() < 60
        ):
            return False

    write_header = not VICTIM_SIGHTING_LOG_PATH.exists() or VICTIM_SIGHTING_LOG_PATH.stat().st_size == 0
    pd.DataFrame(
        [[profile_id, name, timestamp, location]],
        columns=["profile_id", "name", "timestamp", "location"],
    ).to_csv(VICTIM_SIGHTING_LOG_PATH, mode="a", header=write_header, index=False)
    return True


def get_victim_sighting_history(profile_id: str = "") -> pd.DataFrame:
    """Return the latest sighting for each location for a victim profile."""
    columns = ["profile_id", "name", "timestamp", "location"]
    try:
        sighting_df = pd.read_csv(VICTIM_SIGHTING_LOG_PATH)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)

    if profile_id:
        sighting_df = sighting_df[
            sighting_df["profile_id"].astype(str) == str(profile_id)
        ]
    if sighting_df.empty:
        return pd.DataFrame(columns=columns)

    return (
        sighting_df.sort_values("timestamp", ascending=False)
        .drop_duplicates("location", keep="first")
        .reset_index(drop=True)
    )


@st.cache_resource
def load_known_face_encodings():
    """Loads all known face encodings from the registration directory into memory."""
    if DeepFace is None:
        return

    KNOWN_FACE_ENCODINGS.clear()
    for img_path in REG_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        name = get_profile_name(img_path.stem)
        role = get_profile_category(img_path.stem)
        embedding = None
        try:
            embedding_obj = DeepFace.represent(
                img_path=str(img_path), model_name=DEEPFACE_MODEL, enforce_detection=False
            )
            if embedding_obj and len(embedding_obj) > 0:
                embedding = embedding_obj[0].get("embedding")
        except Exception as e:
            st.warning(f"Could not process {img_path.name}: {e}")

        KNOWN_FACE_ENCODINGS.append(
            {
                "profile_id": img_path.stem,
                "name": name,
                "role": role,
                "encoding": embedding,
            }
        )


def find_face_in_known_cache(
    face_encoding: list,
    threshold: float = COSINE_THRESHOLD,
    profile_id: str | None = None,
    role: str | None = None,
) -> dict | None:
    """Return the closest known profile when it is inside the match threshold."""
    best_match = None
    best_distance = float("inf")
    for entry in KNOWN_FACE_ENCODINGS:
        if profile_id and entry.get("profile_id") != profile_id:
            continue
        if role and entry.get("role") != role:
            continue
        if entry.get("encoding") is None:
            continue
        distance = cosine(np.array(face_encoding), np.array(entry["encoding"]))
        if distance < best_distance:
            best_match = entry
            best_distance = distance

    if best_match is None or best_distance > threshold:
        return None

    return {**best_match, "distance": float(best_distance)}


def record_detection_result(
    result_type: str,
    subject: str = "",
    distance: float | None = None,
    details: str = "",
    image_name: str = "",
    location: str = "",
    previous_timestamp: str = "",
    previous_location: str = "",
    profile_id: str = "",
) -> None:
    """Store the latest recognition event for the live status panel."""
    st.session_state.last_detection = {
        "type": result_type,
        "subject": subject,
        "distance": distance,
        "details": details,
        "image_name": image_name,
        "location": location,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "previous_timestamp": previous_timestamp,
        "previous_location": previous_location,
        "profile_id": profile_id,
    }


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


def get_last_unknown_sighting(unknown_id: str) -> dict[str, str]:
    """Read the last saved date/time and location before a new re-identification."""
    try:
        sighting_df = pd.read_csv(UNKNOWN_SIGHTING_LOG_PATH)
        matches = sighting_df[sighting_df["unknown_id"].astype(str) == unknown_id]
        if not matches.empty:
            last = matches.sort_values("timestamp").iloc[-1]
            return {
                "timestamp": str(last.get("timestamp", "")),
                "location": str(last.get("location", "")),
            }
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, IndexError):
        pass

    try:
        unknown_df = pd.read_csv(UNKNOWN_DB_PATH)
        matches = unknown_df[unknown_df["unknown_id"].astype(str) == unknown_id]
        if not matches.empty:
            last = matches.iloc[-1]
            return {
                "timestamp": str(last.get("last_seen_timestamp", "")),
                "location": str(last.get("last_known_location", "")),
            }
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, IndexError):
        pass

    return {"timestamp": "", "location": ""}


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


def process_frame(
    frame: np.ndarray,
    mode: str,
    target_profile_id: str | None = None,
    location: str = "Main Feed",
) -> tuple[np.ndarray, str]:
    """The core processing pipeline for each video frame."""
    annotated_frame = frame.copy()
    detection_summary = "Status: Idle"
    victim_search = is_victim_search_mode(mode)
    victim_found = None

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
        record_detection_result("no_threat", details=detection_summary)
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
        record_detection_result("error", details=f"Face analysis error: {e}")
        return annotated_frame, f"Error during face representation: {e}"

    if not face_objs:
        st.session_state.last_status = "No faces detected in frame."
        record_detection_result("no_face", details="No face detected in the current frame.")
        return annotated_frame, "No faces detected in frame."

    detection_summary = f"Found {len(face_objs)} face(s), analyzing..."
    
    for face_obj in face_objs:
        facial_area = face_obj["facial_area"]
        x = int(facial_area["x"])
        y = int(facial_area["y"])
        w = int(facial_area["w"])
        h = int(facial_area["h"])
        face_roi = frame[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue
            
        face_encoding = face_obj["embedding"]

        # Step A: Check against the KNOWN faces in-memory cache
        match_threshold = float(
            st.session_state.get("similarity_threshold", COSINE_THRESHOLD)
        )
        if victim_search:
            known_match = (
                find_face_in_known_cache(
                    face_encoding,
                    match_threshold,
                    profile_id=target_profile_id,
                    role="Victim",
                )
                if target_profile_id
                else None
            )
        else:
            known_match = find_face_in_known_cache(face_encoding, match_threshold)
        if known_match:
            name, role = known_match["name"], known_match["role"]
            distance = known_match["distance"]
            color = (0, 255, 0) if role == "Staff" else (255, 0, 0)
            if victim_search:
                victim_found = known_match
                color = (0, 255, 0)
                if record_victim_sighting(
                    known_match["profile_id"], name, location
                ):
                    log_event(
                        mode=mode,
                        subject_id=known_match["profile_id"],
                        role="Victim",
                        event_type="Victim Found",
                        details=f"Camera location: {location}",
                    )
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            label = f"{role}: {name}"
            cv2.putText(annotated_frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            detection_summary = f"Matched: {label} (distance {distance:.3f})"
            if victim_search:
                record_detection_result(
                    "victim_found",
                    subject=name,
                    distance=distance,
                    details="Selected victim matched against the camera feed.",
                    location=location,
                    profile_id=known_match["profile_id"],
                )
            else:
                record_detection_result(
                    "known_match",
                    subject=label,
                    distance=distance,
                    details="Matched against a registered profile.",
                )
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
                unknown_matches = dfs[0].sort_values("distance")
                best_unknown = unknown_matches.iloc[0]
                match_path = Path(best_unknown["identity"])
                unknown_id = match_path.stem
                distance = float(best_unknown.get("distance", 0.0))
                previous_sighting = get_last_unknown_sighting(unknown_id)
                update_unknown_sighting(unknown_id, location, mode)
                if victim_search:
                    continue
                detection_summary = f"Re-identified previous unknown: {unknown_id} (distance {distance:.3f})"
                record_detection_result(
                    "unknown_match",
                    subject=unknown_id,
                    distance=distance,
                    details="Matched against a previously saved unknown face.",
                    image_name=match_path.name,
                    location=location,
                    previous_timestamp=previous_sighting.get("timestamp", ""),
                    previous_location=previous_sighting.get("location", ""),
                )
                color = (255, 165, 0) # Orange
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated_frame, f"ID: {unknown_id}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                continue
        except Exception:
             # This can fail if UNKNOWN_DIR is empty, which is fine on first run.
             pass
        
        # Step C: If not known and not previously unknown, register as a NEW UNKNOWN
        unknown_id = register_new_unknown(face_roi, location, mode)
        if victim_search:
            continue
        detection_summary = f"No registered match - saved as new unknown: {unknown_id}"
        record_detection_result(
            "new_unknown",
            subject=unknown_id,
            details="No known or previous-unknown match; face image was saved.",
            image_name=f"{unknown_id}.jpg",
            location=location,
        )
        color = (255, 255, 0) # Cyan
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated_frame, f"New ID: {unknown_id}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    if victim_search:
        if victim_found:
            detection_summary = (
                f"Victim found: {victim_found['name']} at {location}"
            )
        else:
            target_name = next(
                (
                    entry["name"]
                    for entry in KNOWN_FACE_ENCODINGS
                    if entry.get("profile_id") == target_profile_id
                ),
                target_profile_id or "selected victim",
            )
            detection_summary = (
                f"Searching for {target_name}. Selected victim not found in this frame."
            )
            record_detection_result(
                "victim_search",
                details="No selected victim match in the current camera frame.",
                location=location,
                profile_id=target_profile_id or "",
            )

    st.session_state.last_status = detection_summary
    return annotated_frame, detection_summary


def render_detection_result(container) -> None:
    """Render a clear, human-readable result for the most recent face event."""
    panel = container.container()
    result = st.session_state.get("last_detection", {})
    if not isinstance(result, dict):
        panel.info(str(result))
        return

    result_type = result.get("type", "idle")
    subject = result.get("subject", "")
    distance = result.get("distance")
    distance_text = f" | distance: {distance:.3f}" if distance is not None else ""
    timestamp = result.get("timestamp", "")
    details = result.get("details", "")
    previous_timestamp = result.get("previous_timestamp", "")
    previous_location = result.get("previous_location", "")
    current_location = result.get("location", "")
    profile_id = result.get("profile_id", "")

    if result_type == "victim_found":
        panel.success(f"VICTIM FOUND: {subject}{distance_text}")
        panel.markdown(f"**Camera location:** {current_location or 'Location unavailable'}")
        history = get_victim_sighting_history(profile_id)
        if not history.empty:
            panel.write("**Victim sighting history by location**")
            panel.dataframe(
                history[["timestamp", "location"]].rename(
                    columns={"timestamp": "Last seen", "location": "Camera location"}
                ),
                hide_index=True,
                width="stretch",
            )
    elif result_type == "victim_search":
        panel.info("SEARCHING: Selected victim was not found in the current frame.")
    elif result_type == "known_match":
        panel.success(f"MATCH FOUND: {subject}{distance_text}")
    elif result_type == "unknown_match":
        panel.warning(f"PREVIOUS UNKNOWN MATCH: {subject}{distance_text}")
        if previous_timestamp or previous_location:
            panel.markdown(
                "**Previous sighting:** "
                f"{previous_timestamp or 'date/time unavailable'} at "
                f"{previous_location or 'location unavailable'}"
            )
    elif result_type == "new_unknown":
        panel.warning(f"NO MATCH - SAVED AS NEW UNKNOWN: {subject}")
    elif result_type == "no_face":
        panel.info("NO FACE DETECTED")
    elif result_type == "error":
        panel.error(details or "Face analysis failed.")
    elif result_type == "no_threat":
        panel.info(details or "No threat detected.")
    else:
        panel.info(details or "Waiting for a detection result.")

    if details and result_type not in {"error", "no_threat"}:
        panel.caption(details)
    if result.get("image_name"):
        image_path = UNKNOWN_DIR / result["image_name"]
        panel.caption(f"Saved/matched image: {result['image_name']}")
        if image_path.exists():
            panel.image(str(image_path), caption="Previous/saved unknown face", width="content")
    if timestamp:
        event_label = "Current match" if result_type == "unknown_match" else "Last event"
        current_context = f" at {current_location}" if current_location else ""
        panel.caption(f"{event_label}: {timestamp}{current_context}")


def configure_session_state() -> None:
    defaults = {
        "authenticated": False, "role": "Guest", "streaming": False,
        "last_frame": None, "last_status": "Idle",
        "last_detection": {
            "type": "idle",
            "subject": "",
            "distance": None,
            "details": "No detections yet.",
            "image_name": "",
            "timestamp": "",
            "location": "",
            "previous_timestamp": "",
            "previous_location": "",
            "profile_id": "",
        },
        "camera_location": "Main Feed",
        "victim_search_target": None,
        "last_logged_event": "",
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


def detect_face_regions(image: np.ndarray) -> list[dict]:
    """Return face bounding boxes from either the real or fallback backend."""
    if hasattr(DeepFace, "detect_faces"):
        try:
            return [
                item["facial_area"]
                for item in DeepFace.detect_faces(image)
                if item.get("facial_area")
            ]
        except Exception:
            pass

    try:
        extracted = DeepFace.extract_faces(
            img_path=image,
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
        )
        return [
            item["facial_area"]
            for item in extracted
            if item.get("facial_area")
        ]
    except Exception:
        return []


def prepare_single_face_crop(uploaded_file) -> tuple[np.ndarray | None, str]:
    """Validate a registration photo and return a face-only BGR crop."""
    if uploaded_file is None:
        return None, "Choose an image or capture a photo first."

    try:
        image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        return None, f"Could not read the image: {exc}"

    regions = detect_face_regions(frame)
    if len(regions) == 0:
        return None, "No face detected. Face the camera directly with good lighting."
    if len(regions) > 1:
        return None, "Multiple faces detected. Keep only one person in the frame."

    region = regions[0]
    x, y, width, height = (
        int(region["x"]),
        int(region["y"]),
        int(region["w"]),
        int(region["h"]),
    )
    if width < 80 or height < 80:
        return None, "Face is too small. Move closer to the camera."

    padding = int(max(width, height) * 0.25)
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(frame.shape[1], x + width + padding)
    bottom = min(frame.shape[0], y + height + padding)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None, "Face crop was empty. Please capture the image again."

    return crop, "One face detected. Only the face crop will be saved."


def save_registered_profile(
    name: str, role_label: str, uploaded_file
) -> tuple[bool, str]:
    """Save a face-only profile. This operation is Administrator-only."""
    if not name:
        return False, "Provide a name or identifier."

    crop, message = prepare_single_face_crop(uploaded_file)
    if crop is None:
        return False, message

    prefix = "Victim" if "Victim" in role_label or "Lost" in role_label else "Staff"
    safe_name = "".join(
        character if character.isalnum() else "_" for character in name
    ).strip("_")
    if not safe_name:
        return False, "Provide a valid name or identifier."

    destination = REG_DIR / f"{prefix}_{safe_name}.jpg"
    cv2.imwrite(str(destination), crop)

    load_known_face_encodings.clear()
    load_known_face_encodings()
    return True, message


def clear_audit_log() -> None:
    """Clear audit records. This operation is available to Administrators only."""
    if CSV_LOG_PATH.exists():
        CSV_LOG_PATH.unlink()
    if VICTIM_SIGHTING_LOG_PATH.exists():
        VICTIM_SIGHTING_LOG_PATH.unlink()
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
        photo_source = st.sidebar.radio(
            "Profile photo source",
            ["Upload image", "Camera capture"],
            horizontal=True,
            key="registration_photo_source",
        )
        if photo_source == "Camera capture":
            uploaded_photo = st.sidebar.camera_input(
                "Capture one face",
                key="registration_camera",
                help="Keep one person centered, facing the camera, with good lighting.",
                resolution="720p",
            )
        else:
            uploaded_photo = st.sidebar.file_uploader(
                "Upload face image", type=["jpg", "jpeg", "png"], key="registration_upload"
            )

        prepared_crop = None
        if uploaded_photo is not None:
            prepared_crop, crop_message = prepare_single_face_crop(uploaded_photo)
            if prepared_crop is not None:
                st.sidebar.success(crop_message)
                st.sidebar.image(
                    cv2.cvtColor(prepared_crop, cv2.COLOR_BGR2RGB),
                    caption="Face-only profile preview",
                    width="content",
                )
            else:
                st.sidebar.warning(crop_message)

        with st.sidebar.form("registration_form"):
            reg_name = st.text_input("Full Name / Identifier")
            reg_role = st.selectbox(
                "Classification Role",
                ["Staff", "Victim"],
            )
            submitted = st.form_submit_button("Save Profile")
            if submitted:
                saved, save_message = save_registered_profile(
                    reg_name, reg_role, uploaded_photo
                )
                if saved:
                    st.sidebar.success(f"Registered profile for {reg_name}")
                    st.session_state.last_status = "Profile saved"
                else:
                    st.sidebar.error(save_message)

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

    camera_location = st.text_input(
        "Camera location",
        key="camera_location",
        help="This location is saved with unknown tracking and victim sightings.",
    ).strip() or "Main Feed"

    st.markdown("---")
    log_df = pd.read_csv(CSV_LOG_PATH) if CSV_LOG_PATH.exists() and CSV_LOG_PATH.stat().st_size > 0 else pd.DataFrame()
    registered_df = get_registered_profiles_df()

    victim_search_target = None
    if is_victim_search_mode(mode):
        victim_profiles = registered_df[registered_df["category"] == "Victim"]
        if victim_profiles.empty:
            st.warning("No Victim profile is registered. An Administrator must register a Victim first.")
        else:
            victim_labels = {
                row["profile_id"]: row["name"]
                for _, row in victim_profiles.iterrows()
            }
            victim_search_target = st.selectbox(
                "Victim to search for",
                options=list(victim_labels),
                format_func=lambda profile_id: victim_labels[profile_id],
                key="victim_search_target",
                help="Only this Victim profile will be shown as a match. Other faces are tracked silently as unknown.",
            )
            st.caption(
                "Victim search mode matches only the selected Victim. Unknown faces are saved silently for later review."
            )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Active Role", st.session_state.role)
    col_m2.metric("Registered Faces", len(registered_df))
    col_m3.metric("Unknown Faces", get_unknown_face_count())
    col_m4.metric("Total Log Events", len(log_df))

    render_face_inventory_panel(registered_df)

    button_label = "Stop Surveillance" if st.session_state.streaming else "Start Surveillance"
    if st.button(button_label, width='stretch', type="primary" if not st.session_state.streaming else "secondary"):
        if not st.session_state.streaming and video_target is None:
            st.error("Cannot start stream: no valid video source selected.")
        elif not st.session_state.streaming and is_victim_search_mode(mode) and not victim_search_target:
            st.error("Select a Victim profile before starting Lost Person Search.")
        else:
            st.session_state.streaming = not st.session_state.streaming
            st.rerun()

    st.markdown("---")
    status_area = st.empty()
    match_area = st.empty()
    render_detection_result(match_area)
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
                
                annotated_frame, detection_summary = process_frame(
                    frame,
                    mode,
                    target_profile_id=victim_search_target,
                    location=camera_location,
                )
                
                status_text = st.session_state.get('last_status', "Idle")
                status_area.info(f"Last Status: {status_text}")
                render_detection_result(match_area)
                final_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(final_frame, width='stretch')
                
                time.sleep(0.01)

        if cap.isOpened():
            cap.release()
    else:
        frame_placeholder.info("Surveillance feed is offline. Select a source and start the stream.")


def render_face_inventory_panel(registered_df: pd.DataFrame | None = None) -> None:
    """Show registered-face categories and counts to Administrators and Operators."""
    profiles = registered_df if registered_df is not None else get_registered_profiles_df()
    staff_count = int((profiles["category"] == "Staff").sum()) if not profiles.empty else 0
    victim_count = int((profiles["category"] == "Victim").sum()) if not profiles.empty else 0

    st.markdown("---")
    st.subheader("Registered face inventory")
    count_col1, count_col2, count_col3 = st.columns(3)
    count_col1.metric("All registered faces", len(profiles))
    count_col2.metric("Staff", staff_count)
    count_col3.metric("Victim", victim_count)

    with st.container(border=True):
        selected_category = st.segmented_control(
            "Show registered faces",
            ["All", "Staff", "Victim"],
            default="All",
            key="registered_face_category",
            width="stretch",
        ) or "All"
        visible_profiles = profiles
        if selected_category != "All":
            visible_profiles = profiles[profiles["category"] == selected_category]

        if visible_profiles.empty:
            st.info(f"No {selected_category.lower()} profiles registered yet.")
        else:
            st.dataframe(
                visible_profiles[["name", "category", "file_name"]].rename(
                    columns={
                        "name": "Name",
                        "category": "Category",
                        "file_name": "Profile file",
                    }
                ),
                hide_index=True,
                width="stretch",
            )


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

    st.subheader("Victim sighting history")
    victim_history = get_victim_sighting_history()
    if victim_history.empty:
        st.info("No Victim sightings recorded yet.")
    else:
        st.dataframe(
            victim_history[["name", "timestamp", "location"]].rename(
                columns={
                    "name": "Victim",
                    "timestamp": "Last seen",
                    "location": "Camera location",
                }
            ),
            hide_index=True,
            width="stretch",
            height=300,
        )


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
