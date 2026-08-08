# Project N-ONE: Advanced AI Surveillance Platform

## 1. Project Overview

N-ONE (no one escapes) is an Advanced Multi-Mode AI Surveillance & Threat Tracking Platform designed for modern security and monitoring challenges. It provides a comprehensive solution for lost person tracking, perimeter security, and threat detection.

The platform integrates real-time video stream analysis with sophisticated facial recognition and weapon detection capabilities. It is built on a Python ecosystem, utilizing Streamlit for an intuitive UI dashboard, OpenCV for core image processing, and the DeepFace framework for high-accuracy facial recognition.

## 2. Features

- **Multi-Modal Surveillance:** Dynamically switch between:
    - **Lost Person Search:** Scans for specific pre-registered individuals.
    - **Member Attendance Logger:** Tracks registered members and logs unknown individuals.
    - **Threat & Weapon Detection:** Detects hazardous objects using contour analysis and deep learning models.
- **Role-Based Access Control (RBAC):** Secure login with `Administrator` and `Operator` roles to manage access to sensitive features like subject registration.
- **High-Accuracy Facial Recognition:** Powered by the **DeepFace** library, supporting multiple models like `VGG-Face`, `Facenet`, `ArcFace`, and `SFace`.
- **Advanced Face Detection:** Utilizes various backends for robust face detection, including `OpenCV`, `MTCNN`, `RetinaFace`, and `MediaPipe`.
- **Facial Attribute Analysis:** Extracts demographic and emotional information, including **age, gender, emotion, and race**.
- **Persistent Unknown Person Tracking:** Automatically registers, logs, and re-identifies unknown individuals, maintaining a database of all sightings.
- **Dynamic Video Ingestion:** Supports a wide range of video sources, including local webcams, pre-recorded video files, and RTSP/TCP streams from IP cameras.
- **Structured Audit & Logging:** All significant events are logged to CSV files with timestamps, available for download and analysis through the UI.
- **Intuitive User Interface:** A modern, dark-themed dashboard built with Streamlit provides real-time video feeds, system controls, and data visualization.

## 3. System Architecture

N-ONE uses a modular, three-layered architecture:

1.  **Streamlit UI Layer:** Handles user interaction, including authentication (RBAC), mode selection, camera controls, and subject registration.
2.  **Processing Layer:** The core of the system, responsible for:
    - Capturing frames via **OpenCV**.
    - Performing facial recognition and matching using **DeepFace**.
    - Running weapon detection heuristics.
    - Annotating video frames with status updates.
3.  **Storage Layer:** Manages all persistent data on the file system:
    - `registered_faces/`: Stores images of registered members and lost persons.
    - `unknown_faces/`: Stores images of automatically detected unknown individuals.
    - `detection_logs/`: Contains all CSV audit logs.
    - `unknown_person_db.csv`: A database of unique unknown persons.
    - `unknown_sighting_log.csv`: A log of all sightings of unknown persons.

### Data Flow
1.  A user authenticates and selects a video source and surveillance mode via the Streamlit UI.
2.  The Processing Layer captures video frames using OpenCV.
3.  Based on the active mode, the frame is processed for facial recognition or threat detection.
4.  Results are annotated on the live video feed. In "Member Attendance" mode, new faces are logged to the unknown persons' database.
5.  All events are appended to the appropriate CSV log file and displayed in the UI.

## 4. Technology Stack

- **Language:** Python 3.10+
- **UI Framework:** Streamlit
- **Computer Vision:** OpenCV
- **Facial Recognition:** DeepFace
- **Data Handling:** Pandas, NumPy
- **Image Manipulation:** Pillow

## 5. Directory Structure

```text
project_n_one/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python package dependencies
├── README.md             # Project README
├── project.md            # This file: a comprehensive project overview
├── ARCHITECTURE.md       # Detailed architecture document
├── SYNOPSIS.md           # Project synopsis
├── registered_faces/     # Images for registered individuals
├── unknown_faces/        # Images of auto-detected unknown persons
├── detection_logs/       # CSV audit logs
├── unknown_person_db.csv # Database of unique unknown persons
└── unknown_sighting_log.csv # Sighting log for unknown persons
```

## 6. Setup and Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
3.  **Access the UI:** Open the URL provided by Streamlit in your web browser.

## 7. Facial Analysis Capabilities

The platform integrates a rich set of features from the DeepFace library.

### Recognition & Detection
- **Recognition Models:** VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace, SFace.
- **Detection Backends:** OpenCV, MTCNN, RetinaFace, MediaPipe, DLib, SSD, YOLO, YuNet.

### Facial Attribute Analysis
- **Age:** Predicts the estimated age of a person.
- **Gender:** Classifies as Male or Female.
- **Emotion:** Detects Happy, Angry, Sad, Surprise, Fear, Disgust, and Neutral.
- **Race:** Classifies individuals into categories such as Asian, White, Middle Eastern, Indian, Latino, and Black.

### Advanced Features
- **Face Verification (1-to-1):** Compares two images to verify if they belong to the same person.
- **Anti-Spoofing:** Detects liveness to prevent attacks using photos or videos.
- **Batch Analysis:** Processes images with multiple people and provides analysis for each detected face.
- **Customizable Tuning:** All models, backends, and similarity metrics can be adjusted through the UI for optimal performance.
