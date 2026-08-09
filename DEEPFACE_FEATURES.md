# DeepFace Integrated Features - N-ONE Surveillance Platform

## Overview
The N-ONE Surveillance Platform has been enhanced with comprehensive DeepFace features for advanced facial recognition, analysis, and verification capabilities.

---

## Core Features Added

### 1. **Facial Recognition Models**
Multiple pre-trained models for facial recognition:
- **VGG-Face** - Traditional deep learning model
- **Facenet** - Google's deep learning model (default)
- **Facenet512** - Enhanced Facenet with 512-dimensional embeddings
- **OpenFace** - Open-source facial recognition
- **DeepFace** - Facebook's DeepFace model
- **ArcFace** - Additive Angular Margin Loss
- **SFace** - Softmax loss-based model

**Configuration**: Select model from sidebar settings under "🔧 DeepFace Settings"

---

### 2. **Face Detection Backends**
Multiple detection methods for robustness:
- **OpenCV** - Fastest, lightweight (default)
- **MTCNN** - Multi-task Cascaded Convolutional Networks
- **RetinaFace** - State-of-the-art detection
- **MediaPipe** - Google's efficient detection framework
- **DLib** - Classical computer vision approach
- **SSD** - Single Shot Detector
- **YOLOv8/v11/v12** - Real-time object detection variants
- **YuNet** - Novel face detection
- **FastMTCNN** - Optimized MTCNN
- **CenterFace** - Efficient centroid-based detection

**Configuration**: Select backend from sidebar settings under "🔧 DeepFace Settings"

---

### 3. **Facial Attribute Analysis**
Automatically extract demographic and emotional information:

#### Age Detection
- Predicts estimated age of detected faces
- Displayed in "Batch Analysis" tab

#### Gender Classification
- Binary classification: Male/Female
- Confidence-based results

#### Emotion Recognition
- Detects: Happy, Angry, Sad, Surprise, Fear, Disgust, Neutral
- Useful for behavior analysis

#### Race Classification
- Identifies racial/ethnic characteristics
- Multiple categories support

**Configuration**: Enable/disable in sidebar under "📊 Analysis Features"

---

### 4. **Face Verification (1-to-1 Matching)**
Compare two face images to verify identity:

#### Comparison Metrics
- **Cosine Distance** (default) - Angular similarity
- **Euclidean Distance** - Direct spatial distance
- **Euclidean L2** - Normalized Euclidean

#### Usage
1. Navigate to "🔬 Facial Analytics" → "Face Comparison" tab
2. Upload two face images
3. Click "Verify Match"
4. View distance, threshold, and verification result

#### Configuration
- Adjust similarity threshold in sidebar (0.0 = strict, 1.0 = lenient)
- Select comparison metric (Cosine/Euclidean/Euclidean_L2)

---

### 5. **Anti-Spoofing Detection**
Detect face spoofing attacks (photos, videos, masks):

#### Features
- Liveness detection using Fasnet model
- Real vs. Fake classification
- Confidence scoring

#### Status
- Currently available in detection pipeline
- Can be enabled from sidebar settings
- Provides real/fake classification with confidence

**Note**: Requires compatible Python environment with TensorFlow support

---

### 6. **Enhanced Face Extraction**
Extract and align faces from images:

#### Functions
- Extracts individual face regions
- Aligns faces for optimal processing
- Returns aligned face embeddings
- Useful for preprocessing

#### Use Cases
- Pre-processing before analysis
- Face quality verification
- Image normalization

---

### 7. **Batch Facial Analysis**
Analyze multiple faces in a single image:

#### Capabilities
- Process images with multiple people
- Generate individual analysis per face
- Extract attributes for each person
- Export analysis results

#### Usage
1. Navigate to "🔬 Facial Analytics" → "Batch Analysis" tab
2. Upload an image with one or more faces
3. Click "Analyze Attributes"
4. View detailed results for each detected face

---

### 8. **Advanced Detection Tuning**
Fine-tune detection and recognition parameters:

#### Available Settings
- Model selection
- Detection backend choice
- Distance metric selection
- Similarity threshold adjustment
- Feature enablement

#### Detection Tuning Tab
- View current model configuration
- Monitor enabled features
- Real-time feedback on settings

---

## Configuration Panel

### Location
Sidebar → "🔧 DeepFace Settings" (expandable)

### Configurable Parameters

#### Recognition Settings
```
- Facial Recognition Model: [VGG-Face | Facenet | Facenet512 | ...]
- Face Detection Backend: [OpenCV | MTCNN | RetinaFace | ...]
- Similarity Metric: [Cosine | Euclidean | Euclidean_L2]
- Similarity Threshold: [0.0 - 1.0] (lower = stricter)
```

#### Analysis Features
```
- Enable Facial Attributes: [Toggle] (Age, Gender, Emotion, Race)
- Enable Anti-Spoofing: [Toggle] (Liveness detection)
```

---

## Surveillance Modes

### 1. **Lost Person Search**
- Requires one selected Victim profile
- Restricts known-face matching to that Victim only
- Shows the Victim name/location when found
- Silently stores or re-identifies all other faces as unknown

### 2. **Member Attendance Logger**
- Tracks registered Staff profiles
- Records attendance with timestamps
- Maintains audit logs
- Fast recognition preferred

### 3. **Threat Detection Mode**
- Weapon/threat detection via contour analysis
- Behavioral analysis
- Real-time alerts
- Security-focused operation

---

## New Functions Added

### Facial Attribute Analysis
```python
analyze_facial_attributes(frame, face_objs) -> list[dict]
```
Analyzes age, gender, emotion, and race for detected faces.

### Face Extraction
```python
extract_and_align_faces(frame) -> list[np.ndarray]
```
Extracts and aligns individual face regions.

### Face Verification
```python
verify_face_pair(img1_path, img2_path, model) -> dict
```
Compares two face images for identity verification.

### Spoofing Detection
```python
detect_spoofing(frame) -> dict
```
Detects face spoofing attacks (photos, masks, videos).

### Face Embeddings
```python
get_face_embeddings(frame, model) -> list
```
Extracts facial embeddings for comparison.

---

## Performance Considerations

### Model Selection Impact
- **VGG-Face**: Slower, high accuracy
- **Facenet**: Balanced speed/accuracy (default)
- **OpenFace**: Fast, moderate accuracy
- **ArcFace**: Excellent accuracy, moderate speed

### Backend Impact on Speed
1. **Fastest**: OpenCV, YuNet
2. **Fast**: MediaPipe, MTCNN
3. **Moderate**: RetinaFace, SSD
4. **Slowest**: DLib, YOLOv12

### Threshold Tuning
- **Lower threshold (0.2-0.4)**: Stricter matching, fewer false positives
- **Higher threshold (0.6-0.8)**: More permissive, more matches

---

## Best Practices

### For Lost Person Search
1. Use high-accuracy model (Facenet512 or ArcFace)
2. Set lower similarity threshold (0.3-0.4)
3. Enable facial attributes for description
4. Use RetinaFace or MediaPipe backend

### For Real-Time Surveillance
1. Use fast model (Facenet or OpenFace)
2. Use lightweight backend (OpenCV or YuNet)
3. Set moderate threshold (0.4-0.5)
4. Disable heavy features if performance needed

### For Security/Threat Detection
1. Enable anti-spoofing detection
2. Combine with weapon detection
3. Use robust backend (RetinaFace)
4. Set strict threshold (0.3)

### For Batch Processing
1. Use highest accuracy model
2. Enable facial attributes
3. Process in batches if possible
4. Extract embeddings for faster future matching

---

## Analytics Dashboard

### Facial Analytics Panel
Located below main surveillance feed after authentication

#### Tabs Available
1. **Face Comparison**: 1-to-1 verification tool
2. **Batch Analysis**: Multi-face analysis
3. **Detection Tuning**: Parameter adjustment

#### Features
- Real-time face comparison
- Attribute visualization
- Model performance metrics
- Configuration monitoring

---

## Data Privacy

### Temporary Files
- Verification images stored as temporary files
- Automatically deleted after analysis
- No persistent storage of comparison data

### Registered Profiles
- Stored in `/registered_faces/` directory
- Organized by category (`Staff`/`Victim`); legacy `Member_`/`Lost_` prefixes remain supported
- Audit logged for access

### Unknown Persons
- Stored in `/unknown_faces/` directory
- CSV database tracks sightings
- Timestamped events

---

## Troubleshooting

### "DeepFace runtime is unavailable"
- Indicates TensorFlow not installed
- App continues with fallback features
- Upgrade to compatible Python version (3.10-3.12)

### Poor Detection Performance
1. Switch to RetinaFace or MediaPipe backend
2. Ensure good lighting conditions
3. Adjust threshold settings
4. Use high-resolution input

### Slow Performance
1. Switch to faster model (Facenet or OpenFace)
2. Use lightweight backend (OpenCV)
3. Disable facial attributes analysis
4. Reduce input resolution

### Inaccurate Matches
1. Lower similarity threshold
2. Switch to Facenet512 or ArcFace model
3. Ensure clear face images
4. Register multiple angles of person

---

## System Requirements

### Minimum
- Python 3.10 or higher
- 4GB RAM
- CPU with AVX support

### Recommended
- Python 3.10-3.12
- 8GB+ RAM
- GPU support (NVIDIA with CUDA)
- SSD storage

### For Full TensorFlow Support
- Python < 3.13
- TensorFlow 2.13+
- tf-keras 2.19+
- CUDA 11.8+ (for GPU acceleration)

---

## Future Enhancements

Potential features for roadmap:
- 3D face recognition
- Facial expression analysis
- Micro-expression detection
- Face aging prediction
- Facial landmark tracking
- Real-time face re-identification
- Multi-face clustering
- Facial similarity ranking

---

## Support & Documentation

For more information:
- DeepFace GitHub: https://github.com/serengil/deepface
- Model documentation: See DeepFace README
- Configuration examples: Check sidebar settings

---

**Last Updated**: 2026-08-09
**Platform Version**: N-ONE v1.1  
**DeepFace Version**: 0.0.100

## Current N-ONE workflow update (2026-08-09)

### Registered profile categories

The application stores face-only crops in `registered_faces/` and classifies them as `Staff` or `Victim`. The filename parser remains backward compatible with legacy `Member_` (Staff) and `Lost_` (Victim) prefixes.

### Victim-only search

In `1. Lost Person Search`, an operator selects one Victim profile. The DeepFace/OpenCV embedding comparison is restricted to that profile. A Staff profile, another Victim, or an unknown person cannot appear as the visible Victim result.

Unknown faces are still sent through the existing unknown database workflow. New faces are saved to `unknown_faces/`; repeated faces update `unknown_person_db.csv` and `unknown_sighting_log.csv`. In Victim search these unknown events are intentionally silent in the rendered video/result panel.

### Victim sighting history

The operator supplies a camera location. When the selected Victim matches, the app records the profile ID, name, timestamp, and location in `detection_logs/victim_sighting_log.csv`. Duplicate frames at the same camera are throttled for 60 seconds, and the UI shows the latest sighting for each distinct location.

### Current verification

```powershell
python -m pytest tests -q
python -m py_compile app.py deepface_adapter.py
```
