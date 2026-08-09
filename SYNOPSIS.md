# PROJECT SYNOPSIS: N-ONE (no one escapes)

## 1. INTRODUCTION
N-ONE is an Advanced Multi-Mode AI Surveillance & Threat Tracking Platform designed for modern security and monitoring challenges. Leveraging cutting-edge AI-driven computer vision, N-ONE offers a robust solution for diverse applications, including lost person tracking, perimeter security, and threat detection. The platform integrates real-time video stream analysis with sophisticated facial recognition and weapon detection capabilities. Built on a Python 3.10+ ecosystem, N-ONE utilizes Streamlit for an intuitive, dark-themed UI dashboard, OpenCV for core image processing, and the DeepFace framework (with Facenet/ArcFace backends) for high-accuracy facial embedding extraction. Data logging and analysis are managed via Pandas, while PIL handles image manipulation, and RTSP/TCP Stream decoders ensure versatile video feed ingestion. This comprehensive system is engineered to provide proactive monitoring, rapid identification, and structured audit trails, significantly enhancing situational awareness and response capabilities in various operational environments.

## 2. ADVANTAGES OF THE PROPOSED SYSTEM & LIMITATIONS

### Advantages:
*   **Multi-Modal Surveillance:** Seamless integration of lost person tracking, member attendance logging, and threat detection within a single, unified platform.
*   **High-Accuracy Facial Recognition:** Utilizes DeepFace with robust backends (Facenet/ArcFace) to ensure high precision in identifying individuals, minimizing false positives and negatives.
*   **Dynamic Video Ingestion:** Supports diverse video sources including laptop webcams, pre-recorded files, and IP camera RTSP/TCP streams, offering deployment flexibility.
*   **Intuitive User Interface:** A modern dark UI dashboard built with Streamlit provides an accessible and efficient operational experience for both administrators and operators.
*   **Role-Based Access Control (RBAC):** Ensures secure operation with distinct access levels for Admin and Operator roles, enhancing system integrity and data protection.
*   **Comprehensive Audit & Logging:** Real-time CSV logging and downloadable interactive data tables facilitate thorough post-event analysis, compliance, and operational transparency.
*   **Proactive Threat Detection:** Incorporates heuristic/contour analysis and deep learning models for early detection of hazardous objects and weapons, significantly improving security posture.
*   **Persistent Unknown Person Tracking:** Automatically registers, logs, and re-identifies unknown individuals; during Victim Search this tracking is silent while only the selected Victim is shown.

### Limitations:
*   **Environmental Dependencies:** Performance of computer vision models can be affected by lighting conditions, occlusions, camera angle, and image quality.
*   **Computational Resources:** High-accuracy deep learning models, especially for real-time processing of multiple streams, demand substantial computational power (e.g., GPU acceleration).
*   **Data Privacy Concerns:** Facial recognition capabilities necessitate stringent adherence to privacy regulations and ethical guidelines for data handling and storage.
*   **Scalability:** While designed for versatility, scaling to a very large number of simultaneous high-resolution video streams might require further optimization and distributed computing architectures.
*   **False Positives/Negatives in Threat Detection:** Heuristic and model-based weapon detection, while advanced, can still produce false alerts or miss novel/obscure threats, requiring human oversight.

## 3. PROBLEM STATEMENT
Current surveillance systems often lack the integration and intelligence required to address multi-faceted security challenges effectively. Traditional setups typically operate in siloed modes, offering either basic recording, rudimentary motion detection, or isolated facial recognition, without a unified approach to diverse threat vectors. This fragmentation leads to delayed responses in critical situations, inefficient monitoring of specific targets (e.g., lost persons), and a reactive rather than proactive security posture. Furthermore, the absence of robust role-based access controls and structured audit trails compromises data integrity and accountability. The manual processing of surveillance footage for identifying individuals or detecting threats is labor-intensive, error-prone, and unsustainable for large-scale or continuous monitoring operations. There is a critical need for an integrated, intelligent, and adaptable surveillance platform that can dynamically switch between operational modes, provide high-accuracy identification, and offer comprehensive logging and access control, thereby enhancing overall security efficacy and operational efficiency.

## 4. OBJECTIVES
The primary objectives of the N-ONE platform are:
*   To design and implement a multi-mode AI surveillance system capable of dynamically switching between lost person tracking, member attendance logging, and threat detection functionalities.
*   To develop a secure Role-Based Access Control (RBAC) gateway to differentiate between administrative and operational privileges, ensuring system integrity and data security.
*   To integrate high-accuracy facial recognition capabilities using the DeepFace framework (Facenet/ArcFace backends) for reliable subject identification and embedding extraction.
*   To enable versatile video feed ingestion from various sources, including laptop webcams, recorded video files, and real-time IP camera RTSP/TCP streams.
*   To construct a Subject Profile Registration Engine for secure enrollment and management of target individuals (lost persons) and authorized personnel.
*   To implement a system for tracking and re-identifying unknown individuals, complete with a persistent database and sighting logs.
*   To implement a robust Structured Audit & Logging Engine that provides real-time event logging in CSV format and offers interactive, downloadable data tables for retrospective analysis.
*   To create an intuitive and responsive user interface using Streamlit, providing a modern dark-themed dashboard for efficient system operation and monitoring.
*   To enhance perimeter security through advanced heuristic/contour analysis and deep learning models for the detection of hazardous objects and weapons.

## 5. LITERATURE REVIEW & SYSTEM COMPARISON

### Literature Review:
The field of computer vision for surveillance has seen rapid advancements, driven by deep learning breakthroughs. Early systems relied on traditional image processing techniques like Haar cascades for object detection (e.g., Viola-Jones algorithm for faces), often suffering from high false positive rates and sensitivity to variations in pose, lighting, and occlusion. The advent of Convolutional Neural Networks (CNNs) revolutionized this domain, enabling more robust and accurate object detection (e.g., R-CNN, Faster R-CNN, YOLO, SSD) and facial recognition.

Facial recognition specifically has evolved from Eigenfaces and Fisherfaces to deep learning-based embedding networks. Models like Facenet (Schroff et al., 2015) and ArcFace (Deng et al., 2019) have set new benchmarks for accuracy by learning highly discriminative face embeddings in a high-dimensional space, significantly improving performance in challenging real-world scenarios. These models typically utilize triplet loss or additive angular margin loss functions to maximize inter-class variance and minimize intra-class variance.

Weapon detection research has similarly benefited from deep learning, moving beyond simple contour analysis to sophisticated CNN architectures trained on large datasets of various weapon types. These models can identify firearms, knives, and other hazardous objects with increasing precision, though challenges remain in detecting partially obscured or unusually presented items. Real-time processing of these complex models requires efficient inference techniques and often specialized hardware accelerators. The integration of RTSP/TCP streaming protocols is standard for modern IP camera surveillance, necessitating efficient decoding and frame processing. Streamlit has emerged as a popular framework for rapidly deploying interactive data applications and dashboards in Python, offering a simpler alternative to traditional web frameworks for visualization and control interfaces.

### System Comparison (N-ONE vs. Traditional/Basic Systems):

| Feature / Aspect             | Traditional / Basic Surveillance Systems                   | N-ONE Platform                                                                      |
| :--------------------------- | :--------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| **Operational Modes**        | Single-purpose (e.g., recording, motion detection, basic FR) | **Multi-Mode:** Lost Person Search, Member Attendance, Threat Detection             |
| **Facial Recognition**       | Often rule-based, less accurate, limited by environmental factors | **AI-Driven (DeepFace - Facenet/ArcFace):** High accuracy, robust to variations   |
| **Unknown Person Tracking**  | Not available                                              | **Persistent Logging & Re-identification:** Tracks non-registered individuals       |
| **Threat Detection**         | Simple motion detection, basic contour analysis, human-dependent | **Advanced AI:** Heuristic/contour + deep model detection of weapons               |
| **Video Ingestion**          | Limited to specific camera types, often proprietary formats | **Dynamic & Versatile:** Laptop Webcam, Recorded Files, RTSP/TCP IP Cameras         |
| **User Interface**           | Often clunky, desktop-only, or dated web interfaces        | **Modern Streamlit UI:** Intuitive, dark-themed, web-based dashboard              |
| **Access Control**           | Basic user accounts, often lacking granular permissions    | **Role-Based Access Control (RBAC):** Admin & Operator roles with distinct privileges |
| **Audit & Logging**          | Manual review of footage, simple event logs, non-structured | **Structured Audit & Logging Engine:** Real-time CSV, interactive data tables     |
| **Scalability (initial)**    | Limited by hardware and software integration complexity    | Designed for modularity; extensible for diverse environments (requires resources) |
| **Deployment Complexity**    | Often vendor-locked, complex setup and configuration       | Python-based, leverages open-source libraries, simpler deployment for developers |
| **Proactiveness**            | Reactive (post-event analysis)                             | **Proactive:** Real-time alerts, dynamic mode switching, instant identification    |

## 6. METHODOLOGY & SYSTEM ARCHITECTURE

N-ONE employs a modular and layered architecture to ensure scalability, maintainability, and robust performance. The system's core is built around a data processing pipeline that dynamically adapts to selected operational modes.

### System Architecture Diagram (Conceptual)
```mermaid
graph TD
    A[Video Sources: Webcam, Files, RTSP/IP Cam] --> B{Video Ingestion & Pre-processing}
    B --> C{Frame Buffer / Queue}
    C --> D{Dynamic UI-Driven Mode Selector}
    D --> E1[Mode 1: Lost Person Search]
    D --> E2[Mode 2: Member Attendance Logger]
    D --> E3[Mode 3: Threat & Weapon Detection]

    E1 --> F1[Facial Recognition (DeepFace: Facenet/ArcFace)]
    E2 --> F1

    F1 -- Match --> G1[Subject Profile Matching]
    F1 -- No Match in Mode 2 --> F3[Unknown Person Re-ID]

    F3 -- Known Unknown --> G2[Update Unknown DB]
    F3 -- New Unknown --> G3[Register New Unknown]
    
    G1 --> H[Audit & Logging Engine]
    G2 --> H
    G3 --> H
    
    E3 --> F2[Heuristic/Contour & Deep Model Weapon Detection]
    F2 --> H

    H --> I[Streamlit Dashboard (Modern Dark UI)]
    J[RBAC Gateway] --> I
    J --> K[Subject Profile Registration Engine (Admin Only)]
    K --> G1

    I -- User Interaction --> D
    I -- User Interaction --> K
    I -- Admin/Operator --> J
```

### Module Workflow Breakdown:

1.  **RBAC Gateway (Role-Based Access Control):**
    *   **Function:** Authenticates users and authorizes access based on predefined roles (Admin, Operator).
    *   **Workflow:** User attempts to access system -> RBAC Gateway verifies credentials and role -> Grants appropriate access levels to UI features and modules (e.g., Admin can register subjects, Operators can monitor).

2.  **Subject Profile Registration Engine (Admin Module):**
    *   **Function:** Admin-only module for enrolling new subjects (lost persons, staff, members) into the system's database.
    *   **Workflow:** Admin uploads subject image/video -> System extracts facial embeddings (via DeepFace) -> Stores embeddings and metadata in a secure database -> New subject becomes trackable.

3.  **Dynamic Video Feed Ingestion:**
    *   **Function:** Decodes and processes video streams from multiple sources.
    *   **Workflow:** User selects video source (Webcam, File, RTSP URL) -> RTSP/TCP decoders or OpenCV video capture module initiates stream -> Frames are extracted and buffered for downstream processing.

4.  **Dynamic UI-Driven Mode Selector:**
    *   **Function:** Allows the operator to switch between operational modes.
    *   **Workflow:** Operator selects desired mode via Streamlit UI -> System configures the processing pipeline to activate specific AI models and logic corresponding to the chosen mode.

5.  **Mode 1: Lost Person Search (Target Alert Mode):**
    *   **Function:** Continuously scans video feeds for a specific pre-registered target.
    *   **Workflow:** Incoming frames -> Facial Detection -> Facial Embedding Extraction (DeepFace) -> Comparison with registered target profiles -> If match, trigger "Target Alert" -> Log event in Audit & Logging Engine.

6.  **Mode 2: Member Attendance Logger:**
    *   **Function:** Identifies registered members vs. unknown individuals in a designated area.
    *   **Workflow:** Incoming frames -> Facial Detection -> Facial Embedding Extraction (DeepFace) -> Comparison with registered member profiles -> If match, categorizes as "Registered Member". If no match, triggers the **Unknown Person Tracking** workflow.

7.  **Mode 2a: Unknown Person Tracking & Re-identification:**
    *   **Function:** Automatically logs, tracks, and re-identifies individuals not present in the registered member database.
    *   **Workflow:**
        *   An unknown face is detected. The system compares it against the `unknown_faces` database.
        *   **Re-identification:** If a match is found, the system retrieves the person's unique ID, updates their `last_seen_timestamp`, and logs the sighting. The UI displays the ID and first-seen date.
        *   **New Registration:** If no match is found, the system saves the face, assigns a new unique ID (e.g., `unknown_001`), and creates a new entry in the `unknown_person_db.csv` and `unknown_sighting_log.csv`.

8.  **Mode 3: Threat & Weapon Pattern Detection Mode:**
    *   **Function:** Detects hazardous objects or weapons in the video stream.
    *   **Workflow:** Incoming frames -> Object Detection (Heuristic/Contour Analysis for general shapes, Deep Learning Model for specific weapon types) -> If weapon detected -> Trigger "Threat Alert" -> Log event in Audit & Logging Engine.

9.  **Structured Audit & Logging Engine:**
    *   **Function:** Records all significant events, alerts, and identifications.
    *   **Workflow:** Events from operational modes (alerts, identifications) are timestamped and captured -> Data written to real-time CSV logs -> Data is presented in interactive, downloadable Pandas data tables via Streamlit UI.

10. **Streamlit Dashboard (Modern Dark UI):**
    *   **Function:** Provides the central interactive interface for system control, monitoring, and data visualization.
    *   **Workflow:** Receives processed data and alerts from various modules -> Renders real-time video feeds with overlays, alerts, and statistical summaries -> Allows user interaction for mode selection, subject registration (for Admin), and log retrieval.

## 7. FEASIBILITY STUDY

### Technical Feasibility:
The project is technically feasible, leveraging mature and well-documented open-source technologies. Python 3.10+ offers a robust ecosystem for AI and computer vision development. Streamlit provides a rapid development environment for interactive dashboards, while OpenCV is the de facto standard for real-time image processing. The DeepFace framework abstracts the complexity of state-of-the-art facial recognition models (Facenet, ArcFace), making their integration practical. Pandas is highly efficient for data manipulation and logging. The integration of RTSP/TCP decoders is standard practice for IP camera systems. The primary technical challenge lies in optimizing real-time performance for high-resolution, multiple-stream scenarios, requiring careful resource management and potentially multiprocessing.

### Hardware Feasibility:
*   **Minimum Requirements:**
    *   CPU: Multi-core processor (Intel i5/i7 equivalent or better).
    *   RAM: 8GB+ for basic operations; 16GB+ recommended for multiple streams/models.
    *   Storage: 256GB SSD (for OS and application); additional HDD for long-term video storage.
    *   GPU: Dedicated NVIDIA GPU (GTX 1060 or equivalent) with CUDA support is highly recommended for efficient DeepFace and deep model weapon detection inference, especially for real-time applications.
*   **Optimal Requirements:**
    *   CPU: High-performance CPU (Intel i7/i9, AMD Ryzen 7/9).
    *   RAM: 32GB+
    *   Storage: 500GB+ NVMe SSD.
    *   GPU: High-end NVIDIA GPU (RTX 3060/4060 or better) for parallel processing of multiple high-resolution streams and complex AI models.
    *   Cameras: High-resolution IP cameras with RTSP/ONVIF support for optimal surveillance quality.

### Software Feasibility:
All core software components are open-source and readily available:
*   **Operating System:** Windows, Linux (Ubuntu recommended for easier dependency management and GPU driver support).
*   **Programming Language:** Python 3.10+
*   **Key Libraries/Frameworks:** Streamlit, OpenCV-Python, DeepFace, Pandas, PIL, numpy.
*   **Virtual Environment:** Use of `venv` or `conda` is essential for managing dependencies.
*   **Development Tools:** VS Code or similar IDE with Python extensions.
The project's reliance on established libraries mitigates significant software development risks. Compatibility and dependency management will be key considerations.

### Skill Requirements:
*   **Python Programming:** Advanced proficiency in Python is fundamental.
*   **Computer Vision:** Strong understanding of OpenCV, image processing techniques, and video stream manipulation.
*   **Deep Learning/Machine Learning:** Experience with deep learning frameworks (TensorFlow/Keras, PyTorch - implicitly used by DeepFace), model deployment, and understanding of CNN architectures.
*   **Facial Recognition:** Familiarity with concepts like face detection, alignment, embedding extraction, and similarity metrics.
*   **UI/UX Development:** Experience with Streamlit or similar rapid web application frameworks for dashboard creation.
*   **Database Concepts:** Basic understanding of data storage (CSV, potentially lightweight databases for profiles).
*   **Network Protocols:** Knowledge of RTSP/TCP for IP camera integration.
*   **Software Architecture:** Ability to design modular, scalable, and maintainable systems.

## 8. REFERENCES

1.  **Schroff, F., Kalenichenko, D., & Philbin, J. (2015).** *FaceNet: A Unified Embedding for Face Recognition and Clustering*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [Modern AI - Facial Recognition]
2.  **Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019).** *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [Modern AI - Facial Recognition]
3.  **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).** *You Only Look Once: Unified, Real-Time Object Detection*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [Computer Vision - Object Detection]
4.  **OpenCV Foundation. (Current Year).** *Open Source Computer Vision Library*. Retrieved from `https://opencv.org/` [Computer Vision - Standard Library]
5.  **Abadi, M., et al. (2016).** *TensorFlow: A System for Large-Scale Machine Learning*. OSDI. (Underlying framework for many DeepFace models). [Modern AI - Deep Learning Framework]
6.  **Chollet, F. (2017).** *Deep Learning with Python*. Manning Publications. (General deep learning principles relevant to model understanding). [Modern AI - Deep Learning Concepts]
7.  **RTSP (Real-Time Streaming Protocol) RFC 2326.** (Describes the protocol for controlling the delivery of data with real-time properties). [Surveillance Standards - Streaming Protocol]
8.  **Streamlit Documentation.** Retrieved from `https://docs.streamlit.io/` [Modern UI/Dashboard Development]
9.  **Pandas Development Team. (Current Year).** *pandas: powerful Python data analysis and manipulation library*. Retrieved from `https://pandas.pydata.org/` [Data Analysis & Logging]

## 9. Current implementation update (2026-08-09)

The current application extends the original synopsis with the following operational behavior:

- The Administrator and Operator dashboards both show registered-face and unknown-face totals.
- Registered profiles are classified as `Staff` or `Victim`. Existing `Member_` files remain Staff and existing `Lost_` files remain Victim.
- The registered-face inventory can be filtered with `All`, `Staff`, and `Victim` options.
- `Lost Person Search` requires the operator to select a Victim target. Only that selected Victim is eligible for a visible match.
- Other faces are silently stored or re-identified as unknowns. Their IDs are not drawn on the Victim-search screen.
- The operator supplies a camera location. A successful Victim match displays the Victim name, current location, and the latest sighting for each previously recorded location.
- Victim sightings are stored in `detection_logs/victim_sighting_log.csv`; unknown data remains in `unknown_person_db.csv` and `unknown_sighting_log.csv`.
- A browser/network disconnect during the synchronous live loop can produce Windows `WinError 10054`. This is a closed Streamlit client connection, not an identity-match result.
