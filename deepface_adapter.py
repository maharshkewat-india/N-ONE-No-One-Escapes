from __future__ import annotations

import importlib
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent
DEEPFACE_ROOT = ROOT_DIR / "deepface"
if str(DEEPFACE_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPFACE_ROOT))


class OpenCVFaceBackend:
    """OpenCV-only facial recognition backend - no TensorFlow/PyTorch required."""

    def __init__(self, error_message: str | None = None) -> None:
        self.error_message = error_message or "OpenCV fallback mode active."
        self.face_cascade = None
        self.profile_cascade = None
        self.recognizer = None
        self._init_opencv()

    def _init_opencv(self) -> None:
        """Initialize OpenCV face detection and recognition."""
        # Face detection and LBPH recognition are independent capabilities.
        # opencv-python-headless does not provide cv2.face, but it still
        # provides the Haar cascade needed by the fallback detector.
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
                self.error_message = "OpenCV Haar cascade could not be loaded."
        except Exception as exc:
            self.face_cascade = None
            self.error_message = f"OpenCV face-detector initialization error: {exc}"

        # A frontal-only cascade misses people who are looking sideways.  The
        # profile cascade is optional, so keep the frontal detector usable if
        # this file is not present in a minimal OpenCV installation.
        try:
            profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
            self.profile_cascade = cv2.CascadeClassifier(profile_path)
            if self.profile_cascade.empty():
                self.profile_cascade = None
        except Exception:
            self.profile_cascade = None

        try:
            # Optional: not required by this adapter's embedding pipeline.
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except Exception:
            self.recognizer = None

    def detect_faces(self, img: np.ndarray) -> List[Dict]:
        """Detect frontal and profile faces using OpenCV Haar cascades.

        The fallback is used when the full DeepFace runtime is unavailable.
        Running on an equalized image, checking both profile directions, and
        retrying an upscaled frame helps with dim footage and small faces while
        keeping the returned coordinates in the original frame's space.
        """
        if self.face_cascade is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray = cv2.equalizeHist(gray)

        def detect_with_cascade(
            cascade: Any | None,
            image: np.ndarray,
            coordinate_scale: float = 1.0,
            mirrored: bool = False,
        ) -> List[Dict]:
            if cascade is None:
                return []
            detected = cascade.detectMultiScale(
                image,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(24, 24),
            )
            image_width = image.shape[1]
            results = []
            for x, y, w, h in detected:
                if mirrored:
                    x = image_width - x - w
                results.append(
                    {
                        "facial_area": {
                            "x": max(0, int(x / coordinate_scale)),
                            "y": max(0, int(y / coordinate_scale)),
                            "w": max(1, int(w / coordinate_scale)),
                            "h": max(1, int(h / coordinate_scale)),
                        },
                        "confidence": 1.0,
                    }
                )
            return results

        results = detect_with_cascade(self.face_cascade, gray)
        results.extend(detect_with_cascade(self.profile_cascade, gray))
        # The profile cascade is left-facing.  Mirror the image to detect the
        # opposite profile, then mirror the coordinates back.
        results.extend(
            detect_with_cascade(
                self.profile_cascade,
                cv2.flip(gray, 1),
                mirrored=True,
            )
        )

        # If the first pass found nothing, enlarge the frame so faces smaller
        # than the cascade's native minimum size still get a chance.
        if not results and min(gray.shape[:2]) < 900:
            scale = 1.5
            enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            results.extend(detect_with_cascade(self.face_cascade, enlarged, scale))
            results.extend(detect_with_cascade(self.profile_cascade, enlarged, scale))
            results.extend(
                detect_with_cascade(
                    self.profile_cascade,
                    cv2.flip(enlarged, 1),
                    scale,
                    mirrored=True,
                )
            )

        return self._deduplicate_faces(results)

    @staticmethod
    def _deduplicate_faces(faces: List[Dict]) -> List[Dict]:
        """Remove overlapping Haar detections for the same face."""
        kept: List[Dict] = []
        for candidate in sorted(
            faces,
            key=lambda item: item["facial_area"]["w"] * item["facial_area"]["h"],
            reverse=True,
        ):
            box = candidate["facial_area"]
            candidate_area = box["w"] * box["h"]
            overlaps = False
            for existing in kept:
                other = existing["facial_area"]
                left = max(box["x"], other["x"])
                top = max(box["y"], other["y"])
                right = min(box["x"] + box["w"], other["x"] + other["w"])
                bottom = min(box["y"] + box["h"], other["y"] + other["h"])
                overlap_area = max(0, right - left) * max(0, bottom - top)
                if overlap_area / max(1, candidate_area) >= 0.35:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(candidate)
        return kept

    def extract_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract a lighting-tolerant fallback feature vector from a face ROI.

        This is not a replacement for a learned face-recognition model, but a
        HOG descriptor is substantially more stable than comparing raw gray
        pixels when the same person changes lighting, scale, or expression.
        """
        try:
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi

            resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            normalized = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            ).apply(resized)

            # HOG captures stable facial structure (eye sockets, nose, mouth,
            # and jaw edges) while reducing sensitivity to brightness changes.
            descriptor = cv2.HOGDescriptor(
                _winSize=(64, 64),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9,
            )
            embedding = descriptor.compute(normalized).reshape(-1).astype(np.float64)
            embedding /= np.linalg.norm(embedding) + 1e-8
            return embedding

        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None

    def _compute_lbp(self, image: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
        """Compute Local Binary Pattern texture features."""
        try:
            # Simple LBP implementation
            lbp = np.zeros_like(image)
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    center = image[i, j]
                    code = 0
                    for k in range(P):
                        # Sample points on circle
                        x = int(np.round(i + R * np.cos(2 * np.pi * k / P)))
                        y = int(np.round(j + R * np.sin(2 * np.pi * k / P)))
                        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                            if image[x, y] >= center:
                                code |= (1 << k)
                    lbp[i, j] = code
            return lbp
        except:
            return image  # fallback

    def represent(self, img_path: str, model_name: str = "VGG-Face",
                  enforce_detection: bool = True, **kwargs) -> List[Dict]:
        """Face representation using OpenCV - mimics DeepFace.represent interface."""
        try:
            # Load image
            if isinstance(img_path, str):
                img = cv2.imread(img_path)
                if img is None:
                    if enforce_detection:
                        raise ValueError(f"Unable to load image: {img_path}")
                    return []
            else:
                # Assume it's already a numpy array
                img = img_path

            # Detect faces
            faces = self.detect_faces(img)
            if not faces and enforce_detection:
                return []

            # Unknown faces are stored as cropped face images. They may no
            # longer contain enough surrounding context for Haar detection,
            # so the find() path can explicitly use the whole crop.
            if not faces and kwargs.get("allow_full_image", False):
                height, width = img.shape[:2]
                if min(height, width) >= 32:
                    embedding = self.extract_embedding(img)
                    if embedding is not None:
                        return [{
                            "embedding": embedding.tolist(),
                            "facial_area": {"x": 0, "y": 0, "w": width, "h": height},
                            "confidence": 1.0,
                        }]

            results = []
            for face_info in faces:
                # Extract face ROI
                facial_area = face_info['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                face_roi = img[y:y+h, x:x+w]

                # Extract embedding
                embedding = self.extract_embedding(face_roi)
                if embedding is not None:
                    results.append({
                        "embedding": embedding.tolist(),
                        "facial_area": facial_area,
                        "confidence": face_info.get('confidence', 1.0)
                    })

            return results

        except Exception as e:
            if enforce_detection:
                raise e
            return []

    def find(self, img_path: str, db_path: str, model_name: str = "VGG-Face",
             distance_metric: str = "cosine", enforce_detection: bool = True,
             silent: bool = False, **kwargs) -> List[Any]:
        """Find faces in database using OpenCV - mimics DeepFace.find interface."""
        try:
            # Load target image and get its embedding
            target_representations = self.represent(
                img_path,
                model_name,
                enforce_detection,
                silent=True,
                allow_full_image=True,
            )
            if not target_representations:
                return [pd.DataFrame()]  # Return empty DataFrame in list to match DeepFace format

            target_embedding = np.array(target_representations[0]['embedding'])

            # Load all images from db_path
            db_path = Path(db_path)
            if not db_path.exists():
                return [pd.DataFrame()]

            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(db_path.glob(ext))
                image_files.extend(db_path.glob(ext.upper()))

            if not image_files:
                return [pd.DataFrame()]

            # Compare with each image in database
            results = []
            thresholds = {
                'cosine': 0.4,
                'euclidean': 0.55,
                'euclidean_l2': 0.75
            }
            threshold = thresholds.get(distance_metric, 0.4)

            for img_file in image_files:
                try:
                    db_representations = self.represent(
                        str(img_file),
                        model_name,
                        False,
                        silent=True,
                        allow_full_image=True,
                    )
                    if db_representations:
                        db_embedding = np.array(db_representations[0]['embedding'])

                        # Calculate distance
                        if distance_metric == 'cosine':
                            from scipy.spatial.distance import cosine
                            distance = cosine(target_embedding, db_embedding)
                        elif distance_metric == 'euclidean':
                            distance = np.linalg.norm(target_embedding - db_embedding)
                        elif distance_metric == 'euclidean_l2':
                            distance = np.linalg.norm(target_embedding - db_embedding) / \
                                     (np.linalg.norm(target_embedding) + np.linalg.norm(db_embedding))
                        else:
                            # Default to cosine
                            from scipy.spatial.distance import cosine
                            distance = cosine(target_embedding, db_embedding)

                        if distance <= threshold:
                            results.append({
                                'identity': str(img_file.absolute()),
                                'target_x': 0, 'target_y': 0, 'target_w': 50, 'target_h': 50,
                                'source_x': 0, 'source_y': 0, 'source_w': 50, 'source_h': 50,
                                'distance': distance
                            })
                except Exception:
                    continue  # Skip problematic images

            # Return as DataFrame in list (matching DeepFace format)
            if results:
                import pandas as pd
                df = pd.DataFrame(results)
                return [df]
            else:
                import pandas as pd
                return [pd.DataFrame()]

        except Exception as e:
            if not silent:
                print(f"Find error: {e}")
            import pandas as pd
            return [pd.DataFrame()]

    # Additional methods to match DeepFace interface
    def verify(self, img1_path: str, img2_path: str, model_name: str = "VGG-Face",
               distance_metric: str = "cosine", enforce_detection: bool = True,
               silent: bool = False, **kwargs) -> Dict:
        """Verify if two faces match - mimics DeepFace.verify."""
        try:
            rep1 = self.represent(img1_path, model_name, enforce_detection, silent=True)
            rep2 = self.represent(img2_path, model_name, enforce_detection, silent=True)

            if not rep1 or not rep2:
                return {
                    "verified": False,
                    "distance": 1.0,
                    "threshold": 0.4,
                    "model": model_name,
                    "similarity_metric": distance_metric
                }

            emb1 = np.array(rep1[0]['embedding'])
            emb2 = np.array(rep2[0]['embedding'])

            # Calculate distance
            if distance_metric == 'cosine':
                from scipy.spatial.distance import cosine
                distance = cosine(emb1, emb2)
            elif distance_metric == 'euclidean':
                distance = np.linalg.norm(emb1 - emb2)
            elif distance_metric == 'euclidean_l2':
                distance = np.linalg.norm(emb1 - emb2) / \
                         (np.linalg.norm(emb1) + np.linalg.norm(emb2))
            else:
                from scipy.spatial.distance import cosine
                distance = cosine(emb1, emb2)

            threshold = {'cosine': 0.4, 'euclidean': 0.55, 'euclidean_l2': 0.75}.get(distance_metric, 0.4)
            verified = distance <= threshold

            return {
                "verified": verified,
                "distance": float(distance),
                "threshold": threshold,
                "model": model_name,
                "similarity_metric": distance_metric
            }
        except Exception as e:
            if not silent:
                print(f"Verify error: {e}")
            return {
                "verified": False,
                "distance": 1.0,
                "threshold": 0.4,
                "model": model_name,
                "similarity_metric": distance_metric
            }


# Try to import real DeepFace first
_BACKEND: Any | None = None
_DEEPFACE_IMPORT_ERROR: str | None = None

def load_deepface_backend() -> Any:
    global _BACKEND, _DEEPFACE_IMPORT_ERROR

    if _BACKEND is not None:
        return _BACKEND

    # Try to load real DeepFace first
    try:
        # The DeepFace class is in the `DeepFace` module within the `deepface` package
        deepface_module = importlib.import_module("deepface.DeepFace")
        backend = getattr(deepface_module, "DeepFace", None)
        if backend is None:
            # This error is more specific and should not happen if the submodule structure is correct
            raise AttributeError("The 'deepface.DeepFace' module does not expose the 'DeepFace' class.")
        _BACKEND = backend
        _DEEPFACE_IMPORT_ERROR = None
        return _BACKEND
    except Exception as exc:  # pragma: no cover - defensive fallback for runtime environments
        _DEEPFACE_IMPORT_ERROR = str(exc)
        # Use OpenCV fallback
        _BACKEND = OpenCVFaceBackend(_DEEPFACE_IMPORT_ERROR)
        return _BACKEND


DeepFace = load_deepface_backend()
DEEPFACE_IMPORT_ERROR = _DEEPFACE_IMPORT_ERROR

__all__ = ["DeepFace", "DEEPFACE_IMPORT_ERROR", "load_deepface_backend"]
