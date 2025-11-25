# detector.py
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

class LungAssistantDetector:
    def __init__(self, model_path=None):
        # choose model_path: best.pt if present else yolov8n.pt
        if model_path and os.path.exists(model_path):
            path = model_path
        else:
            path = "yolov8n.pt"   # fallback
        print(f"[detector] Loading model: {path}")
        self.model = YOLO(path)
        self.last_detection_time = 0
        self.is_smoking = False

    def detect_smoking(self, frame, conf_thresh=0.35):
        """
        frame: RGB image (H,W,3)
        returns: (is_smoking:bool, detections:list)
          detections: list of (class_name, conf, (x1,y1,x2,y2))
        """
        results = self.model(frame, conf=conf_thresh)
        # results may be list-like
        res = results[0] if isinstance(results, (list, tuple)) else results

        detections = []
        person_boxes = []
        object_boxes = []

        for box in getattr(res, "boxes", []):
            try:
                cls = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
                conf = float(box.conf[0]) if hasattr(box, "conf") else float(box.conf)
                xy = box.xyxy[0] if hasattr(box, "xyxy") else None
                if xy is None:
                    continue
                x1, y1, x2, y2 = map(int, xy.tolist())
                name = res.names.get(cls, str(cls)) if getattr(res, "names", None) else str(cls)
                detections.append((name, conf, (x1, y1, x2, y2)))
                area = (x2 - x1) * (y2 - y1)
                if name == "person":
                    person_boxes.append((x1, y1, x2, y2))
                else:
                    # treat small objects as candidate hand-held items
                    if area < 80000:
                        object_boxes.append((name, x1, y1, x2, y2))
            except Exception:
                continue

        # Heuristic: object near mouth OR smoke-like pixels -> smoking_detected
        smoking_detected = False

        for (px1, py1, px2, py2) in person_boxes:
            face_y_end = py1 + int(0.35 * (py2 - py1))  # top region as face
            face_center = ((px1 + px2) // 2, (py1 + face_y_end) // 2)
            for (oname, ox1, oy1, ox2, oy2) in object_boxes:
                obj_center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
                dist = np.linalg.norm(np.array(face_center) - np.array(obj_center))
                if dist < 140:
                    # object close to face -> suspicious
                    smoking_detected = True

        # Smoke-color detection (grayish pixels near face area)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        smoke_mask = cv2.inRange(gray, 160, 230)
        smoke_pixels = cv2.countNonZero(smoke_mask)
        if smoke_pixels > 400:
            smoking_detected = True

        # temporal smoothing: require detection for 1.5 seconds to confirm
        cur = time.time()
        if smoking_detected:
            if not self.is_smoking:
                self.last_detection_time = cur
            if cur - self.last_detection_time > 1.5:
                self.is_smoking = True
        else:
            # small cooldown
            if cur - self.last_detection_time > 2.5:
                self.is_smoking = False

        return self.is_smoking, detections
