import cv2
import numpy as np
import torch
import argparse
import time
from ultralytics import YOLO
import face_recognition
import os
from pathlib import Path

class WebcamFacePhoneDetector:
    def __init__(self, model_path=None, conf_threshold=0.3, device='cuda', face_db_path='C:\Users\Keith\Pictures\Webcam'):
        """
        Initialize webcam detector for faces and phones with geofence and movement detection.

        Args:
            model_path (str): Path to YOLOv11 model weights (None for default).
            conf_threshold (float): Confidence threshold for YOLO detections.
            device (str): Device for inference ('cuda' or 'cpu').
            face_db_path (str): Path to face image database.
        """
        self.conf_threshold = conf_threshold
        self.device = 'cpu' if device == 'cuda' and not torch.cuda.is_available() else device
        self.face_db_path = face_db_path

        # Load YOLO model
        self.model = YOLO(model_path if model_path else 'yolo11m.pt')
        self.model.to(self.device)

        # Target classes (COCO: 0=person, 67=cell phone)
        self.target_classes = {'person': 0, 'phone': 67}

        # Load Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")

        # Face recognition database
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_face_database()

        # Phone tracking
        self.prev_phone_centers = []  # List of (x, y) centers from previous frame
        self.movement_threshold = 20  # Pixels for movement detection

    def _load_face_database(self):
        """Load face encodings from database images."""
        if not os.path.isdir(self.face_db_path):
            print(f"Error: Directory {self.face_db_path} does not exist")
            return

        for img_path in Path(self.face_db_path).glob('*.[jp][pn][gf]'):
            try:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) != 1:
                    print(f"Skipped {img_path.name}: Found {len(encodings)} faces (expected 1)")
                    continue
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(img_path.stem)
                print(f"Loaded face: {img_path.stem}")
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")

        if not self.known_face_encodings:
            print("Warning: No valid face images loaded")

    def _recognize_face(self, face_image):
        """
        Recognize a face by comparing to known faces.

        Args:
            face_image (np.ndarray): Face image in RGB format.

        Returns:
            str: Name of matched person or 'Unknown'.
        """
        encodings = face_recognition.face_encodings(face_image)
        if not encodings:
            return 'Unknown'

        encoding = encodings[0]
        if not self.known_face_encodings:
            return 'Unknown'

        distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.6)
        best_match_idx = np.argmin(distances) if distances.size > 0 else -1
        return self.known_face_names[best_match_idx] if best_match_idx >= 0 and matches[best_match_idx] else 'Unknown'

    def _is_outside_geofence(self, center, geofence):
        """
        Check if a point is outside the geofence.

        Args:
            center (tuple): (x, y) coordinates of the point.
            geofence (tuple): (x1, y1, x2, y2) geofence boundaries.

        Returns:
            bool: True if outside geofence.
        """
        x, y = center
        x1, y1, x2, y2 = geofence
        return not (x1 <= x <= x2 and y1 <= y <= y2)

    def _detect_movement(self, current_centers):
        """
        Detect phone movement by comparing centers to previous frame.

        Args:
            current_centers (list): List of (x, y) centers in current frame.

        Returns:
            bool: True if any phone is moving.
        """
        if not self.prev_phone_centers or not current_centers:
            self.prev_phone_centers = current_centers
            return False

        moving = False
        for curr_center in current_centers:
            min_dist = float('inf')
            for prev_center in self.prev_phone_centers:
                dist = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                min_dist = min(min_dist, dist)
            if min_dist > self.movement_threshold:
                moving = True
                print("Phone is moving")

        self.prev_phone_centers = current_centers
        return moving

    def process_webcam(self):
        """Process webcam stream for face and phone detection."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Processing webcam stream (press 'q' to quit)...")
        start_time = time.time()
        frame_count = 0

        # Get frame dimensions for geofence
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read webcam frame")
            cap.release()
            return
        h, w = frame.shape[:2]
        margin = 0.1  # 10% margin
        geofence = (int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin)))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for efficiency
            max_size = 640
            scale = min(max_size / w, max_size / h)
            resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
            scale = scale if scale < 1 else 1.0

            processed_frame = frame.copy()
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # YOLO inference
            results = self.model(rgb_frame, conf=self.conf_threshold)
            detections = []
            phone_centers = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    if cls_id not in self.target_classes.values():
                        continue

                    conf = box.conf.item()
                    x1, y1, x2, y2 = [int(coord / scale) for coord in box.xyxy[0].tolist()]

                    if cls_id == self.target_classes['person']:
                        person_roi = frame[y1:y2, x1:x2]
                        if person_roi.size == 0:
                            continue

                        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5)

                        for (fx, fy, fw, fh) in faces:
                            face_x1 = x1 + fx
                            face_y1 = y1 + fy
                            face_x2 = face_x1 + fw
                            face_y2 = face_y1 + fh

                            face_rgb = cv2.cvtColor(frame[face_y1:face_y2, face_x1:face_x2], cv2.COLOR_BGR2RGB)
                            name = self._recognize_face(face_rgb)
                            if name == 'Unknown':
                                print("Unknown face detected")

                            detections.append({
                                'class': 'face',
                                'confidence': conf,
                                'bbox': (face_x1, face_y1, face_x2, face_y2),
                                'name': name
                            })

                            cv2.rectangle(processed_frame, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)
                            cv2.putText(processed_frame, f'{name} {conf:.2f}', (face_x1, face_y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    elif cls_id == self.target_classes['phone']:
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 0
                        if 0.3 <= aspect_ratio <= 3.0:
                            detections.append({
                                'class': 'phone',
                                'confidence': conf,
                                'bbox': (x1, y1, x2, y2)
                            })

                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            phone_centers.append(center)
                            if self._is_outside_geofence(center, geofence):
                                print("Transformer out of geofence")

                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(processed_frame, f'Transformer {conf:.2f}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Detect phone movement
            self._detect_movement(phone_centers)

            # Draw geofence
            cv2.rectangle(processed_frame, (geofence[0], geofence[1]), (geofence[2], geofence[3]), (0, 255, 0), 1)

            # Display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                start_time = time.time()
                frame_count = 0

            cv2.imshow('Webcam Face and Phone Detector', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Webcam Face and Phone Detector with Recognition')
    parser.add_argument('--model', type=str, default=None, help='Path to YOLOv11 model weights')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--face-db', type=str, default='C:\Users\Keith\Pictures\Webcam', help='Path to face image database')

    args = parser.parse_args()

    detector = WebcamFacePhoneDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
        face_db_path=args.face_db
    )
    detector.process_webcam()

if __name__ == '__main__':
    main()