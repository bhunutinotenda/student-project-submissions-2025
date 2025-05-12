import cv2
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from deepface import DeepFace
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Use a faster backend (mtcnn is generally faster)
detector_backend = "mtcnn"
model_name = "ArcFace"

# Path to the reference images
reference_images_folder = "C:\Users\Keith\Pictures\Webcam"


# Load all reference images and precompute embeddings
reference_embeddings = []
reference_labels = []

for filename in os.listdir(reference_images_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(reference_images_folder, filename)
        try:
            embedding = DeepFace.represent(image_path, model_name=model_name, detector_backend=detector_backend, enforce_detection=True)
            if embedding:
                reference_embeddings.append(embedding[0]["embedding"])
                reference_labels.append(filename.split("-")[0])  # Extract name before '-'
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Global variables
recognized_faces = {}  # Stores recognized faces with timestamps
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=4)
TEXT_DISPLAY_TIME = 0.25  # Time (in seconds) to display name before clearing

# Function to recognize faces
def recognize_face(face_image, face_coords):
    x, y, w, h = face_coords
    try:
        embedding = DeepFace.represent(face_image, model_name=model_name, detector_backend=detector_backend, enforce_detection=False)
        if not embedding:
            with lock:
                recognized_faces[(x, y, w, h)] = ("UNKNOWN", time.time())
            return

        embedding_vector = np.array(embedding[0]["embedding"]).reshape(1, -1)
        best_match = "UNKNOWN"
        max_similarity = -1

        for ref_emb, label in zip(reference_embeddings, reference_labels):
            similarity = cosine_similarity(embedding_vector, np.array(ref_emb).reshape(1, -1))[0][0]
            if similarity > max_similarity and similarity > 0.75:  # Increased threshold for better accuracy
                max_similarity = similarity
                best_match = label

        with lock:
            recognized_faces[(x, y, w, h)] = (best_match, time.time())  # Store with timestamp

    except Exception as e:
        print(f"Error in face recognition: {e}")
        with lock:
            recognized_faces[(x, y, w, h)] = ("UNKNOWN", time.time())

# Helper function to find closest recognized face
def find_closest_face(x, y, w, h):
    with lock:
        for (fx, fy, fw, fh), (label, timestamp) in recognized_faces.items():
            face_center = (fx + fw // 2, fy + fh // 2)
            new_center = (x + w // 2, y + h // 2)
            if np.linalg.norm(np.array(face_center) - np.array(new_center)) < 20:
                return label
    return "UNKNOWN"

# Start the video stream
def start_video_stream():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_skip = 5  # Process face detection every 5 frames
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip == 0:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (320, 240))

            # Detect faces using a faster backend
            faces = DeepFace.extract_faces(small_frame, detector_backend=detector_backend, enforce_detection=False)

            for face_info in faces:
                x, y, w, h = face_info["facial_area"]["x"], face_info["facial_area"]["y"], face_info["facial_area"]["w"], face_info["facial_area"]["h"]
                
                # Scale back coordinates to full frame size
                x, y, w, h = int(x * 2), int(y * 2), int(w * 2), int(h * 2)

                face_roi = frame[y:y+h, x:x+w]
                executor.submit(recognize_face, face_roi, (x, y, w, h))

        # Draw recognized faces
        current_time = time.time()
        with lock:
            keys_to_remove = []
            for (x, y, w, h), (label, timestamp) in recognized_faces.items():
                if current_time - timestamp < TEXT_DISPLAY_TIME:
                    color = (255, 0, 0) if label != "UNKNOWN" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                else:
                    keys_to_remove.append((x, y, w, h))

            # Remove expired names
            for key in keys_to_remove:
                del recognized_faces[key]

        # Display the frame
        cv2.imshow("Fast Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=True)

if __name__ == "__main__":
    start_video_stream()
