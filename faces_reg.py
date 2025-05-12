import cv2
import os
import time

# Directory to save images
save_dir = "C:\Users\Keith\Pictures\Webcam"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

# Define person’s name (Change as needed)
person_name = "Tinotenda"  

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 1
total_images = 30

while count <= total_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    filename = os.path.join(save_dir, f"{person_name}-{count:02d}.jpg")
    
    # Save image
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

    count += 1
    time.sleep(0.5)  # Wait to avoid duplicate frames

    # Show frame (optional)
    cv2.imshow("Capturing Images", frame)
    
    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Image capture complete!")
