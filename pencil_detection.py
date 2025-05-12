import cv2
import os
import time
import numpy as np

# Directory to save images
save_dir = "C:\Users\Keith\Pictures\Webcam"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load pre-trained model for object detection
# We'll use a Haar cascade for basic object detection, but for better results
# you might want to use a more sophisticated model like YOLO or SSD
# For now, let's create a simple color-based detector for pen/pencil-like objects

def detect_pen_pencil(frame):
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for common pen/pencil colors
    # Modify these ranges based on your specific pens/pencils
    
    # Blue pen range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Black/Gray pencil range (low saturation, low value)
    lower_pencil = np.array([0, 0, 10])
    upper_pencil = np.array([180, 50, 100])
    
    # Red pen range (Hue wraps around in HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for each color
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_pencil = cv2.inRange(hsv, lower_pencil, upper_pencil)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Combine all masks
    mask = cv2.bitwise_or(mask_blue, mask_pencil)
    mask = cv2.bitwise_or(mask, mask_red)
    
    # Clean up mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by shape (pens/pencils tend to be elongated)
    pen_pencil_contours = []
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Minimum area to avoid noise
        if area > 500:
            # Get the minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            # Fix: Use astype(int) instead of np.int0
            box = box.astype(int)
            
            # Calculate aspect ratio
            width = rect[1][0]
            height = rect[1][1]
            aspect_ratio = max(width, height) / (min(width, height) + 1e-5)  # Avoid division by zero
            
            # Pens and pencils typically have high aspect ratio (elongated)
            if aspect_ratio > 3.0:
                pen_pencil_contours.append((contour, box))
    
    return pen_pencil_contours, mask

count = 1
total_images = 50

while count <= total_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break
    
    # Detect pen/pencil
    pen_pencil_contours, mask = detect_pen_pencil(frame)
    
    # Draw detection results on frame
    result_frame = frame.copy()
    detected = False
    
    for contour, box in pen_pencil_contours:
        # Draw rectangle around pen/pencil
        cv2.drawContours(result_frame, [box], 0, (0, 255, 0), 2)
        detected = True
    
    # Add text to indicate detection
    if detected:
        cv2.putText(result_frame, "Pen/Pencil Detected", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Generate filename only when pen/pencil is detected
        filename = os.path.join(save_dir, f"PenPencil-{count:02d}.jpg")
        
        # Save image
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1
    
    # Show frame with detection
    cv2.imshow("Pen/Pencil Detection", result_frame)
    
    # Show mask for debugging (optional)
    cv2.imshow("Detection Mask", mask)
    
    # Wait for 100ms between frames
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Pen/Pencil detection complete!")