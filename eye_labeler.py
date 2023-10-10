import cv2
import os
import json

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifiers for face
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create a directory to save the images and labels
os.makedirs("dataset", exist_ok=True)

# Create a window
cv2.namedWindow("Face and Eye Detection")

frame_count = 100
face_eye_dict = {}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Create a copy of the frame before drawing rectangles
    original_frame = frame.copy()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get the region of interest to find eyes
        roi_color = frame[y : y + h, x : x + w]

        # Convert the ROI to HSV color space
        hsv_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

        # Create a mask for white color (common for sclera)
        lower_white = (0, 0, 200)
        upper_white = (180, 255, 255)
        mask = cv2.inRange(hsv_roi, lower_white, upper_white)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        eye_labels = []
        for cnt in contours:
            ex, ey, ew, eh = cv2.boundingRect(cnt)

            # Ignore small contours that are not likely to be eyes
            if ew * eh < 100:
                continue

            # Adjust eye coordinates relative to the whole frame
            ex, ey = ex + x, ey + y

            # Draw rectangle around the eyes
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_labels.append((ex, ey, ew, eh))

        # Associate eyes with the face
        face_eye_dict[str((x, y, w, h))] = eye_labels

    # Display the frame with face and eye rectangles
    cv2.imshow("Face and Eye Detection", frame)

    # Wait for user input
    key = cv2.waitKey(1)

    # Press 'q' to quit the application
    if key == ord("q"):
        break

    # Press 's' to save the current frame and labels
    elif key == ord("s"):
        cv2.imwrite(f"dataset/frame_{frame_count}.png", original_frame)
        with open(f"dataset/labels_{frame_count}.json", "w") as f:
            json.dump(face_eye_dict, f)
        frame_count += 1

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
