import cv2
import os
import json

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Create a directory to save the images and labels
os.makedirs('dataset', exist_ok=True)

frame_count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    labels = []
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get the region of interest to find eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            labels.append((ex, ey, ew, eh))

    # Save the frame and labels
    cv2.imwrite(f'dataset/frame_{frame_count}.png', frame)
    with open(f'dataset/labels_{frame_count}.json', 'w') as f:
        json.dump(labels, f)

    frame_count += 1

    # Display the frame with face and eye rectangles
    cv2.imshow('Face and Eye Detection', frame)

    # Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()