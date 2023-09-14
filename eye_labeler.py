import cv2
import os
import json

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Create a directory to save the images and labels
os.makedirs("dataset", exist_ok=True)

# Create a window
cv2.namedWindow("Face and Eye Detection")

frame_count = 0
face_eye_dict = {}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Create a copy of the frame before drawing rectangles
    original_frame = frame.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get the region of interest to find eyes
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eye_labels = []
        for ex, ey, ew, eh in eyes:
            # Adjust eye coordinates
            ex += x
            ey += y
            # Draw rectangle around the eyes
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_labels.append((int(ex), int(ey), int(ew), int(eh)))

        # Associate eyes with the face
        face_eye_dict[str((x, y, w, h))] = eye_labels

    while True:
        # Display the frame with face and eye rectangles
        cv2.imshow("Face and Eye Detection", frame)

        # Wait for user to press 'Enter' to save the frame and labels, or 'd' to delete the last rectangle
        key = cv2.waitKey(0)
        if key == 13:  # Enter key
            cv2.imwrite(f"dataset/frame_{frame_count}.png", original_frame)
            with open(f"dataset/labels_{frame_count}.json", "w") as f:
                json.dump(face_eye_dict, f)
            frame_count += 1
            break
        elif key == ord("d"):
            if face_eye_dict:
                last_face = list(face_eye_dict.keys())[-1]
                if face_eye_dict[last_face]:
                    face_eye_dict[last_face].pop()
                else:
                    del face_eye_dict[last_face]
                frame = original_frame.copy()  # Create a new copy of the original frame
                for face, eyes in face_eye_dict.items():
                    face = tuple(map(int, face.strip("()").split(",")))
                    cv2.rectangle(
                        frame,
                        (face[0], face[1]),
                        (face[0] + face[2], face[1] + face[3]),
                        (255, 0, 0),
                        2,
                    )
                    for eye in eyes:
                        cv2.rectangle(
                            frame,
                            (eye[0], eye[1]),
                            (eye[0] + eye[2], eye[1] + eye[3]),
                            (0, 255, 0),
                            2,
                        )

        # Press 'q' to quit the application
        if key == ord("q"):
            break

    # If 'q' was pressed, break the outer loop as well
    if key == ord("q"):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
