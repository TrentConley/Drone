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


# Function to handle mouse events
def draw_rectangle(event, x, y, flags, param):
    global labels, drawing, start_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing rectangle
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the end point of the rectangle while moving the mouse
            end_point = (x, y)
            labels[-1] = (
                start_point[0],
                start_point[1],
                end_point[0] - start_point[0],
                end_point[1] - start_point[1],
            )

    elif event == cv2.EVENT_LBUTTONUP:
        # Stop drawing rectangle
        drawing = False
        end_point = (x, y)
        labels[-1] = (
            start_point[0],
            start_point[1],
            end_point[0] - start_point[0],
            end_point[1] - start_point[1],
        )


# Create a window and attach the mouse callback function
cv2.namedWindow("Face and Eye Detection")
cv2.setMouseCallback("Face and Eye Detection", draw_rectangle)

frame_count = 0
drawing = False
start_point = (0, 0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Create a copy of the frame before drawing rectangles
    original_frame = frame.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    labels = []
    for x, y, w, h in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get the region of interest to find eyes
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for ex, ey, ew, eh in eyes:
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            labels.append((int(ex), int(ey), int(ew), int(eh)))

    # Display the frame with face and eye rectangles
    cv2.imshow("Face and Eye Detection", frame)

    # Wait for user to press 'Enter' to save the frame and labels, or 'd' to delete the last rectangle
    key = cv2.waitKey(0)
    if key == 13:  # Enter key
        cv2.imwrite(f"dataset/frame_{frame_count}.png", original_frame)
        with open(f"dataset/labels_{frame_count}.json", "w") as f:
            json.dump(labels, f)
        frame_count += 1
    elif key == ord("d"):
        if labels:
            labels.pop()

    # Press 'q' to quit the application
    if key == ord("q"):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
