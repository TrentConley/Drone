import cv2
import dlib

# Face detector
face_detector = dlib.get_frontal_face_detector()

# Landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat.bz2")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # Detect faces
    faces = face_detector(frame)

    for face in faces:
        # Detect landmarks
        landmarks = predictor(frame, face)

        # Get coordinates of eyes
        leye = landmarks.part(36)
        reye = landmarks.part(45)

        # Draw circles on eyes
        cv2.circle(frame, (leye.x, leye.y), 3, (0, 255, 0), 2)
        cv2.circle(frame, (reye.x, reye.y), 3, (0, 255, 0), 2)

        # Crop eye regions
        left_eye = frame[leye.y - 5 : leye.y + 5, leye.x - 5 : leye.x + 5]
        right_eye = frame[reye.y - 5 : reye.y + 5, reye.x - 5 : reye.x + 5]

    # Display output
    cv2.imshow("Webcam", frame)

    # Stop on q keypress
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
