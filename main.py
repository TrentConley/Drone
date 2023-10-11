# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import torch
from torchvision import transforms
from PIL import Image


from trainer import EyeNet  # Import the EyeNet class

# Initialize the model
model = EyeNet()

# Load the state dictionary
model.load_state_dict(torch.load("eye_position_model.pth"))

# Now you can call eval on the model
model.eval()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
mapping = {
    1: "Upper Left",
    2: "Upper Middle",
    3: "Upper Right",
    4: "Middle Left",
    5: "Middle Middle",
    6: "Middle Right",
    7: "Lower Left",
    8: "Lower Middle",
    9: "Lower Right",
}

cap = cv2.VideoCapture(0)
index = -1

from drone_controller import *
import time
import threading
from Tello.tello import *

start()
print("started")
start_video()
takeoff()
print("taken off")


def keep_alive():
    while True:
        # Send a no-op command to keep the drone alive
        send_and_wait("command")
        time.sleep(5)  # Send every 5 seconds


# Start the keep_alive thread
keep_alive_thread = threading.Thread(target=keep_alive)
keep_alive_thread.start()
i = 0

while True:
    index = index + 1
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for i, rect in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the right eye coordinates
        rightEye = shape[42:48]

        # compute the bounding box of the eye and then draw it on the image
        rightEyeHull = cv2.convexHull(rightEye)

        # extract the eyes from the image and save them
        rightEyeRect = cv2.boundingRect(rightEyeHull)

        rightEyeImage = image[
            rightEyeRect[1] : rightEyeRect[1] + rightEyeRect[3],
            rightEyeRect[0] : rightEyeRect[0] + rightEyeRect[2],
        ]

        # Preprocess the eye image and prepare it for prediction
        rightEyeImage = Image.fromarray(cv2.cvtColor(rightEyeImage, cv2.COLOR_BGR2RGB))
        # Convert to grayscale
        rightEyeImage = rightEyeImage.convert("L")
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),  # Resize all images to 64x64
                transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
            ]
        )
        rightEyeImage = transform(rightEyeImage).unsqueeze(0)

        # Predict the eye position
        with torch.no_grad():
            output = model(rightEyeImage)
            _, predicted = torch.max(output.data, 1)

        # Print the predicted eye position
        position = mapping[predicted.item()]
        print(f"The predicted eye position is: {position}")
        execute_movement(position)

        cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for x, y in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
stop_video()
cv2.destroyAllWindows()
cap.release()
