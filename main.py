# import the necessary packages
from imutils import face_utils
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
position = "upper_right"


cap = cv2.VideoCapture(0)

index = 0
while True:
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

        # extract the left and right eye coordinates
        # leftEye = shape[36:42]
        rightEye = shape[42:48]

        # compute the bounding box of the eye and then draw it on the image
        # leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

        # extract the eyes from the image and save them
        # leftEyeRect = cv2.boundingRect(leftEyeHull)
        rightEyeRect = cv2.boundingRect(rightEyeHull)

        # leftEyeImage = image[
        #     leftEyeRect[1] : leftEyeRect[1] + leftEyeRect[3],
        #     leftEyeRect[0] : leftEyeRect[0] + leftEyeRect[2],
        # ]
        rightEyeImage = image[
            rightEyeRect[1] : rightEyeRect[1] + rightEyeRect[3],
            rightEyeRect[0] : rightEyeRect[0] + rightEyeRect[2],
        ]

        # cv2.imwrite("left_eye.png", leftEyeImage)
        cv2.imwrite(f"dataset/right_eye_{index}_{position}.png", rightEyeImage)
        index = index + 1
        cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # for x, y in shape:
        #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
