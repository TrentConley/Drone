from djitellopy import tello
import keypad_module as km
from time import sleep
import cv2
from main import run

km.init()

drone = tello.Tello()
drone.connect()
print(drone.get_battery())


def getImput():
    # drone.takeoff()
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 80

    if km.getKeys("a"):
        lr = -speed
        print("LEFT KEY PRESSED...")
    elif km.getKeys("d"):
        lr = speed
        print("RIGHT KEY PRESSED...")

    if km.getKeys("w"):
        fb = speed
        print("UP KEY PRESSED...")
    elif km.getKeys("s"):
        fb = -speed
        print("DOWN KEY PRESSED...")

    if km.getKeys("UP"):
        ud = speed
        print("W KEY PRESSED...")
    elif km.getKeys("DOWN"):
        ud = -speed
        print("S KEY PRESSED...")

    if km.getKeys("LEFT"):
        yv = speed
        print("A KEY PRESSED...")
    elif km.getKeys("RIGHT"):
        yv = -speed
        print("D KEY PRESSED...")
    if km.getKeys("q"):
        drone.land()
        print("Q KEY PRESSED...")
    elif km.getKeys("r"):
        drone.takeoff()
        print("R KEY PRESSED...")

    return [lr, fb, ud, yv]


drone.streamon()

while True:
    val = getImput()
    result = run()
    if result is not None:
        left, right = result
    else:
        left, right = 200, 200  # or any other default values
    if left < 120:
        val[3] = 80
    elif right < 120:
        val[3] = -80
    drone.send_rc_control(val[0], val[1], val[2], val[3])
    sleep(0.05)
    frame = drone.get_frame_read().frame
    cv2.imshow("Frame", frame)

    # code for reading eye position
