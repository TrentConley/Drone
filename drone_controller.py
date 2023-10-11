from Tello.tello import *


def execute_movement(position):
    words = position.split()
    if words[0] == "Upper":
        up(20)
    elif words[0] == "Lower":
        down(20)

    if words[1] == "Left":
        anticlockwise(90)
    elif words[1] == "Right":
        clockwise(90)
