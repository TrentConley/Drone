from djitellopy import tello
from time import sleep

me = tello.Tello()
me.connect()
print(me.get_battery())


def execute_movement(position):
    # Split the position into its components
    position_parts = position.split()

    # Determine the vertical movement
    if "Upper" in position_parts:
        move_up()
    elif "Lower" in position_parts:
        move_down()

    # Determine the horizontal movement
    if "Left" in position_parts:
        rotate_counter_clockwise()
    elif "Right" in position_parts:
        rotate_clockwise()


def takeoff():
    me.takeoff()


def land():
    me.land()


def move_up(distance=10):
    me.move_up(distance)


def move_down(distance=10):
    me.move_down(distance)


def move_left(distance=10):
    me.move_left(distance)


def move_right(distance=10):
    me.move_right(distance)


def rotate_clockwise(angle=10):
    me.rotate_clockwise(angle)


def rotate_counter_clockwise(angle=10):
    me.rotate_counter_clockwise(angle)
