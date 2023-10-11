from Tello.tello import *
import threading
import queue

# Create a queue to hold the movements
movement_queue = queue.Queue()


def execute_movement(position):
    # Add the movement to the queue
    movement_queue.put(position)


def perform_movement():
    while True:
        # Get the next movement from the queue
        movement = movement_queue.get()
        print(f"Processing movement: {movement}")
        words = movement.split()
        if words[0] == "Upper":
            up(20)
        elif words[0] == "Lower":
            down(20)

        if words[1] == "Left":
            anticlockwise(90)
        elif words[1] == "Right":
            clockwise(90)
        # Mark the task as done
        movement_queue.task_done()


# Start a thread to process movements
movement_thread = threading.Thread(target=perform_movement)
movement_thread.start()
