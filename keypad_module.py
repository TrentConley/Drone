import pygame

def init():
    pygame.init()
    win = pygame.display.set_mode((400,400))

def getKeys(key):
    ans = False

    for event in pygame.event.get(): pass
    key_input = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(key))

    if key_input[myKey]:
        ans = True
    pygame.display.update()
    return ans

def main():
    if getKeys("LEFT"):
        print("you pressed the left key!")
    if getKeys("RIGHT"):
        print("you pressed the right key!")

if __name__ == '__main__':
    init()
    while True:
        main()
