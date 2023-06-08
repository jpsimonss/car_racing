# Installs:
# pip install gymnasium
# pip install gymnasium[box2d]

# Relative path to car_racing.py
# ai_project_env\Lib\site-packages\gymnasium\envs\box2d\car_racing.py
# play: python ai_project_env\Lib\site-packages\gymnasium\envs\box2d\car_racing.py

import car_racing_ainn as cr
import pygame
import numpy as np


trial = 1
RECORDING = True
FIRST_REMOVE_SHOTS = 30


# Create game
env = cr.CarRacing(render_mode="human")

a = np.array([0.0, 0.0, 0.0])
def register_input():
    global quit, restart
    action = 0

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            # print(f'Key pressed')
            if event.key == pygame.K_LEFT:
                action = 2
                a[0] = -1.0
            if event.key == pygame.K_RIGHT:
                action = 3
                a[0] = +1.0
            if event.key == pygame.K_UP:
                action = 1
                a[1] = +0.6
            if event.key == pygame.K_DOWN:
                action = 4
                a[2] = +0.3  # set 1.0 for wheels to block to zero rotation
            if event.key == pygame.K_RETURN:
                print('Return key is pressed')
                restart = True
            if event.key == pygame.K_ESCAPE:
                quit = True
                print('Escape key is pressed')

        if event.type == pygame.KEYUP:
            # print(f'Key released')
            action = -1
            if event.key == pygame.K_LEFT:
                a[0] = 0
            if event.key == pygame.K_RIGHT:
                a[0] = 0
            if event.key == pygame.K_UP:
                a[1] = 0
            if event.key == pygame.K_DOWN:
                a[2] = 0

        if event.type == pygame.QUIT:
            quit = True
    
    return action

# Start game
restart = False
quit = False

def record():
    actions = []

    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        previous_action = 0

        while True:
            
            # Record actions
            action = register_input()
            
            if action > 0: # Key pressed
                previous_action = action
            elif action == 0 and previous_action != 0: # Still holding key
                action = previous_action
            elif action == -1: #Key released
                action = 0
            previous_action = action
            actions.append(action)

            # Save window (from step 30)
            if steps >= FIRST_REMOVE_SHOTS:
                if RECORDING:
                    image_name = './data/trial{}_step{}.jpg'.format(trial,steps - FIRST_REMOVE_SHOTS)
                    pygame.image.save(env.screen, image_name)

            # Next pygame step
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            
            steps += 1
            if terminated or truncated or restart or quit:
                break
            
        env.close()
    
    # Get rid of first 30 steps
    actions = actions[FIRST_REMOVE_SHOTS:]

    return actions

actions = record()

if RECORDING:
    np.save('./data/actions_trial{}'.format(trial), actions)
    print(f'Length actions list = {len(actions)}')