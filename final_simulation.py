
# Imports
import car_racing_ainn as cr
import pygame
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from data_processing import process_img, SCALE_PERCENT, PROCESS_ALL

# pip install gymnasium
# pip install gymnasium[box2d]

# Global variables
SIMULATE = True
WEIGHTING_FACTORS = np.array([1.5, 2, 5, 5, 0.75])
ACTIONS_TO_CONTROL_INPUT = {

    0 : [ 0,  0,  0,    'Do Nothing'    ],
    1 : [ 0,  4.5,  0,  'Accelerate'    ],
    2 : [-1,  0,  0,    'Steer Left'    ],
    3 : [ 1,  0,  0,    'Steer Right'   ],
    4 : [ 0,  0,  0.01, 'Brake'         ],
    }


# Load model
env = cr.CarRacing(render_mode="human")
model = keras.models.load_model('model1')
quit = False
restart = False
a = np.array([0.0, 0.0, 0.0])

def determine_action():
    
    pygame.image.save(env.screen, 'temp.jpg')
    img = process_img('temp.jpg',SCALE_PERCENT, show=False,include_action=False)
    img = np.expand_dims(img,0)
    
    predict = model.predict((img))
    predict = np.multiply(predict, WEIGHTING_FACTORS)
    action = np.argmax(predict)
    print(f'Predicted action: {action} : {ACTIONS_TO_CONTROL_INPUT[action][3]}')
    return action

def simulate():
    global quit, restart, a

    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False

        while True:
            
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        quit = True
                        print('Escape key is pressed')
        
            # Generate action
            action = determine_action()
            if action == 0:
                a = a = np.array([0.0, 0.0, 0.0])
            else:
                a += ACTIONS_TO_CONTROL_INPUT[action][:3]

            # Go to next step
            _, r, terminated, truncated, _ = env.step(a)
            total_reward += r
            
            steps += 1
            if terminated or truncated or restart or quit:
                break
            
        env.close()

if SIMULATE:
    simulate()

