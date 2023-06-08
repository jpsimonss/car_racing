# Install with pip:
# pip install --upgrade pip
# pip install opencv-python
# pip install -U matplotlib

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import datetime as dt

SCALE_PERCENT = 10
PROCESS_ALL = False
processed_image_shape = []

def process_img(img_name: str, scale_percent, show=False, include_action=False):
    global processed_image_shape
    """ 
    Input: Image filename (Image in RBG)
    Output: numpy-array of picture (Image in greyscale array (HxW)) + action
    NOTE: Make sure img_name = './data/trial12_step16.jpg'
    """
    
    # Load and make gray
    image = cv2.imread(img_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    img_array = np.matrix(resized_gray)
    processed_image_shape = img_array.shape

    # Perhaps do something intersting here with 
    # contrast or brightness (?) or PCA (sklearn)
    # Add some random noise, also nicel

    if include_action:
    # Include action -> bit long, but okay
        al_idx = img_name.find('al')
        dash_idx = img_name.find('_s')
        trial = img_name[al_idx+2:dash_idx]
        ep_idx = img_name.find('ep')
        jpg_idx = img_name.find('.j')
        step = int(img_name[ep_idx+2:jpg_idx])

        actions = np.load('./data/actions_trial{}.npy'.format(trial))
        action = actions[step]
    
    if show:
        plt.imshow(resized_gray, cmap='gray')
        plt.title('Trial {}, step {}'.format(trial, step))
        plt.show()

    if include_action:
        return img_array , action
    else:
        return img_array


def process_all():
    images, actions = np.array([]), []
    counter = 0
    for filename in os.listdir('data'):
        file = os.path.join('data', filename)
        if os.path.isfile(file) and file.endswith('.jpg'):
            
            print(file)
            processed_img, action = process_img(file, scale_percent=SCALE_PERCENT, include_action=True)
            images = np.append(images,processed_img.flatten())
            actions.append(action)
            counter += 1
    images = np.reshape(images,(counter,processed_image_shape[0], processed_image_shape[1]))
    print(f'\nTotal: {counter} images processed')

    print(f'{images.shape = }, {len(actions) = }\n')

    # Save both items
    now = dt.datetime.today().strftime('%b-%d_%H-%M-%S')
    print(f'Date and time: {now}')
    processed_img_name = './data_processed/img_{}'.format(now)
    processed_action_name = './data_processed/action_{}'.format(now)
    np.save(processed_img_name, images)
    np.save(processed_action_name, actions)

    # Show random image to check
    random_idx = np.random.randint(images.shape[0])
    plt.imshow(images[random_idx,:,:], cmap = 'gray')
    plt.title(f'Action: {actions[random_idx]}')
    plt.show()

if PROCESS_ALL:
    process_all()