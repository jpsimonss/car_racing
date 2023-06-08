import tensorflow as tf
import numpy as np

  #Import processed data
images = np.load('./data_processed/img_Nov-29_16-44-53.npy')        # X
images = images.astype('float32') / 255.
images = images[..., tf.newaxis]
actions = np.load('./data_processed/action_Nov-29_16-44-53.npy')    # y
image_height, image_width = images.shape[1] , images.shape[2]
print(f'\nData loaded:\n{images.shape = } & {actions.shape = }\n')

#Spit data -> train, test & valid data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, actions, test_size=0.5)

model1 = tf.keras.models.load_model('model1')
print('\nEvaluation on test_batch:\n')
model1.evaluate(X_test, y_test)
print('\n')
model1.save('model1')