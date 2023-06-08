# Imports
# pip install -U scikit-learn
# No GPU: pip install tensorflow-cpu # you have a GPU:# pip install tensorflow

import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as layers
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Fix that it uses my GPU

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


# Encoding

def encoding(y_orig, one_hot=True, binarizer=False):
  if one_hot:
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(y_orig.reshape(-1,1))
    return transformed
    
  if binarizer:
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(y_orig)
    return lb.transform(y_orig)

y_train_encoded = encoding(y_train)
y_test_encoded = encoding(y_test)

# NEURAL NETWORK
model1 = True
if model1:
  model1 = tf.keras.models.Sequential(
    [layers.Conv2D(24, kernel_size=5, activation='relu', input_shape=(X_train.shape[1:])),
    layers.Dropout(0.25),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Conv2D(48, kernel_size=5, activation='relu'),
    layers.Dropout(0.25),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(5, activation='softmax')
    ])

  model1.compile(optimizer='Adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model1.fit(X_train, y_train, epochs=25)
  # model1 = tf.keras.models.load_model('model1')
  print('\nEvaluation on test_batch:\n')
  model1.evaluate(X_test, y_test)
  print('\n')
  model1.save('model1')


# MODEL 2
model2 = False
if model2:
  model2 = models.Sequential()
  model2.add(layers.Flatten())
  model2.add(layers.Dense(image_height * image_width // 2))
  model2.add(layers.Dropout(0.25))
  model2.add(layers.Dense(image_height * image_width // 4))
  model2.add(layers.Dropout(0.25))
  model2.add(layers.Dense(image_height * image_width // 8))
  model2.add(layers.Dropout(0.25))
  model2.add(layers.Dense(5, activation='softmax'))

  model2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  model2.fit(X_train, y_train, epochs=5)
  model2.evaluate(X_test,y_test, verbose=2)
  model2.save('model2')


# # TEST NETWORK
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score, recall_score, f1_score
# from tensorflow import keras


