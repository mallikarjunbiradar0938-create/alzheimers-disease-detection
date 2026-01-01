import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('alzheimers_model.h5')

img = cv2.imread('AugmentedAlzheimerDataset/VeryMildDemented/0a1d2c6b-8a59-4e07-879f-fd4f4b76db34.jpg')  # path to any brain MRI image
img = cv2.resize(img, (128,128))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
class_names = ['MildDemented', 'NonDemented', 'VeryMildDemented', 'ModerateDemented']
print("Predicted:", class_names[np.argmax(pred)], "with confidence:", np.max(pred)*100)
