import os
from keras.models import load_model
from cv2 import cv2 as cv
from keras import layers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
import numpy as np
from PIL import Image
from keras.preprocessing import image
import base64
import io

model_flower = load_model('./trained_models/best_res_Flowers_1e-7.h5')

def pre(img_path):
    img = image.load_img(img_path,target_size=(300, 300))
    imgsh = img
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    preds = model_flower.predict(img_tensor)
    class_name=['daisy','rose','sunflower','tulip']
    # np.argmax(preds[0])
    print(class_name[np.argmax(preds)])
    ans = class_name[np.argmax(preds)]
    return imgsh, ans

