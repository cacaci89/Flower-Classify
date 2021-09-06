from django.shortcuts import render

import requests
from PIL import Image

import matplotlib.pyplot as plt
import io
from io import BytesIO
import cv2
import numpy as np
import base64
import os
import platform
import shutil
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from pathlib import Path
import urllib.request
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import tensorflow.keras
from keras import models
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.datasets import letterbox as original_size_to_letterbox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img




#api_url = "http://163.18.23.119:8000/api/detect/"
api_url = "http://127.0.0.1:8000/api/api_classify_image/"
#=====================================================================================

import re
from django.http import JsonResponse
from PIL import ExifTags
#model_flower = load_model('./trained_models/best_res_Flowers_1e-7.h5')
from predict import pre
from preyolo import yolo_detect
#======================================================================================
#@csrf_exempt
def predict(request):

    if request.method == 'POST':
        # 讀取前端網頁送過來的影像檔案InMemoryUploadedFile
        img_bin = request.FILES["upload_image"]

        fs = FileSystemStorage()
        file_path = fs.save('img_uploaded.jpg', img_bin)
        img_pil = Image.open(img_bin)
        img_origin = np.array(img_pil)

        img_result, obj_info = yolo_detect(img_origin, file_path)

        os.remove(file_path)

        if obj_info == None:
            
            response ={
                'img_result_b64': [],
                'obj_info': "沒有偵測到物件!"
            }
            return render(request, 'app_detect/home.html', response)

        
        img_result = array_to_img( img_result ) # 圖片轉回PIL格式
        output_buffer = BytesIO()  # 產生一個二進位檔案的buffer
        img_result.save(output_buffer, format='PNG')  # 將img影像存到該二進位檔案的buffer
        byte_data = output_buffer.getvalue()  # 拿出該buffer的二進位格式資料
        img_base64 = base64.b64encode(byte_data).decode()  # 編碼成base64再decode()轉碼成文字格式
    

        # 注意: img.tobytes()得到的影像編碼不同，在網頁端無法顯示
        # img_base64 = base64.b64encode(img_result.tobytes()).decode()

        # 若需要讀本地端的圖片時的寫法
        # with open('static/images/dog.jpg', "rb") as image_file:
        #    img_base64 = base64.b64encode(image_file.read()).decode()
    
        # objs_info本身是dict，此處增加一個'img_result'鍵值，以夾帶編碼後的影像到前端網頁
        #response['img_result'] = img_base64
        response ={
            'img_result_b64': img_base64,
            'obj_info': obj_info
        }


        return render(request, 'app_detect/detect.html', response)

    return render(request,"app_detect/detect.html")
#======================================================================================


#======================================================================================


def home(request):

    if request.method == 'POST':
        # 上傳過來的檔案存放於記憶體中
        # <class 'django.core.files.uploadedfile.InMemoryUploadedFile'>
        # 讀取前端網頁送過來的影像檔案InMemoryUploadedFile
        img_bin = request.FILES["upload_image"]
        
        # 酬載 (payload)
        payload = {"upload_image": img_bin}


        result = requests.post(api_url, files=payload)
        result = result.json()

        obj_info = result['obj_info']
        print(obj_info)

        if obj_info == None:
            
            response ={
                'img_result_b64': [],
                'obj_info': "沒有偵測到物件!"
            }
            return render(request, 'app_detect/home.html', response)

        img_result_b64 = result['img_result']

        response ={
            'img_result_b64': img_result_b64,
            'obj_info': obj_info
        }
        return render(request, 'app_detect/home.html', response)

    return render(request,"app_detect/home.html")

def home_v0(request):

    if request.method == 'POST':
        # 上傳過來的檔案存放於記憶體中
        # <class 'django.core.files.uploadedfile.InMemoryUploadedFile'>
        # 讀取前端網頁送過來的影像檔案InMemoryUploadedFile
        img_bin = request.FILES["upload_image"]
        
        # 酬載 (payload)
        payload = {"upload_image": img_bin}


        result = requests.post(api_url, files=payload)
        result = result.json()

        obj_info = result['obj_info']
        print(obj_info)

        if obj_info == None:
            
            response ={
                'img_result_b64': [],
                'obj_info': "沒有偵測到物件!"
            }
            return render(request, 'app_detect/home_v0.html', response)

        img_result_b64 = result['img_result']

        response ={
            'img_result_b64': img_result_b64,
            'obj_info': obj_info
        }
        return render(request, 'app_detect/home_v0.html', response)

    return render(request,"app_detect/home_v0.html")


# Create your views here.
