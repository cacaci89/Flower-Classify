import os

import shutil
from pathlib import Path
import numpy as np
from matplotlib import pyplot
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from PIL import Image
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img

from utils.datasets import letterbox as original_size_to_letterbox
from keras.preprocessing import image
#================================================================================================
# 獲取設備
device=torch.device('cpu')
weights = './weights/best1229.pt'
model_yolo = attempt_load(weights, map_location=device)
# 獲取類別名字
names = model_yolo.names

# 設置畫框的顏色
rnd = np.random.RandomState(123)
colors = [[rnd.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
#================================================================================================
model_flower = load_model('./trained_models/best_res_Flowers_1e-7.h5')
class_name=['daisy','rose','sunflower','tulip']
#================================================================================================

def yolo_detect(img_origin, file_path, img_size = 416):
    img = original_size_to_letterbox(img_origin, new_shape=img_size)[0]

    # Reshape  格式轉成通道在前
    img = img.transpose(2, 0, 1)  # convert from 416x416x3 to 3x416x416
    img = np.ascontiguousarray(img) # 讓array資料連續位置存放 運算比較快

    img = torch.from_numpy(img) # to torch tensor
    img = img.float()/255.0 # uint8 to fp16/32 --> 0-1
    img = img.unsqueeze(0) #reshape成為4個維度 (1,3,height,width) (神經網路的輸入)

#### 偵測可能物件
    pred = model_yolo(img)[0]
    #print(pred[..., 5:-1])

##### 過濾最可能的物件，依據信心值與iou閾值
# Apply NMS 進行NMS

    det = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0] # 只有一張圖 因此選取編號0即可
#======================================================================================================================================
# 若沒有物件被偵測到，返回
    if det is None:
        print("沒有偵測到物件")
        return None, None # 後面的步驟不要做了

    #####
    ##### 對每一個偵測到的物件做處理Process detections
    # 將det的xyxy尺寸調整為original size圖片的座標
    # Rescale boxes from img_size to original size
    # 調整預測框的座標，將img_size(經過resize+pad)的圖片的座標-->轉為original size圖片的座標
    # det_scaled內每個物件的資訊為[x1,y1,x2,y2,conf,cls]    
    # det_scaled --> tensor([[108.00000, 399.00000, 148.00000, 463.00000,   0.82306,   0.00000],
    #                        [263.00000, 393.00000, 307.00000, 472.00000,   0.79723,   0.00000]])
    det_scaled = det.clone().detach()
    det_scaled[:, :4] = scale_coords(img.shape[2:], det_scaled[:, :4], img_origin.shape).round()
    ####
    #### 針對每個偵測物件畫上外框
    # Process detections對每一個偵測到的物件做處理
    # 原始圖片img_origin的複製，用來畫偵測到物件外框
    #img_result = img_origin.copy()
    img_result = img_origin.copy()
    img_height, img_width = img_result.shape[0:2]

    ## 有多個物件被偵測到，一一處理之
    photo_obj_info=[]  #存放物件資訊的list
    for obj_id, (*xyxy, conf, class_num) in enumerate(det_scaled):
        
        # 物件名稱外框等資訊
        x1,y1,x2,y2 = [int(val.tolist()) for val in xyxy]
        #box_origin = [x1,y1,x2,y2] # 未加大前的尺寸
        #print(box_origin)              

        
        # 稍微把切臉的框框加大些:目的是將頭髮等性別的特徵納入，提高判別準確度
        hy = y2 - y1
        wx = x2 - x1
        ws = 0.2 #自訂:寬度加大比率
        hs = 0.2 #自訂:高度加大比率
        x1, y1, x2, y2 = max(0,int(x1-wx*ws)),max(0,int(y1-hy*hs)),  min(img_width,int(x2+wx*ws)), min(img_height,int(y2+hy*hs)) # 需考慮加大後超出圖片的情況

        box = (x1, y1, x2, y2) # 加大後的尺寸
        
        # 進行物件圖片切割crop object
        obj_img = img_origin[y1:y2, x1:x2]
        #print(obj_img.shape)
        # 物件圖片存檔
        # save_img('output/obj_id_{}.jpg'.format(obj_id), obj_img)
        
        
        # 物件資訊
        obj_info = {} #輸出dictionary
        obj_name = names[ int(class_num) ] # 物件名稱
        obj_info['obj_id'] = obj_id
        obj_info['obj_name'] = obj_name
        obj_info['confidence'] = round(float(conf),2)
        obj_info['box'] = box
        #================================================================================================
        # img = cv2.resize(obj_img, (300,300))
        # img = np.expand_dims(img, axis=0)
        # #img = preprocess_input(img)

        # flower_preds = model_flower.predict(img)
        # print(np.argmax(flower_preds[0]))
        # flower_num = np.argmax( flower_preds[0] )
        # flower_proba = float(flower_preds[0][flower_num]) #轉成python float
        # flower_proba = int(flower_proba*100) 

        # # np.argmax(preds[0])
        # flower_label = class_name[flower_num]
        # print(flower_label)

        img = image.load_img(file_path, target_size=(300, 300))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        preds = model_flower.predict(img_tensor)
        flower_num = np.argmax( preds[0] )
        flower_proba = float(preds[0][flower_num])
        flower_proba = int(flower_proba*100)
        print(class_name[np.argmax(preds)])
        flower_label = class_name[np.argmax(preds)]

        #加上年齡與性別資訊
        obj_info["flower"] = {"flower": flower_label, "proba": flower_proba}
        #========================================================    
        
        # print(obj_info)
        # photo object info加入每個物件資訊
        photo_obj_info.append(obj_info)
        

        # 在img_result(原圖大小)上畫物件外框
        # box_label = "{},{}{:.2f}".format(obj_id, obj_name, conf) # 0,bus0.89 # 只有物件資訊
        box_label = "{},{}{}%".format(obj_id, flower_label, flower_proba) #加上年齡與性別文字資訊
        # draw box on img_result
        plot_one_box(xyxy, img_result, label=box_label, color=colors[int(class_num)], line_thickness=3)
        #----------------
        # end of for each object
        

    #### Save resulted image (image with detections)
    # save_img("./media/img_result.jpg", img_result)
    if img_result.shape[0] > 800 :
        img_result = original_size_to_letterbox(img_result, new_shape=800)[0]
    #print(img_result.shape)
    return img_result, photo_obj_info
