import pytesseract
from group import meanshift
from detection import util, Detect
# from layout import layout_classify
from copy import deepcopy
import numpy as np
import os, json, cv2, random
from pprint import pprint
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
from matplotlib import cm
import img_tool

import streamlit as st
from functools import lru_cache

# dt = Detect(model = "model_final.pth")
u = util()


def get_contour(img):
    contours_list = []
    BG_COLOR = img_tool.BackgroundColorDetector(img).detect()
    if BG_COLOR == (255,255,255):
        img = cv2.bitwise_not(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    dilated = cv2.dilate(img.copy(), kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print("contour found", len(contours))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contours_list.append((x, y, w, h))
        
    return contours_list

def crop_img(img, bbox, padding = 2):
    
    try:
        x1 = int(bbox[0]) - padding
        y1 = int(bbox[1]) - padding
        x2 = int(bbox[2]) + padding
        y2 = int(bbox[3]) + padding
        img = img[:, :, ::-1][y1:y2,x1:x2]
        
    except:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        img = img[:, :, ::-1][y1:y2,x1:x2]
        
    return img

def preprocess_img(img):
    imgr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgr = cv2.GaussianBlur(imgr, (3,3), 1)
    _, imgr = cv2.threshold(imgr, 0, 255, cv2.THRESH_OTSU)
    BG_COLOR = img_tool.BackgroundColorDetector(imgr).detect()
    if BG_COLOR == (0,0,0):
        imgr = cv2.bitwise_not(imgr)
    return imgr

def img_to_string(img):
    custom_config = r'--oem 1 -c preserve_interword_spaces=1 --psm 3'
    tessdata_dir_config = r'--tessdata-dir "/home/pun"'
    width = int(img.shape[1])
    height = int(img.shape[0])
    return pytesseract.image_to_string(img, lang="eng+tha",config=custom_config)

@lru_cache
def extract(image, acc_threshold=0.5, bbox_combine_ratio=0.01, column_combine_ratio=0.001):
    im = cv2.imread(image)
    
    start = datetime.now()

    from layout_MPRO import predict, layout_classify
    layout = predict(image)
    print("Layout",datetime.now()-start)
    
    # Detect text bbox
    dt = Detect(model = "model_final.pth",acc_threshold = acc_threshold)
    detected = dt.predict(im)
    
    #Save bboxes images to debug
    dt.visualize_detection(image, detected)
    
    detected = dt.tensorToList(detected)
    print("detected",datetime.now()-start)
    if len(detected) < 1:
        return None, None
    
    
    print("=========== Detection Info ===========")
    
    # Merge overlaping boxes
    boxes = sorted(detected, key= lambda x : x[1])
    print("BOXES :", len(boxes))
    boxes_combined = u.combine_boxes(boxes,overlap_ratio=bbox_combine_ratio)
    
    # Calculate center of bboxes
    center_arr = u.center_arr(boxes_combined)
    center_group = meanshift(center_arr)
    
    # Get header Boxes
    boxes_combined = np.array(boxes_combined)
    boxes_combined_copy = deepcopy(boxes_combined)
    print("BOXES COMBINED :", len(boxes_combined))
    header_boxes = boxes_combined[center_group == center_group[0]]
    header_boxes = u.combine_boxes(header_boxes,overlap_ratio=0.8)
    
    # Finding columns
    if layout == 1:
        # multi columns
        boxes_no_header = boxes_combined_copy[center_group != center_group[0]]
        boxes_no_header = u.combine_boxes(boxes_no_header,overlap_ratio=0.8)
        expanded = u.expand(boxes_no_header,0,im.shape[0], axis='y')
    else:
        # single column
        expanded = u.expand(boxes_combined_copy,0,im.shape[0], axis='y')

    columns = u.combine_boxes(expanded, overlap_ratio=column_combine_ratio)
    columns = sorted(columns,key=lambda x : x[0])
    print("COLUMNS :",len(columns))
    
    
    print("======================================")
    
    # Export Data
    data = {}
    for idx, bc in enumerate(boxes_combined):
        data['box'+ str(idx)] = {}
        data['box'+ str(idx)]['bbox'] = bc
        data['box'+ str(idx)]['column'] = []
        data['box'+ str(idx)]['group'] = 'detail'
        
        for hd in header_boxes:
            if u.bb_intersection_over_union(bc,hd) > 0.8:
                data['box'+ str(idx)]['group'] = 'header'
                break            
        
        for i,col in enumerate(columns):
            if u.bb_intersection_over_union(bc, col) > 0.001:
                data['box'+ str(idx)]['column'].append(i)
                
        cropped = crop_img(im.copy(), bc)
        if cropped.size == 0:
            print("Empty crop")
            continue
        preprocessed = preprocess_img(cropped)
        data['box'+ str(idx)]['text'] = img_to_string(preprocessed)
    
        
    return data, len(columns)

def get_text(data_dict, num_of_columns):
    data_sort_by_column = []
    for i in range(num_of_columns):
        b = []
        for _ , value in data_dict.items():
            if i in value['column']:
                try:
                    b.append((value['bbox'],value['text']))
                except:
                    continue
        
        b = sorted(b,key=lambda x: x[0][1])
        data_sort_by_column.append([i[1].strip() for i in b if i is not None])
    return data_sort_by_column

def raw_text(data_sort_by_column):
    text = ""
    for idx, item in enumerate(data_sort_by_column):
        text = text + f"column{idx}" + "\n" + "-----" + "\n"
        for i in item:
            text = text + i + "\n"
            text = text + "-----" + "\n"
            
    return text
   
def display(data_sort_by_column):
    for idx, item in enumerate(data_sort_by_column):
        print(f"======== Column {idx} ==========")
        for i in item:
            print(i)
            print("..........................")
 
    
def main(img_path, acc_threshold = 0.5, bbox_combine_ratio=0.01, column_combine_ratio=0.001):
    start = datetime.now()
    extracted, num_of_columns = extract(img_path, acc_threshold, bbox_combine_ratio, column_combine_ratio)
    print("extracted",datetime.now()-start)
    # pprint(extracted)
    if extracted is None:
        return ''
    data_sorted = get_text(extracted, num_of_columns)
    print("started",datetime.now()-start)
    # pprint(data_sorted)
    text = raw_text(data_sorted)
    print("raw text",datetime.now()-start)
    # display(data_sorted)
    print("Elapsed Time :",datetime.now() - start)
    return text