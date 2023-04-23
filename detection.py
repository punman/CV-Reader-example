import copy
import numpy as np
import pandas as pd

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
import pathlib


class Detect:
    def __init__(self, model, device='cpu', acc_threshold=0.6):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.WEIGHTS = os.path.join(str(pathlib.Path(__file__).parent.absolute()) + "/model/", model)  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = acc_threshold   # set a custom testing threshold
        self.cfg.MODEL.DEVICE= device
        self.predictor = DefaultPredictor(self.cfg)
        
    def load_model(self, model):
        self.cfg.MODEL.WEIGHTS = os.path.join("model/", model)
        self.predictor = DefaultPredictor(self.cfg)
        
    def tensorToList(self, tensorArray):
        boxes = []
        for i in range(len(tensorArray['instances'])):
            boxes.append(
                [
                int(tensorArray['instances'][i].pred_boxes.tensor[0][0]),
                int(tensorArray['instances'][i].pred_boxes.tensor[0][1]),
                int(tensorArray['instances'][i].pred_boxes.tensor[0][2]),
                int(tensorArray['instances'][i].pred_boxes.tensor[0][3])
                ]
            )
            
        return boxes
        
    def predict(self,img):
        try:
            print("Text Detection....")
            outputs = self.predictor(img)
        except Exception as e:
            print("Detecttion error!!!")
            raise e
        return outputs
    
    def visualize_detection(self, img, model_output):
        """
        img - cv2 object
        model_output - prediction object from model
        """
        im = cv2.imread(img)
        v = Visualizer(im[:, :, ::-1],
                metadata=MetadataCatalog.get("cv_train"), 
                scale=0.5, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
        out = v.draw_instance_predictions(model_output["instances"].to("cpu"))
        cv2.imwrite("output/output.jpg", out.get_image()[:, :, ::-1])
        
    

class util:
    
    def bb_intersection_over_union(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def union(self, a, b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[2], b[2])
        h = max(a[3], b[3])
        return (x, y, w, h)

    @staticmethod
    def expand( boxes, min, max, axis='y'):
        expanded = []
        for box in boxes:
            if axis == 'y':
                box[1] = min
                box[3] = max
            elif axis == 'x':
                box[0] = min
                box[2] = max
            else:
                print("axis could only be 'x' or 'y'")
                return None
            expanded.append(box)
        return expanded
    
    
    @staticmethod
    def tensorToList(tensor_output):
        boxes = []
        for i in range(len(tensor_output['instances'])):
            boxes.append(
                [
                float(tensor_output['instances'][i].pred_boxes.tensor[0][0]),
                float(tensor_output['instances'][i].pred_boxes.tensor[0][1]),
                float(tensor_output['instances'][i].pred_boxes.tensor[0][2]),
                float(tensor_output['instances'][i].pred_boxes.tensor[0][3])
                ]
            )
        return boxes
    
    
    @staticmethod
    def center_df(list_boxes):
        centers_x = [] 
        centers_y = [] 

        for i in list_boxes:
            centers_x.append((i[0] + i[2])/2)
            centers_y.append((i[1] + i[3])/2)
            
        df = pd.DataFrame({"centerX": centers_x,"centerY":centers_y})
        return df
    
    @staticmethod
    def center_arr(list_boxes):
        listBox = []
        for i in list_boxes:
            listBox.append([(i[0] + i[2])/2,(i[1] + i[3])/2])
            
        return np.array(listBox)
        