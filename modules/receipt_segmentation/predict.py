import torch
import torchvision
import cv2
import os
import numpy as np
import json
import random
from detectron2.structures import BoxMode
from detectron2.data import Metadata
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer


class RSEGMENT():

    def __init__(self, config):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = './weights/receipt_segment.pth'
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.threshold
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
        self.cfg.MODEL.DEVICE = config.device
        self.cfg.DATALOADER.NUM_WORKERS = config.num_workers
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = Metadata()
        self.metadata.set(thing_classes = ['receipt'])


    def predict(self, img, short_size: int = 736):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = short_size / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        predictions = self.predictor(img)
        # v = Visualizer(img[:, :, ::-1],
        #         metadata=self.metadata, 
        #         scale=0.8, 
        #         instance_mode=ColorMode.IMAGE_BW # removes the colors of unsegmented pixels
        #         )
        # output = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        # cv2.imshow("Result", cv2.cvtColor(output.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        instances_pred_boxes = predictions["instances"].to("cpu").pred_boxes.tensor.numpy()
        X1 = (instances_pred_boxes[0][0])
        Y1 = (instances_pred_boxes[0][1])
        X2 = (instances_pred_boxes[0][2])
        Y2 = (instances_pred_boxes[0][3])
        return (img, X1, Y1, X2, Y2)

if __name__ == '__main__':
    import yaml

    class Config():
        def __init__(self, yaml_path):
            yaml_file = open(yaml_path)
            self._attr = yaml.load(yaml_file, Loader=yaml.FullLoader)

        def __setattr__(self, name, value):
            self.__dict__[name] = value

        def __getattr__(self, attr):
            try:
                return self._attr[attr]
            except KeyError:
                try:
                    return self.__dict__[attr]
                except KeyError:
                    return None

        def __str__(self):
            print("##########   CONFIGURATION INFO   ##########")
            pretty(self._attr)
            return '\n'
            
    def pretty(d, indent=0):
        for key, value in d.items():
            print('    ' * indent + str(key) + ':', end='')
            if isinstance(value, dict):
                print()
                pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))

    config = Config('/home/nhamnx28/ocr/ocr_receipt/test_folder/my_project/tool/config/receipt_segmentation/configs.yaml')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    img_path = '/home/nhamnx28/ocr/ocr_receipt/test_folder/my_project/text_detection/bhx_test.jpg'
    model = RSEGMENT(config)
    model.predict(img_path)


