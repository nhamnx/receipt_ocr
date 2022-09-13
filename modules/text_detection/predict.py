from ctypes import util
import sys
import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
import cv2
import pandas as pd
from skimage import io
import numpy as np
from .CRAFT_pytorch import utils
from .CRAFT_pytorch.craft import CRAFT
from collections import OrderedDict

class Detector():
    def __init__(self, config):
        self.config = config
        self.net = CRAFT()
        if self.config.cuda:
            self.net.load_state_dict(utils.copyStateDict(torch.load(self.config.text_detect_weight)))
        else:
            self.net.load_state_dict(utils.copyStateDict(torch.load(self.config.text_detect_weight, map_location='cpu')))

        if self.config.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
        self.net.eval()
        

    def predict(self, img):
        result_img = img.copy()
        bboxes, polys, score_text, det_scores = utils.test_net(self.net, result_img, \
                                                            self.config.text_threshold, 
                                                            self.config.link_threshold, 
                                                            self.config.low_text, 
                                                            self.config.cuda, self.config.poly)
        bbox_score={}

        for box_num in range(len(bboxes)):
          key = str (det_scores[box_num])
          item = bboxes[box_num]
          bbox_score[key]=item
        df = pd.DataFrame(bbox_score.items(), columns=['Scores', 'BBox'])

        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            cv2.polylines(result_img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        return result_img, df
    
    