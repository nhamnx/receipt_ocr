import os
import cv2
import shutil
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from tool import Config 
import modules.receipt_segmentation as receipt_segmentation
import modules.text_detection as text_detection
from .text_detection import Cropper
from .preprocess import DocScanner

class Preprocess:
    def __init__(self):
        
        self.scanner = DocScanner()

    def __call__(self, image):
        output = self.scanner.scan(image)
        return output

class Segmentation:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = 'tool/config/receipt_segmentation/configs.yaml'
        self.config = Config(config_path)
        self.model = receipt_segmentation.RSEGMENT(self.config)
        
    def __call__(
        self, 
        image):
        resize_img, X1, Y1, X2, Y2 = self.model.predict(image)
        return (resize_img, X1, Y1, X2, Y2)
    
class TextDetection:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = 'tool/config/text_detection/configs.yaml'
        self.config = Config(config_path)
        self.model = text_detection.Detector(self.config)
    
    def __call__(
        self, 
        image):
        bbox_img = self.model.predict(image)
        return bbox_img

class TextCropper:
    def __init__(self, output_folder):
        self.cropper = Cropper(output_folder)
    
    def __call__(
        self, image, df):
        self.cropper.crop(image, df)