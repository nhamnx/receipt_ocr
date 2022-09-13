import time
import os
from PIL import Image
import cv2
import numpy as np
from tool.config import Config
from modules import Segmentation, Preprocess, TextDetection, TextCropper

class Pipeline:
    def __init__(self, config):
        self.output = config.output
        self.load_config(config)
        self.make_cache_folder()
        self.init_modules()

    def load_config(self, config):
        self.seg_config = config.seg_config
        self.seg_weight = config.seg_weight
        self.text_detect_config = config.text_detect_config

    def make_cache_folder(self):
        self.cache_folder = os.path.join(self.output, 'cache')
        os.makedirs(self.cache_folder,exist_ok=True)
        self.segment_cache = os.path.join(self.cache_folder, "segment_crop.jpg")
        self.processed_cache = os.path.join(self.cache_folder, "processed.jpg")
        self.detection_cache = os.path.join(self.cache_folder, 'detection.jpg')
        self.csv_cache = os.path.join(self.cache_folder, 'bbox.csv')
        self.crop_cache = os.path.join(self.cache_folder, 'crops')
        os.makedirs(self.crop_cache,exist_ok=True)
        self.retr_output = os.path.join(self.output, 'result.txt')
    
    def init_modules(self):
        self.seg_model = Segmentation(config_path=self.seg_config)
        self.preprocess = Preprocess()
        self.text_detect = TextDetection(config_path=self.text_detect_config)
        self.text_crop = TextCropper(output_folder=self.crop_cache)

    def start(self, image):
        resize_img, X1, Y1, X2, Y2 = self.seg_model(image)
        crop_img = Image.fromarray(resize_img).crop((X1, Y1, X2, Y2))
        crop_img.save(self.segment_cache)
        processed_img = self.preprocess(np.array(crop_img))
        bbox_img, bbox_data = self.text_detect(np.array(crop_img))
        bbox_data.to_csv(self.csv_cache, sep = ',', na_rep='Unknown')
        self.text_crop(crop_img, bbox_data)
        cv2.imwrite(self.processed_cache, processed_img)
        cv2.imwrite(self.detection_cache, bbox_img)

if __name__ == '__main__':
    config = Config('./tool/config/configs.yaml')
    pipeline = Pipeline(config)
    image = cv2.imread('/home/nhamnx28/ocr/ocr_receipt/test_folder/my_project/demo_img/bhx_test.jpg')
    start_time = time.time()
    pipeline.start(image)
    end_time = time.time()
    print(f"Executed in {end_time - start_time} s")