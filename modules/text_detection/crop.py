import os
import numpy as np
import cv2
import pandas as pd

class Cropper():
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def crop_utils(self, pts):

        """
        Takes inputs as 8 points
        and Returns cropped, masked image with a white background
        """
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        cropped = self.img.copy().crop((x,y,x+w,y+h))
        return cropped


    def crop(self, image, df):
        self.score_bbox = df['Scores']
        self.bbox_coords = df['BBox']
        self.img = image.copy()
        num_bboxes = len(self.score_bbox)
        for num in range(num_bboxes):
            bbox_coords = self.bbox_coords[num]
            if bbox_coords.size != 0:
                l_t = float(bbox_coords[0][0])
                t_l = float(bbox_coords[0][1])
                r_t = float(bbox_coords[1][0])
                t_r = float(bbox_coords[1][1])
                r_b = float(bbox_coords[2][0])
                b_r = float(bbox_coords[2][1])
                l_b = float(bbox_coords[3][0])
                b_l = float(bbox_coords[3][1])
                pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
                
                if np.all(pts) > 0:
                    
                    box = self.crop_utils(pts)
                    try:
                        file_name = os.path.join(self.output_folder,f'{num}.jpg')
                        print(os.getcwd())
                        print(file_name)
                        box.save(file_name)
                    except:
                        continue
                else:
                    print('None')