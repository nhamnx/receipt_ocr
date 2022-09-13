import streamlit as st
from importlib.resources import path
from modules import Preprocess, OCR
import cv2
from PIL import Image
import numpy as np  



def main():
    st.title("Demo Web App")
    docscanner = Preprocess()
    ocr = OCR('vie')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None
    original_image = Image.open(image_file)
    original_image = np.array(original_image)
    res, res_processed = docscanner(original_image)
    text = ocr(res)
    col1, col2, col3, col4 = st.columns(4)
    col1.header("Original")
    col1.image(original_image, use_column_width=True)
    col2.header("Crop")
    col2.image(res, use_column_width=True)
    col3.header("Preprocess")
    col3.image(res_processed, use_column_width=True)
    col4.header("OCR")
    col4.write(text)
    # from io import BytesIO
    # img_from_array = Image.fromarray(res.astype('uint8'))
    # buf = BytesIO()
    # img_from_array.save(buf, format="JPEG")
    # byte_im = buf.getvalue()
    # btn = st.download_button(
    #   label="Download Image",
    #   data=byte_im,
    #   file_name="st_download.png",
    #   mime="image/jpeg",
    #   )

if __name__ == '__main__':
    main()