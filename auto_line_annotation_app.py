import streamlit as st
# import tensorflow as tf
import keras
from keras_preprocessing.image import load_img, array_to_img, img_to_array
from skimage import filters
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import requests
from io import BytesIO


def lineSegmentationCXR(image_path):
    image = load_img(image_path, color_mode='grayscale', target_size=(256, 256))

    image_arr = img_to_array(image).astype('float32') / 255.0
    image_arr_reshape = image_arr[np.newaxis, ...]  # Reshape menjadi (1, width, height, channel) agar bisa predict

    # Prediksi Model UNet
    UNet_trained_model = keras.models.load_model('UNet_Model_Trained.h5', compile=False)
    mask_arr = UNet_trained_model.predict(image_arr_reshape)

    # Create Line Segmentation
    line_mask_arr = filters.sobel(mask_arr[0])

    # Convert Image to uint8
    image_arr_uint8 = (image_arr*255).astype(np.uint8)
    line_mask_arr_uint8 = (line_mask_arr*255).astype(np.uint8)

    # Convert Image to PIL
    image_pil = Image.fromarray(image_arr_uint8[:, :, 0])
    line_mask_pil = Image.fromarray(line_mask_arr_uint8[:, :, 0])

    # Convert image to RGBA
    line_mask_pil_rgba = line_mask_pil.convert("RGBA")
    image_pil_rgba = image_pil.convert("RGBA")

    # Convert image to tuple data
    line_mask_pil_data = line_mask_pil_rgba.getdata()
    image_pil_data = image_pil_rgba.getdata()

    image_result = []
    for mask_line_channels, image_channels in zip(line_mask_pil_data, image_pil_data):
        if mask_line_channels != (0, 0, 0, 255):
            image_channels = list(image_channels)
            image_channels[0] = int((image_channels[0] + 180) / 2)
            image_channels[1] = int((image_channels[1] + 0) / 2)
            image_channels[2] = int((image_channels[2] + 0) / 2)
            # image_channels[3] = 120
            image_channels = tuple(image_channels)
            image_channels = image_channels
            image_result.append(image_channels)
        else:
            image_result.append(image_channels)
    image_pil_rgba.putdata(image_result)

    return image_pil_rgba


def get_image(url):
    img = requests.get(url)
    file = open("sample_image.jpg", "wb")
    file.write(img.content)
    file.close()
    img_file_name = 'sample_image.jpg'
    return img_file_name


def get_image_result(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    file = open("segmentation_result.jpg", "wb")
    file.write(img_byte_arr)
    file.close()
    img_file_name = 'segmentation_result.jpg'
    return img_file_name


# Main driver
st.title("Image Segmentation using UNet pretrained Model")
st.write("Using UNet Model to segmentation the image")

url = st.text_input("Enter Image Url:")
if url:
    image = get_image(url)
    st.image(image)
    segmentation = st.button("Create Annotation")
    if segmentation:
        st.write("")
        st.write("Annotationing...")
        line_image_predicted = lineSegmentationCXR(image)
        line_image = get_image_result(line_image_predicted)
        st.image(line_image)
else:
    st.write("Paste Image URL")
