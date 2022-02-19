import streamlit as st
import urllib.request
import os
import onnxruntime
import mnist
from matplotlib import pyplot as plt

onnx_model_url = 'https://github.com/onnx/models/blob/main/vision/classification/mnist/model/mnist-8.onnx?raw=true'
if not os.path.exists('mnist-8.onnx'):
  urllib.request.urlretrieve(onnx_model_url, filename='mnist-8.onnx')
  st.write('Model downloaded')

if os.path.exists('mnist-8.onnx'):
  session = onnxruntime.InferenceSession('mnist-8.onnx', None)
  st.write('Model loaded')

test_images = mnist.test_images()

chosen_index = st.selectbox('Choose an image', (0, 1, 2))
plt.imshow(test_images[int(chosen_index)], interpolation='nearest')
st.pyplot(plt)
  
upload_file = st.file_uploader(label='Upload Image File')

if upload_file is not None:
  # To read file as bytes:
  bytes_data = upload_file.getvalue()
  st.write(bytes_data)
