import streamlit as st
import os
import onnxruntime
import mnist
import numpy as np
import urllib.request
from matplotlib import pyplot as plt

onnx_model_url = 'https://github.com/onnx/models/blob/main/vision/classification/mnist/model/mnist-8.onnx?raw=true'
if not os.path.exists('mnist-8.onnx'):
  urllib.request.urlretrieve(onnx_model_url, filename='mnist-8.onnx')
  st.write('Model downloaded')

if os.path.exists('mnist-8.onnx'):
  session = onnxruntime.InferenceSession('mnist-8.onnx', None)
  input_name = session.get_inputs()[0].name
  st.write('Model loaded')

test_images = mnist.test_images()

chosen_index = st.number_input('Choose an image', min_value=1, max_value=10000, value=1, step=1)
plt.imshow(test_images[int(chosen_index) - 1], interpolation='nearest')
st.pyplot(plt)
  
upload_file = st.file_uploader(label='Upload Image File (TODO)')

if upload_file is not None:
  # To read file as bytes:
  bytes_data = upload_file.getvalue()
  st.write(bytes_data)

if st.button('Predict'):
  output = session.run([], {input_name: [[test_images[int(chosen_index) - 1].astype(np.float32)]]})[0]
  #print(np.argmax([o[0] for o in outputs], axis=1))
  st.write(np.argmax(output) + 1)
