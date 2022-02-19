import streamlit as st
import urllib.request
import os
import onnxruntime

onnx_model_url = 'https://github.com/onnx/models/blob/main/vision/classification/mnist/model/mnist-8.onnx?raw=true'
if not os.path.exists('mnist-8.onnx'):
  urllib.request.urlretrieve(onnx_model_url, filename='mnist-8.onnx')
  st.write('Model downloaded')

if os.path.exists('mnist-8.onnx'):
  session = onnxruntime.InferenceSession('mnist-8.onnx', None)
  st.write('Model loaded')

upload_file = st.file_uploader(label='Upload Image File')

if upload_file is not None:
  # To read file as bytes:
  bytes_data = upload_file.getvalue()
  st.write(bytes_data)
