import streamlit as st
import io
import os
import onnxruntime
import mnist
import numpy as np
import urllib.request
from PIL import Image
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

if st.button('Predict'):
  output = session.run([], {input_name: [[test_images[int(chosen_index) - 1].astype(np.float32)]]})[0]
  #print(np.argmax([o[0] for o in outputs], axis=1))
  st.write(np.argmax(output))

plt.imshow(test_images[int(chosen_index) - 1], interpolation='nearest')
st.pyplot(plt)

# Ref: https://docs.streamlit.io/library/api-reference/widgets/st.button
if st.button('Download chosen image'):
  # Ref: https://gist.github.com/xkumiyu/c93222f2dce615f4b264a9e71f6d49e0
  Image.fromarray(test_images[int(chosen_index) - 1].reshape(28, 28)).save('user-download.png')
  st.write('Image ready for download')
  with open('user-download.png', "rb") as file:
    # Ref: https://docs.streamlit.io/library/api-reference/widgets/st.download_button
    btn = st.download_button(label="Download",
                             data=file,
                             file_name="mnist_test_%i.png" % (int(chosen_index) - 1),
                             mime="image/png"
                            )

upload_file = st.file_uploader(label='Predict uploaded MNIST image file')

if upload_file is not None:
  # To read file as bytes:
  bytes_data = upload_file.getvalue()
  uploaded_image = Image.open(io.BytesIO(bytes_data))
  # Ref: https://docs.streamlit.io/library/api-reference/media/st.image
  st.image(uploaded_image, 'Uploaded image')
  output = session.run([], {input_name: [[ np.asarray(uploaded_image).reshape(28, 28).astype(np.float32)]]})[0]
  st.write('Predicted output: %d' % np.argmax(output))
