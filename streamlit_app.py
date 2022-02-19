import streamlit as st

upload_file = st.file_uploader(label='Upload Image File')

if uploaded_file is not None:
     # To read file as bytes:
     bytes_data = uploaded_file.getvalue()
     st.write(bytes_data)
