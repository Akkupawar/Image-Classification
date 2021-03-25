import joblib
import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
st.title('Clasification Of Image')
st.text('Upload Image Here : ')

model = joblib.load(open('img_classi.p','rb'))

Uploaded_file = st.file_uploader("Choose an image :",type = "jpg")
if Uploaded_file is not None:
  img = Image.open(Uploaded_file)
  st.image(img,caption='Uploaded Image')

if st.button('PREDICT'):
  categories = ['Cats','Dogs']
  st.write('Result :')
  flat_data=[]
  img = np.array(img)
  img_resize = resize(img,(150,150,3))
  flat_data.append(img_resize.flatten())
  flat_data = np.array(flat_data)
  y_out = model.predict(flat_data)
  y_out = categories[y_out[0]]
  st.title(f'PREDICTED OUTPUT : {y_out}')
  
