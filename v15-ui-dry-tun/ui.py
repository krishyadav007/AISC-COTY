import streamlit as st
from PIL import Image
# m = st.markdown("""
# <style>
# div.stButton > button:first-child {
#      background-color: rgb(204, 49, 49);
# }
# </style>""", unsafe_allow_html=True)

image = Image.open('sunrise.jpg')

col1,col2 = st.columns(2)
col1.image(image, caption='Smoky vehicle detected')

# col2.markdown("<p style='text-align: center;'>License plate number : <strong>"+ "PT ST 300" + "</strong></p>", unsafe_allow_html=True)
col2.markdown('License plate number : **' + "PT ST 300" + '**')
col2.markdown('Should we send challan?')
col2.button("Send")
