import streamlit as st
from PIL import Image

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
st.markdown('<div style="background-color:#00285A;text-align:center;margin-bottom:5rem"><h1 style="color:#fff">AI POLLUTION INSPECTOR<h1></div>',unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file")
cols = []
for ii in range(10):
    image = Image.open('sunrise.jpg')
    tp_col1,tp_col2 = st.columns(2)
    cols.append([tp_col1, tp_col2])
    tp_col1.image(image, caption='Smoky vehicle detected')

    # col2.markdown("<p style='text-align: center;'>License plate number : <strong>"+ "PT ST 300" + "</strong></p>", unsafe_allow_html=True)
    tp_col2.markdown('License plate number : **' + "PT ST 300" + '**')
    tp_col2.markdown('Should we send challan?')
    tp_col2.button("Send", key="key"+str(ii))