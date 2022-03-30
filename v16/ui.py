import infer
import mailer
import re
import streamlit as st
import tensorflow as tf
import tempfile
import cv2
import license_plate as lp
import numpy as np 

PAGE_TITLE = "AI POLLUTION INSPECTOR"

# LIST_OF_LP = ['']
# if st.button('Run Segmentation'):
#         start_time_2 = time.time()
#         semantic_image = do_semantic(image_path)
#         end_time_2 = time.time() - start_time_2
#         st.write("**Semantic Segmentation**")
#         st.image(semantic_image, caption=file_name)
#         st.write(end_time_2)

def smoke_detection_ui(frame):
    no_of_boxes, result_image = infer.predict_infer(frame)
    if no_of_boxes > 0: 
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        try:
            lp_no = lp.license_plate_pipeline(result_image)
            lp_no = lp_no.lower()
            lp_no = re.sub(r'[^\w]', '', lp_no)
            lp_no = lp_no.strip()
            
            if lp_no not in LIST_OF_LP:
                LIST_OF_LP.append(lp_no)
                st.image(result_image, caption="Detected image", width=300)
                st.write('License plate : ' + lp_no)
                st.write('Do we send mail')
                if st.button('Yes', key=lp_no+"y"):
                    mailer.send_mail(lp_no)
                    st.write("Mail sent")
                if st.button('No', key=lp_no+"n"):
                    st.write("Mail cancelled")
        except:
            print("SOME ERROR OCCURED, SKIPPING THE FRAME")
            # st.image(result_image, caption="SKIPPED FRAME", width=300)

def video_upload():
    f = st.file_uploader("Choose a video")
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
        cap = cv2.VideoCapture(tfile.name)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cnt = 0
        while(cap.isOpened()):
            ret, frame = cap.read() 
            if ret:
                smoke_detection_ui(frame)
                # print(np.shape(frame))
                # infer.predict_infer(frame)
            else:
               print('no video')
               cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
               break
            print( "The progress is : ", cnt, "/",length )
            cnt += 1
        cap.release()
        st.markdown("PROCCESSING COMPLETED", unsafe_allow_html=False)


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    video_upload()
    # image_path = file_selector_ui()

if __name__ == "__main__":
    main()