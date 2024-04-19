import streamlit as st 
import numpy as np  
import cv2
from PIL import Image
from inference import get_model
import supervision as sv


model = get_model(model_id="drone_07/1")


with st.sidebar:
    st.title('AI Zend MVP Project')
    st.image('logo.jpeg')
    options = st.radio('SELECT',['drones','birds'])
    st.title('About this app')
    st.markdown("This app helps you detect birds or drones by uploading an image of it.")

if options == 'drones':
    st.header(f':red[{options.upper()} SELECTED]')
    st.image('drones.jpeg')
    image = st.file_uploader("Take a picture")
    if image:
        img = Image.open(image).convert("RGB")
        img_array = np.array(img)
        results = model.infer(img_array)
        detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
        bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=8)
        label_annotator = sv.LabelAnnotator(text_thickness=5,text_scale=4)

        annotated_image = bounding_box_annotator.annotate(
        scene=img_array, detections=detections)
        annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
        st.image(annotated_image,use_column_width=True)
        st.subheader(':red[_Drone_ Description Coming Soon :sunglasses:]')
        


        

