import os
import PIL
import streamlit as st
from ultralytics import YOLO
import subprocess
import shutil 

i = 0
check_item = 0
count_pic = 1
model_path = 'Yolo_mural_85.pt'
output_folder = 'detected_images'  
class_count = {}
zip_file_path = 'detected_images.zip' 

st.set_page_config(
    page_title="Mural Detection using YOLOv8",  
    page_icon="🦖",     
    layout="wide",     
    initial_sidebar_state="expanded"    
)

with st.sidebar:
    st.image("img1.jpg")
    st.header("Uplode your Image")     
    source_imgs = st.file_uploader(
        "Choose one or more images...", 
        type=("jpg", "jpeg", "png", 'bmp', 'webp'),
        accept_multiple_files=True)  

    confidence = float(st.slider(
        "Select Model Confidence (%)", 25, 100, 40)) / 100

st.title("Mural Detection")

if st.sidebar.button('Detect Objects'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for index, source_img in enumerate(source_imgs):
        col1, col2 = st.columns(2)

        with col1:
            if source_img:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img,
                         caption=f"Uploaded Image {index+1}",
                         use_column_width=True
                         )

        try:
            model = YOLO(model_path)
        except Exception as ex:
            st.error(
                f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)
        
        res = model.predict(uploaded_image, conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        
        with col2:
            st.image(res_plotted,
                     caption=f'Detected Image {index+1}',
                     use_column_width=True
                     )
            try:
                with st.expander(f"Detection Results {index+1}"):
                    for box in boxes:
                        class_name = model.names[int(box.cls)]
                        if class_name not in class_count:
                            class_count[class_name] = 0
                        class_count[class_name] += 1
            except Exception as ex:
                st.write("No image is uploaded yet!")

        st.subheader(f"Class Count Detected Image {index+1}")
        for class_name, count in class_count.items():
            if(class_count[class_name] != 0):
                st.write(f'{class_name}: {count}') 
                check_item += 1  
        for class_name, count in class_count.items():
            class_count[class_name] = 0
            
        os.makedirs(output_folder, exist_ok=True)   
        for i, box in enumerate(boxes):
            x_center, y_center, width, height = map(int, box.xywh[0].tolist())
            x_min = x_center - (width // 2)
            y_min = y_center - (height // 2)
            x_max = x_center + (width // 2)
            y_max = y_center + (height // 2)

            cropped_image = uploaded_image.crop((x_min, y_min, x_max, y_max))
            cropped_image.save(os.path.join(output_folder, f'detected_{count_pic+i}.jpg'))
           
        count_pic += i
        if(check_item > 0):
            st.write(f"total {i+1}") 
            check_item = 0
        else:
            st.write("No items were detected !")
            count_pic -= i
            
    shutil.make_archive('detected_images', 'zip', output_folder)
    st.success("Detection and ZIP file creation completed.")

def remove_files():
    try:
        shutil.rmtree(output_folder)
        os.remove(zip_file_path)
    except Exception as e:
        st.error(f"Error deleting files: {e}")

if os.path.exists(zip_file_path):
    with open(zip_file_path, "rb") as fp:
        btn = st.sidebar.download_button(
            label="Download ZIP",
            data=fp,
            file_name="detected_images.zip",
            mime="application/zip",
            on_click=remove_files
        )
