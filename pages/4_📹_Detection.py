import time
import os
import io
import numpy as np
import pandas as pd
from PIL import Image
from utils import make_zip, make_folder, make_folder_only, label_name, computer_vision as cs

# Package for Streamlit
from streamlit import session_state as state
import streamlit as st
from datetime import datetime
import pytz
import cv2

# Package for Machine Learning
import torch
from ultralytics import YOLO

if 'PATH' not in state.keys():
    state['PATH'] = '.'

PATH = state['PATH']

# Title
image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
st1, st2, st3 = st.columns(3)

with st2:
    st.image(image)

st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
            'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
st.markdown('<h3 style=\'text-align:center;\'>Detection Model</h3>', unsafe_allow_html=True)

try:
    restriction = state['login']
except (Exception,):
    state['login'] = False
    restriction = state['login']

if not restriction:
    st.warning('Please login with your registered email!')
else:
    path_object = {'General Detection': 'general-detect',
                   'Coal Detection': 'front-coal',
                   'Seam Detection': 'seam-gb',
                   'Core Detection': 'core-logging',
                   'Smart-HSE': 'hse-monitor'}

    tab1, tab2 = st.tabs(['🎦 Video', '📷 Image'])

    with tab1:
        kind_object = st.selectbox('Please select the kind of object detection do you want.',
                                   ['General Detection',
                                    'Coal Detection',
                                    'Seam Detection',
                                    'Core Detection',
                                    'Smart HSE'],
                                   key='kind-object-detection-1')

        conf = st.slider('Number of Confidence (%)',
                         min_value=0,
                         max_value=100,
                         step=1,
                         value=50,
                         key='confidence-detection-1')
        stop_program = st.slider('Number of Image',
                                 min_value=0,
                                 max_value=500,
                                 step=1,
                                 value=20,
                                 key='stop-program-detection-1')

        st4, st5 = st.columns(2)

        with st4:
            custom = st.radio('Do you want to use custom model that has trained?',
                              ['Yes', 'No'],
                              index=1,
                              key='custom-detection-1')
        with st5:
            type_camera = st.radio('Do you want to use webcam/camera for detection?',
                                   ['Yes', 'No'],
                                   index=1,
                                   key='camera-detection-1')

        st6, st7 = st.columns(2)

        with st6:
            if custom == 'Yes':
                option_model = f'{PATH}/results/{path_object[kind_object]}/weights/best.pt'
                model = YOLO(option_model)
                st.success('The model have successfully loaded!', icon='✅')
            else:
                list_weights = [weight_file for weight_file in
                                os.listdir(f'{PATH}/weights/{path_object[kind_object]}')]
                option_model = st.selectbox('Please select model do you want.',
                                            list_weights,
                                            key='option-model-detection-1')
                model = YOLO(f'{PATH}/weights/{path_object[kind_object]}/{option_model}')

        with st7:
            if type_camera == 'Yes':
                source = st.text_input('Please input your Webcam link.', 'Auto')
                if source == 'Auto':
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoCapture(source)
            else:
                list_files = [file for file in os.listdir(f'{PATH}/datasets/{path_object[kind_object]}/predict')]
                sample_video = st.selectbox('Please select sample video do you want.',
                                            list_files,
                                            key='sample-video-detection-1')
                source = f'{PATH}/datasets/{path_object[kind_object]}/predict/{sample_video}'
                cap = cv2.VideoCapture(source)

        show_label = st.checkbox('Show label predictions',
                                 value=True,
                                 key='show-label-detection-1')
        save_annotate = st.checkbox('Save annotate and images',
                                    value=False,
                                    key='save-annotate-detection-1')

        next_detect = st.button('Process',
                                key='next_detect',
                                use_container_width=True)

        if next_detect:
            if torch.cuda.is_available():
                st.success(
                    f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
                device = 0
            else:
                st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
                device = 'cpu'

            path_detections = f'{PATH}/detections/{path_object[kind_object]}'
            make_folder(path_detections)

            count = 0
            placeholder = st.empty()
            colors = cs.generate_label_colors(model.names)

            # Detection Model
            while cap.isOpened() and count < stop_program:
                with placeholder.container():
                    ret, img = cap.read()

                    if ret:
                        tz_JKT = pytz.timezone('Asia/Jakarta')
                        time_JKT = datetime.now(tz_JKT).strftime('%d-%m-%Y %H:%M:%S')
                        caption = f'The frame image-{label_name(count, 10000)} generated at {time_JKT}'

                        x_size = 640
                        y_size = 640
                        img = cv2.resize(img, (x_size, y_size), interpolation=cv2.INTER_AREA)
                        img, parameter, annotate = cs.draw_image(model, device, img, conf / 100, colors, time_JKT,
                                                                 x_size, y_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img,
                                 channels='RGB',
                                 use_column_width='always',
                                 caption=caption)

                        df1 = pd.DataFrame(parameter)
                        df2 = pd.DataFrame(annotate)

                        if show_label:
                            st.table(df1)

                        if save_annotate:
                            name_image = f'{PATH}/detections/{path_object[kind_object]}/images/' \
                                         f'{label_name(count, 10000)}.png'
                            cv2.imwrite(name_image, img)

                            name_annotate = f'{PATH}/detections/{path_object[kind_object]}/annotations/' \
                                            f'{label_name(count, 10000)}.txt'
                            with open(name_annotate, 'a') as f:
                                df_string = df2.to_string(header=False, index=False)
                                f.write(df_string)

                        count += 1
                        time.sleep(0.5)

                    else:
                        st.error('Image is not found', icon='❎')

            if save_annotate:
                st.success('Your all images have successfully saved', icon='✅')

    with tab2:
        kind_object = st.selectbox('Please select the kind of object detection do you want.',
                                   ['General Detection',
                                    'Coal Detection',
                                    'Seam Detection',
                                    'Core Detection',
                                    'Smart HSE'],
                                   key='kind-object-detection-2')

        conf = st.slider('Number of Confidence (%)',
                         min_value=0,
                         max_value=100,
                         step=1,
                         value=50,
                         key='confidence-detection-2')

        st8, st9 = st.columns(2)

        with st8:
            custom = st.radio('Do you want to use custom model that has trained?',
                              ['Yes', 'No'],
                              index=1,
                              key='custom-detection-2')
        with st9:
            if custom == 'Yes':
                option_model = f'{PATH}/results/{path_object[kind_object]}/weights/best.pt'
                model = YOLO(option_model)
                st.success('The model have successfully loaded!', icon='✅')
            else:
                list_weights = [weight_file for weight_file in os.listdir(f'weights/{path_object[kind_object]}')]
                option_model = st.selectbox('Please select model do you want.',
                                            list_weights,
                                            key='select-model-detection-2')
                model = YOLO(f'{PATH}/weights/{path_object[kind_object]}/{option_model}')

        colors = cs.generate_label_colors(model.names)


        def next_photo(path_images, func):
            if func == 'next':
                st.session_state.counter += 1
                if st.session_state.counter >= len(path_images):
                    st.session_state.counter = 0
            elif func == 'back':
                st.session_state.counter -= 1
                if st.session_state.counter >= len(path_images):
                    st.session_state.counter = 0
                elif st.session_state.counter < 0:
                    st.session_state.counter = len(path_images) - 1


        def save_photo(path_images_1, func, img_file, annotate_file):
            directory = f'{PATH}/detections/custom-data/{path_object[kind_object]}'
            make_folder_only(directory)

            image_name = f'{directory}/images/{label_name(st.session_state.counter, 10000)}.png'
            img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_name, img_file)

            annotate_name = f'{directory}/annotations/{label_name(st.session_state.counter, 10000)}.txt'

            try:
                df = pd.DataFrame(annotate_file)
                with open(annotate_name, 'a') as f:
                    df1_string = df.to_string(header=False, index=False)
                    f.write(df1_string)
            except (Exception,):
                df = pd.DataFrame([0, 0, 0, 0],
                                  columns=['id', 'x', 'y', 'w', 'h'])
                with open(annotate_name, 'a') as data:
                    df2_string = df.to_string(header=False, index=False)
                    data.write(df2_string)

            next_photo(path_images_1, func)


        # if extension_file:
        if torch.cuda.is_available():
            st.success(
                f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
            device = 0
        else:
            st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
            device = 'cpu'

        with st.form("form-upload-image", clear_on_submit=True):
            uploaded_files = st.file_uploader("Upload your image",
                                              type=['jpg', 'jpeg', 'png'],
                                              accept_multiple_files=True)
            st.form_submit_button("Process",
                                  use_container_width=True)

        image_files = [Image.open(io.BytesIO(file.read())) for file in uploaded_files]

        if 'counter' not in st.session_state:
            st.session_state.counter = 0

        tz_JKT = pytz.timezone('Asia/Jakarta')
        time_JKT = datetime.now(tz_JKT).strftime('%d-%m-%Y %H:%M:%S')

        try:
            x_size, y_size = 650, 650

            try:
                photo = image_files[st.session_state.counter]
            except (Exception,):
                st.session_state.counter = 0
                photo = image_files[st.session_state.counter]

            caption = f'The frame image-{st.session_state.counter} generated at {time_JKT}'
            photo_convert = np.array(photo.convert('RGB'))

            st10, st11 = st.columns(2)

            with st10:
                st10.write("Original Image")

                # photo_rgb = cv2.resize(photo_convert, (x_size, y_size), interpolation=cv2.INTER_AREA)
                # photo_rgb = cv2.cvtColor(photo_convert, cv2.COLOR_BGR2RGB)

                st10.image(photo_convert,
                           channels='RGB',
                           use_column_width='always',
                           caption=caption)
            with st11:
                st11.write("Detection Image")
                photo_detect, parameter, annotate = cs.draw_image(model, device, photo_convert, conf / 100, colors,
                                                                  time_JKT, x_size, y_size)
                # photo_rgb = cv2.resize(photo_detect, (x_size, y_size), interpolation=cv2.INTER_AREA)
                # photo_rgb = cv2.cvtColor(photo_detect, cv2.COLOR_BGR2RGB)

                st11.image(photo_detect,
                           channels='RGB',
                           use_column_width='always',
                           caption=caption)

            st12, st13, st14, st15, st16 = st.columns(5)

            with st13:
                st13.button('◀️ Back',
                            on_click=next_photo,
                            use_container_width=True,
                            args=([image_files, 'back']),
                            key='back-photo-detection-1')
            with st14:
                save = st14.button('Save 💾',
                                   on_click=save_photo,
                                   use_container_width=True,
                                   args=([image_files, 'save', photo_detect, annotate]),
                                   key='save-photo-detection-1')

            with st15:
                st15.button('Next ▶️',
                            on_click=next_photo,
                            use_container_width=True,
                            args=([image_files, 'next']),
                            key='next-photo-detection-1')

            if save or os.path.exists(f'{PATH}/detections/custom-data/{path_object[kind_object]}'):
                btn = st.radio('Do you want to download image in single or all files?',
                               ['Single files', 'All files'],
                               index=0,
                               key='download-button-1')

                if btn == 'Single files':
                    st17, st18 = st.columns(2)

                    with st17:
                        path_images = f'{PATH}/detections/custom-data/{path_object[kind_object]}/images'
                        image_name = f'{path_images}/{label_name(st.session_state.counter, 10000)}.png'

                        with open(image_name, 'rb') as file:
                            st17.download_button(label='🔗 Image (.png)',
                                                 data=file,
                                                 use_container_width=True,
                                                 file_name=f'{label_name(st.session_state.counter, 10000)}.png',
                                                 mime="image/png",
                                                 key='download-image-2')

                    with st18:
                        path_annotate = f'{PATH}/detections/custom-data/{path_object[kind_object]}/annotations'
                        annotate_name = f'{path_annotate}/{label_name(st.session_state.counter, 10000)}.txt'

                        with open(annotate_name, 'rb') as file:
                            st18.download_button(label='🔗 Annotation (.txt)',
                                                 data=file,
                                                 use_container_width=True,
                                                 file_name=f'{label_name(st.session_state.counter, 10000)}.txt',
                                                 mime="text/plain",
                                                 key='download-annotate-2')

                elif btn == 'All files':
                    path_folder = f'{PATH}/detections/custom-data/{path_object[kind_object]}'
                    name = path_object[kind_object]
                    make_zip(path_folder, name)

                    with open(f'{path_folder}/{name}.zip', "rb") as fp:
                        st.download_button(label="🔗 Download All Files (.zip)",
                                           data=fp,
                                           use_container_width=True,
                                           file_name=f'detection_{name}.zip',
                                           mime="application/zip",
                                           key='download-zip-2'
                                           )
        except (Exception,):
            pass
