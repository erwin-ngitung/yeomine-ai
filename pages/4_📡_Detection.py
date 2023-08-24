import time
import os
import io
import numpy as np
import pandas as pd
from PIL import Image
from utils import make_folder, label_name, computer_vision as cs

# Package for Streamlit
from streamlit import session_state as state
import streamlit as st
from datetime import datetime as dt
import datetime
import pytz
import cv2
import tempfile

# Package for Machine Learning
import torch
from ultralytics import YOLO

st.set_page_config(
    page_title="Detection | Yeomine App",
    page_icon="📡",
)

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

        if 'object-videos' in state.keys():
            del state['object-videos']

        state['object-videos'] = kind_object

        conf = st.slider('Number of Confidence (%)',
                         min_value=0,
                         max_value=100,
                         step=1,
                         value=50,
                         key='confidence-detection-1')

        st4, st5 = st.columns(2)

        with st4:
            custom = st.radio('Do you want to use custom model that has trained?',
                              ['Yes', 'No'],
                              index=1,
                              key='custom-detection-1')
        with st5:
            if custom == 'Yes':
                option_model = f'{PATH}/results/{path_object[kind_object]}/weights/best.pt'
                model = YOLO(option_model)
                st.success('The model have successfully loaded!', icon='✅')
            else:
                list_weights = [weight_file for weight_file in
                                os.listdir(f'{PATH}/weights/{path_object[kind_object]}')]
                list_model = st.selectbox('Please select model do you want.',
                                           list_weights,
                                           key='option-model-detection-1')
                option_model = f'{PATH}/weights/{path_object[kind_object]}/{list_model}'
                model = YOLO(option_model)

        if 'model-videos' in state.keys():
            del state['model-videos']

        state['model-videos'] = option_model

        type_file = st.radio('Do you want to upload your video or use link streaming?',
                             ['Yes', 'No'],
                             index=1,
                             key='camera-detection-1')

        if type_file == 'Yes':
            streaming_video = st5.text_input("Please input the link streaming if you want to use it.")

            uploaded_video = st.file_uploader("Upload your video file",
                                              type=['mp4', 'mkv', 'mpeg'],
                                              accept_multiple_files=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False)

            try:
                temp_file.write(uploaded_video.read())
                cap = cv2.VideoCapture(temp_file.name)
            except (Exception,):
                source = f'{PATH}/datasets/general-detect/predict/sample-video-01.mp4'
                cap = cv2.VideoCapture(source)

        else:
            list_files = [file for file in os.listdir(f'{PATH}/datasets/{path_object[kind_object]}/predict')]
            sample_video = st.selectbox('Please select sample video do you want.',
                                        list_files,
                                        key='sample-video-detection-1')
            source = f'{PATH}/datasets/{path_object[kind_object]}/predict/{sample_video}'
            cap = cv2.VideoCapture(source)

            process = True

        seconds, minutes, hours = cs.get_time(cap)

        stop_program = st.slider('Stop Time Video',
                                 min_value=datetime.time(0, 0, 1),
                                 max_value=datetime.time(hours, minutes, seconds),
                                 value=datetime.time(0, 0, 0),
                                 format="HH:mm:ss",
                                 step=datetime.timedelta(seconds=1),
                                 key='stop-program-detection-1')

        sum_seconds = stop_program.hour * 3600 + stop_program.minute * 60 + stop_program.second

        show_label = st.checkbox('Show label predictions',
                                 value=True,
                                 key='show-label-detection-1')
        save_annotate = st.checkbox('Save images and annotations',
                                    value=False,
                                    key='save-annotate-detection-1')

        process = st.button('Process',
                            key='next_detect',
                            use_container_width=True)

        if process:
            if torch.cuda.is_available():
                st.success(
                    f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
                device = 0
            else:
                st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
                device = 'cpu'

            path_detections = f'{PATH}/detections/videos/{path_object[kind_object]}'
            make_folder(path_detections)

            count = 0
            placeholder1 = st.empty()
            colors = cs.generate_label_colors(model.names)

            # Detection Model
            while cap.isOpened() and (count < sum_seconds):
                with placeholder1.container():
                    ret, img = cap.read()

                    if ret:
                        tz_JKT = pytz.timezone('Asia/Jakarta')
                        time_JKT = dt.now(tz_JKT).strftime('%d-%m-%Y %H:%M:%S')
                        caption = f'The frame image-{label_name(count, 10000)} generated at {time_JKT}'

                        x_size = 640
                        y_size = 640
                        img = cv2.resize(img, (x_size, y_size), interpolation=cv2.INTER_AREA)
                        img, parameter, annotate = cs.draw_image(model, device, img, conf / 100, colors, time_JKT,
                                                                 x_size, y_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        df1 = pd.DataFrame(parameter)
                        df2 = pd.DataFrame(annotate)

                        st.image(img,
                                 channels='RGB',
                                 use_column_width='always',
                                 caption=caption)

                        if show_label:
                            st.table(df1)

                        if save_annotate:
                            name_image = f'{PATH}/detections/videos/{path_object[kind_object]}/images/' \
                                         f'{label_name(count, 10000)}.png'
                            Image.fromarray(img).save(name_image)

                            name_annotate = f'{PATH}/detections/videos/{path_object[kind_object]}/annotations/' \
                                            f'{label_name(count, 10000)}.txt'
                            with open(name_annotate, 'a') as f:
                                df_string = df2.to_string(header=False, index=False)
                                f.write(df_string)

                        count += 1
                        time.sleep(0.5)

                    else:
                        st.error('Image is not found', icon='❎')

            if save_annotate:
                st.success('Your all images and annotations have successfully saved', icon='✅')

    with tab2:
        kind_object = st.selectbox('Please select the kind of object detection do you want.',
                                   ['General Detection',
                                    'Coal Detection',
                                    'Seam Detection',
                                    'Core Detection',
                                    'Smart HSE'],
                                   key='kind-object-detection-2')

        if 'object-pictures' in state.keys():
            del state['object-pictures']

        state['object-pictures'] = kind_object

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
                list_model = st.selectbox('Please select model do you want.',
                                           list_weights,
                                           key='select-model-detection-2')
                option_model = f'{PATH}/weights/{path_object[kind_object]}/{list_model}'
                model = YOLO(option_model)

        if 'model-pictures' in state.keys():
            del state['model-pictures']

        state['model-pictures'] = option_model

        colors = cs.generate_label_colors(model.names)

        show_label = st.checkbox('Show label predictions',
                                 value=True,
                                 key='show-label-detection-2')
        save_annotate = st.checkbox('Save images and annotations',
                                    value=False,
                                    key='save-annotate-detection-2')

        with st.form("form-upload-image", clear_on_submit=True):
            uploaded_files = st.file_uploader("Upload your image",
                                              type=['jpg', 'jpeg', 'png'],
                                              accept_multiple_files=True)
            process = st.form_submit_button("Process My Image",
                                            use_container_width=True)

        if process:
            if torch.cuda.is_available():
                st.success(
                    f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
                device = 0
            else:
                st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
                device = 'cpu'

            path_detections = f'{PATH}/detections/pictures/{path_object[kind_object]}'
            make_folder(path_detections)

            try:
                count = 0
                x_size, y_size = 650, 650
                placeholder2 = st.empty()

                for file in uploaded_files:
                    with placeholder2.container():
                        tz_JKT = pytz.timezone('Asia/Jakarta')
                        time_JKT = dt.now(tz_JKT).strftime('%d-%m-%Y %H:%M:%S')
                        caption = f'The frame image-{label_name(count, 10000)} generated at {time_JKT}'

                        photo = Image.open(io.BytesIO(file.read()))
                        photo_convert = np.array(photo.convert('RGB'))
                        img, parameter, annotate = cs.draw_image(model, device, photo_convert, conf / 100, colors,
                                                                 time_JKT, x_size, y_size)

                        img = cv2.resize(img, (x_size, y_size), interpolation=cv2.INTER_AREA)
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        st.image(img,
                                 channels='RGB',
                                 use_column_width='always',
                                 caption=caption)

                        df1 = pd.DataFrame(parameter)
                        df2 = pd.DataFrame(annotate)

                        if show_label:
                            st.table(df1)

                        if save_annotate:
                            name_image = f'{PATH}/detections/pictures/{path_object[kind_object]}/images/' \
                                         f'{label_name(count, 10000)}.png'
                            Image.fromarray(img).save(name_image)

                            name_annotate = f'{PATH}/detections/pictures/{path_object[kind_object]}/annotations/' \
                                            f'{label_name(count, 10000)}.txt'

                            with open(name_annotate, 'a') as f:
                                df_string = df2.to_string(header=False, index=False)
                                f.write(df_string)

                        count += 1

                if save_annotate:
                    st.success('Your all images and annotations have successfully saved', icon='✅')

            except (Exception,):
                st.error('Cannot process the image!', icon='❎')
