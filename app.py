import time
import os
import io
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
import logging
from PIL import Image
from utils import make_zip, make_zip_only, make_folder, make_folder_only, label_name, \
    check_email, check_account, update_json, replace_json, computer_vision as cs

# Package for Streamlit
import streamlit as st
from streamlit_multipage import MultiPage
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration
from datetime import datetime
import pytz
import pytesseract

import av
import cv2

# Package for Machine Learning
import torch
from ultralytics import YOLO
import wandb
import warnings

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
PATH = Path(Path(__file__).resolve()).parent
logger = logging.getLogger(__name__)

wandb.init(mode='disabled')

app = MultiPage()


def landing_page(st, **state):
    Image.open(f'{PATH}/data/images/logo_yeomine.png')
    st.markdown('Welcome to the Yeomine Website')
    MultiPage.save({'login': True})


def sign_up(st, **state):
    placeholder = st.empty()

    with placeholder.form('Sign Up'):
        image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
        st1, st2, st3 = st.columns(3)

        with st2:
            st.image(image)

        st.warning('Please sign up your account!')

        # name_ = state['name'] if 'name' in state else ''
        name = st.text_input('Name: ')

        # username_ = state['username'] if 'username' in state else ''
        username = st.text_input('Username: ')

        # email_ = state['email'] if 'email' in state else ''
        email = st.text_input('Email')

        # password_ = state['password'] if 'password' in state else ''
        password = st.text_input('Password', type='password')

        save = st.form_submit_button('Save')

    if save and check_email(email) == 'valid email':
        placeholder.empty()
        st.success('Hello ' + name + ', your profile has been save successfully')
        MultiPage.save({'name': name,
                        'username': username,
                        'email': email,
                        'password': password,
                        'login': True,
                        'edit': True})

        update_json(name, username, email, password)

    elif save and check_email(email) == 'duplicate email':
        st.success('Hello ' + name + ", your profile hasn't been save successfully because your email same with other!")

    elif save and check_email(email) == 'invalid email':
        st.success('Hello ' + name + ", your profile hasn't been save successfully because your email invalid!")
    else:
        pass


def login(st, **state):
    st.snow()
    # Create an empty container
    placeholder = st.empty()

    try:
        # Insert a form in the container
        with placeholder.form('login'):
            image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
            st1, st2, st3 = st.columns(3)

            with st2:
                st.image(image)

            st.markdown('#### Login Yeomine-AI Website')
            email = st.text_input('Email')
            password = st.text_input('Password', type='password')
            submit = st.form_submit_button('Login')

            st.write("Are you ready registered account in this app? If you don't yet, please sign up your account!")

            name, username, status = check_account(email, password)

        if submit and status == 'register':
            # If the form is submitted and the email and password are correct,
            # clear the form/container and display a success message
            placeholder.empty()
            st.success('Login successful')
            MultiPage.save({'name': name,
                            'username': username,
                            'email': email,
                            'password': password,
                            'login': True})

        elif submit and status == 'wrong password':
            st.error('Login failed because your password is wrong!')

        elif submit and status == 'not register':
            st.error("You haven't registered to this app! Please sign up your account!")

        else:
            pass

    except:
        st.error('Please login with your registered email!')


def training(st, **state):
    # Title
    image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
                'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
    st.markdown('<h3 style=\'text-align:center;\'>Train Custom Model</h3>', unsafe_allow_html=True)

    restriction = state['login']

    if 'login' not in state or not restriction:
        st.warning('Please login with your registered email!')
        return

    path_object = {'General Detection': 'general-detect',
                   'Coal Detection': 'front-coal',
                   'Seam Detection': 'seam-gb',
                   'Core Detection': 'core-logging',
                   'Smart-HSE': 'hse-monitor'}

    tab1, tab2, tab3, tab4 = st.tabs(['⌚ Training Model',
                                      '📊 Dashboard Model',
                                      '🎭 Validating Model',
                                      '📦 Download Model'])

    with tab1:
        with st.form("form-training", clear_on_submit=True):
            kind_object = st.selectbox('Please select the kind of object detection do you want.',
                                       ['General Detection',
                                        'Coal Detection',
                                        'Seam Detection',
                                        'Core Detection',
                                        'Smart-HSE'],
                                       key='kind-object-training-1')

            list_model = os.listdir(f'{PATH}/weights/petrained-model')
            kind_model = st.selectbox('Please select the petrained model.',
                                      list_model,
                                      key='kind-model-training-1')
            st4, st5 = st.columns(2)

            with st4:
                epochs = st4.number_input('Number of Epochs',
                                          min_value=1,
                                          max_value=200,
                                          step=1,
                                          key='epochs-training-1')
                imgsz = st4.number_input('Size of Image',
                                         min_value=50,
                                         max_value=1500,
                                         step=5,
                                         key='imgsz-training-1')
                batch = st4.number_input('Number of Batch Size',
                                         min_value=1,
                                         max_value=200,
                                         step=1,
                                         key='batch-training-1')

            with st5:
                lr_rate = st5.number_input('Number of Learning Rate',
                                           min_value=0.01,
                                           max_value=1.0,
                                           step=0.01,
                                           key='lr-rate-training-1')
                momentum = st5.number_input('Number of Size Rate',
                                            min_value=0.01,
                                            max_value=1.0,
                                            step=0.01,
                                            key='momentum-training-1')
                weight_decay = st5.number_input('Number of Weight Decay',
                                                min_value=0.01,
                                                max_value=1.0,
                                                step=0.01,
                                                key='weight-decay-training-1')

            list_yaml = os.listdir(f'{PATH}/data-yaml/{path_object[kind_object]}')
            path_yaml = st.selectbox('Please select your data YAML.',
                                     list_yaml,
                                     key='data-yaml-1')
            next_train = st.form_submit_button("Process")

        if next_train:
            if torch.cuda.is_available():
                st.success(
                    f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
                device = 0
            else:
                st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
                device = 'cpu'

            shutil.rmtree(f'{PATH}/results/{path_object[kind_object]}')

            # Load a model
            model = YOLO(f'{PATH}/weights/petrained-model/{kind_model}')
            model.train(data=f'{PATH}/data-yaml/{path_object[kind_object]}/{path_yaml}',
                        device=device,
                        epochs=int(epochs),
                        batch=int(batch),
                        imgsz=int(imgsz),
                        lrf=lr_rate,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        project='results',
                        name=path_object[kind_object])

            num_weights = len(os.listdir(f'{PATH}/weights/{path_object[kind_object]}'))
            src = f'{PATH}/results/{path_object[kind_object]}/weights/best.pt'
            dest = f'{PATH}/weights/{path_object[kind_object]}/{path_object[kind_object]}-' \
                   f'{label_name(num_weights, 10000)}.pt'

            shutil.copyfile(src, dest)

            st.success('The model have been successfully saved. Now you can download model in the button bellow',
                       icon='✅')

            path_folder = f'{PATH}/datasets/{path_object[kind_object]}/weights'
            name = f'{path_object[kind_object]}-{label_name(num_weights, 10000)}'
            make_zip_only(path_folder, src, name)

            with open(f'{path_folder}/{name}.zip', "rb") as fp:
                st.download_button(label="🔗 Download Weights Model (.zip)",
                                   data=fp,
                                   use_container_width=True,
                                   file_name=f'weight_{name}.zip',
                                   mime="application/zip",
                                   key='download-zip-1')

    with tab2:
        try:
            list_visual = ['Confusion Matrix',
                           'F1_curve',
                           'P_curve',
                           'PR_curve',
                           'R_curve',
                           'Summary']

            visual = st.selectbox('Please choose the curve of training model',
                                  list_visual,
                                  key='visual-training-1')

            if visual == 'Summary':
                visual = 'results'
            elif visual == 'Confusion Matrix':
                visual = 'confusion_matrix_normalized'

            st.image(f'{PATH}/results/{path_object[kind_object]}/{visual}.png',
                     caption=f'The image of {visual}')
        except:
            st.error('Please measure that you have trained model in the sub-menu training model.')

    with tab3:
        try:
            list_visual = ['labels',
                           'train_batch0',
                           'train_batch1',
                           'train_batch2',
                           'val_batch0_labels',
                           'val_batch0_pred']

            visual = st.selectbox('Please choose the validation image!',
                                  list_visual,
                                  key='visual-training-2')

            st.image(f'{PATH}/results/{path_object[kind_object]}/{visual}.jpg',
                     caption=f'The image of {visual}')
        except:
            st.error('Please measure that you have trained model in the sub-menu training model.')

    with tab4:
        try:
            kind_object = st.selectbox('Please select the kind of object detection that you want.',
                                       ['General Detection',
                                        'Coal Detection',
                                        'Seam Detection',
                                        'Core Detection',
                                        'Smart-HSE'],
                                       key='kind-object-training-2')

            list_weights = [weight_file for weight_file in os.listdir(f'{PATH}/weights/{path_object[kind_object]}')]
            option_model = st.selectbox('Please select model do you want.',
                                        list_weights,
                                        key='option-model-detection-1')

            path_folder = f'{PATH}/datasets/{path_object[kind_object]}/weights'
            src = f'{PATH}/weights/{path_object[kind_object]}/{option_model}'
            name = f'{option_model.split(".")[0]}'
            make_zip_only(path_folder, src, name)

            with open(f'{path_folder}/{name}.zip', "rb") as fp:
                st.download_button(label="🔗 Download Weights Model (.zip)",
                                   data=fp,
                                   use_container_width=True,
                                   file_name=f'weight_{name}.zip',
                                   mime="application/zip",
                                   key='download-zip-2')
        except:
            st.error('Please measure that you have trained model in the sub-menu training model.')


def detection(st, **state):
    # Title
    image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
                'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
    st.markdown('<h3 style=\'text-align:center;\'>Detection Model</h3>', unsafe_allow_html=True)

    restriction = state['login']

    if 'login' not in state or not restriction:
        st.warning('Please login with your registered email!')
        return

    path_object = {'General Detection': 'general-detect',
                   'Coal Detection': 'front-coal',
                   'Seam Detection': 'seam-gb',
                   'Core Detection': 'core-logging',
                   'Smart-HSE': 'hse-monitor'}

    tab1, tab2, tab3 = st.tabs(['🎬 Video', '📷 Image', '🎦 Streaming'])

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
                    cap = cv2.VideoStream(source).start()
            else:
                list_files = [file for file in os.listdir(f'{PATH}/datasets/{path_object[kind_object]}/predict')]
                sample_video = st.selectbox('Please select sample video do you want.',
                                            list_files,
                                            key='sample-video-detection-1')
                source = f'{PATH}/datasets/{path_object[kind_object]}/predict/{sample_video}'
                cap = cv2.VideoCapture(source)

        next_detect = st.button('Process',
                                key='next_detect')

        if next_detect:
            if torch.cuda.is_available():
                st.success(
                    f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
                device = 0
            else:
                st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
                device = 'cpu'
                
            st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' '
                        'stroke=\'black\' stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)

            path_detections = f'{PATH}/detections/{path_object[kind_object]}'
            make_folder(path_detections)

            show_label = st.checkbox('Show label predictions',
                                     value=True,
                                     key='show-label-detection-1')
            save_annotate = st.checkbox('Save annotate and images',
                                        value=False,
                                        key='save-annotate-detection-1')

            count = 0
            placeholder = st.empty()
            colors = cs.generate_label_colors(model.names)

            # Detection Model
            while count < stop_program:
                with placeholder.container():
                    ret, img = cap.read()

                    if ret:
                        tz_JKT = pytz.timezone('Asia/Jakarta')
                        time_JKT = datetime.now(tz_JKT).strftime('%d-%m-%Y %H:%M:%S')
                        caption = f'The frame image-{label_name(count, 10000)} generated at {time_JKT}'

                        x_size = 650
                        y_size = 640
                        img = cv2.resize(img, (x_size, y_size), interpolation=cv2.INTER_AREA)
                        img, parameter, annotate = cs.draw_image(model, device, img, conf / 100, colors, time_JKT,
                                                                 x_size, y_size)
                        st.image(img, caption=caption)

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

        extension_file = st.button('Process',
                                   key='extension_file')

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
            cv2.imwrite(image_name, img_file)

            annotate_name = f'{directory}/annotations/{label_name(st.session_state.counter, 10000)}.txt'

            try:
                df = pd.DataFrame(annotate_file)
                with open(annotate_name, 'a') as f:
                    df1_string = df.to_string(header=False, index=False)
                    f.write(df1_string)
            except:
                df = pd.DataFrame([0, 0, 0, 0],
                                  columns=['id', 'x', 'y', 'w', 'h'])
                with open(annotate_name, 'a') as data:
                    df2_string = df.to_string(header=False, index=False)
                    data.write(df2_string)

            next_photo(path_images_1, func)

        if extension_file:
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
                st.form_submit_button("Upload")
                image_files = [Image.open(io.BytesIO(file.read())) for file in uploaded_files]

            if 'counter' not in st.session_state:
                st.session_state.counter = 0

            tz_JKT = pytz.timezone('Asia/Jakarta')
            time_JKT = datetime.now(tz_JKT).strftime('%d-%m-%Y %H:%M:%S')

            try:
                x_size, y_size = 650, 650

                try:
                    photo = image_files[st.session_state.counter]
                except:
                    st.session_state.counter = 0
                    photo = image_files[st.session_state.counter]

                caption = f'The frame image-{st.session_state.counter} generated at {time_JKT}'
                photo_convert = np.array(photo.convert('RGB'))

                st10, st11 = st.columns(2)

                with st10:
                    st10.write("Original Image")
                    st10.image(cv2.resize(photo_convert, (x_size, y_size), interpolation=cv2.INTER_AREA),
                               caption=caption)
                with st11:
                    st11.write("Detection Image")
                    photo_detect, parameter, annotate = cs.draw_image(model, device, photo_convert, conf / 100, colors,
                                                                      time_JKT, x_size, y_size)
                    st11.image(cv2.resize(photo_detect, (x_size, y_size), interpolation=cv2.INTER_AREA),
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
            except:
                pass
    with tab3:
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        webrtc_ctx = webrtc_streamer(key="WYH",
                                     mode=WebRtcMode.SENDRECV,
                                     rtc_configuration=RTC_CONFIGURATION,
                                     media_stream_constraints={"video": True, "audio": False},
                                     video_frame_callback=cs.recv,
                                     async_processing=True)


def validation(st, **state):
    # Title
    image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
                'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
    st.markdown('<h3 style=\'text-align:center;\'>Validation Result</h3>', unsafe_allow_html=True)

    restriction = state['login']

    if 'login' not in state or not restriction:
        st.warning('Please login with your registered email!')
        return

    path_object = {'General Detection': 'general-detect',
                   'Coal Detection': 'front-coal',
                   'Seam Detection': 'seam-gb',
                   'Core Detection': 'core-logging',
                   'Smart-HSE': 'hse-monitor'}

    kind_object = st.selectbox('Please select the kind of object detection do you want.',
                               ['General Detection',
                                'Coal Detection',
                                'Seam Detection',
                                'Core Detection',
                                'Smart HSE'],
                               key='kind-object-validation-1')

    try:
        def next_photo(path_files, func):
            path_images = [str(path_files + '/' + img_file) for img_file in os.listdir(path_files)]
            path_images.sort()

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

        def delete_photo(path_files, func):
            path_images = [str(path_files + '/' + img_file) for img_file in os.listdir(path_files)]
            path_images.sort()
            photo = path_images[st.session_state.counter]
            text = f'{PATH}/detections/{path_object[kind_object]}/annotations/' + \
                   photo.split("/")[-1].split(".")[0] + '.txt'

            os.remove(photo)
            os.remove(text)

            next_photo(path_files, func)

        path_files = f'{PATH}/detections/{path_object[kind_object]}/images'

        if 'counter' not in st.session_state:
            st.session_state.counter = 0

        path_images = [str(path_files + '/' + img_file) for img_file in os.listdir(path_files)]
        path_images.sort()

        try:
            photo = path_images[st.session_state.counter]
        except:
            st.session_state.counter = 0
            photo = path_images[st.session_state.counter]

        st.image(photo, caption=f'image-{photo.split("/")[-1]}')

        st1, st2, st3, st4, st5 = st.columns(5)

        with st2:
            st2.button('◀️ Back',
                       on_click=next_photo,
                       use_container_width=True,
                       args=([path_files, 'back']),
                       key='back-photo-validation-1')
        with st3:
            st3.button('Delete ♻️',
                       on_click=delete_photo,
                       use_container_width=True,
                       args=([path_files, 'delete']),
                       key='delete-photo-validation-1')
        with st4:
            st4.button('Next ▶️',
                       on_click=next_photo,
                       use_container_width=True,
                       args=([path_files, 'next']),
                       key='next-photo-validation-1')

        btn = st.radio('Do you want to download image in single or all files?',
                       ['Single files', 'All files', 'Not yet'],
                       index=2,
                       key='download-button-2')

        if btn == 'Single files':
            st.success(f'Now, you can download the image-{label_name(st.session_state.counter, 10000)} with annotation '
                       f'in the button bellow.', icon='✅')
            st6, st7 = st.columns(2)

            with st6:
                with open(photo, 'rb') as file:
                    st6.download_button(label='🔗 Image (.png)',
                                        data=file,
                                        use_container_width=True,
                                        file_name=f'{photo.split("/")[-1]}',
                                        mime="image/png",
                                        key='download-image-1')

            with st7:
                annotate_path = f'{PATH}/detections/{path_object[kind_object]}/annotations/' + \
                                photo.split("/")[-1].split(".")[0] + '.txt'

                with open(annotate_path, 'rb') as file:
                    st7.download_button(label='🔗 Annotation (.txt)',
                                        data=file,
                                        use_container_width=True,
                                        file_name=f'{photo.split("/")[-1].split(".")[0]}.txt',
                                        mime="text/plain",
                                        key='download-annotate-1')

        elif btn == 'All files':
            st.success(f'Now, you can download the all images with annotation '
                       f'in the button bellow.', icon='✅')
            path_folder = f'{PATH}/detections/{path_object[kind_object]}'
            name = path_object[kind_object]
            make_zip(path_folder, name)

            with open(f'{path_folder}/{name}.zip', "rb") as fp:
                st.download_button(label="🔗 Download All Files (.zip)",
                                   data=fp,
                                   use_container_width=True,
                                   file_name=f'detection_{name}.zip',
                                   mime="application/zip",
                                   key='download-zip-1')
    except:
        st.error('Please go to the menu Detection first!', icon='❎')


def report(st, **state):
    # Title
    image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
                'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
    st.markdown('<h3 style=\'text-align:center;\'>Messages Report</h3>', unsafe_allow_html=True)

    restriction = state['login']

    if 'login' not in state or not restriction:
        st.warning('Please login with your registered email!')
        return

    placeholder = st.empty()

    with placeholder.form('Message'):
        email = st.text_input('Email')
        text = st.text_area('Messages')
        submit = st.form_submit_button('Send')

    if submit and check_email(email) == 'valid email' or check_email(email) == 'duplicate email':
        placeholder.empty()
        st.success('Before your message will be send, please confirm your messages again!')
        vals = st.write("<form action= 'https://formspree.io/f/xeqdqdon' "
                        "method='POST'>"
                        "<label> Email: <br> <input type='email' name='email' value='" + str(email) +
                        "'style='width:705px; height:50px;'></label>"
                        "<br> <br>"
                        "<label> Message: <br> <textarea name='Messages' value='" + str(text) +
                        "'style='width:705px; height:200px;'></textarea></label>"
                        "<br> <br>"
                        "<button type='submit'>Confirm</button>"
                        "</form>", unsafe_allow_html=True)

        if vals is not None:
            st.success('Your messages has been send successfully!')

    elif submit and check_email(email) == 'invalid email':
        st.success("Your message hasn't been send successfully because email receiver not in list")

    else:
        pass


def account(st, **state):
    # Title
    image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
                'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
    st.markdown('<h3 style=\'text-align:center;\'>Account Setting</h3>', unsafe_allow_html=True)

    restriction = state['login']
    password = state['password']

    if ('login' not in state or not restriction) or ('password' not in state):
        st.warning('Please login with your registered email!')
        return

    placeholder = st.empty()

    st.write('Do you want to edit your account?')
    edited = st.button('Edit')
    state['edit'] = np.invert(edited)

    old_email = state['email']

    with placeholder.form('Account'):
        name_ = state['name'] if 'name' in state else ''
        name = st.text_input('Name', placeholder=name_, disabled=state['edit'])

        username_ = state['username'] if 'username' in state else ''
        username = st.text_input('Username', placeholder=username_, disabled=state['edit'])

        email_ = state['email'] if 'email' in state else ''
        email = st.text_input('Email', placeholder=email_, disabled=state['edit'])

        if edited:
            current_password = st.text_input('Old Password', type='password', disabled=state['edit'])
        else:
            current_password = password

        # current_password_ = state['password'] if 'password' in state else ''
        new_password = st.text_input('New Password', type='password', disabled=state['edit'])

        save = st.form_submit_button('Save')

    if save and current_password == password:
        st.success('Hi ' + name + ', your profile has been update successfully')
        MultiPage.save({'name': name,
                        'username': username,
                        'email': email,
                        'password': new_password,
                        'edit': True})

        replace_json(name, username, old_email, email, new_password)

    elif save and current_password != password:
        st.success(
            'Hi ' + name + ", your profile hasn't been update successfully because your current password doesn't match!")

    elif save and check_email(email) == 'invalid email':
        st.success('Hi ' + name + ", your profile hasn't been update successfully because your email invalid!")

    else:
        pass


def logout(st, **state):
    # Title
    image = Image.open(f'{PATH}/data/images/logo_yeomine.png')

    st.success('Your account has been log out from this app')
    MultiPage.save({'login': False})


app.st = st

app.navbar_name = 'Application Menu'
app.navbar_style = 'VerticalButton'

app.hide_menu = False
app.hide_navigation = True

app.add_app('Landing Page      ', landing_page, initial_page=True)
app.add_app('🔐 Sign Up        ', sign_up)
app.add_app('🔓 Login          ', login)
app.add_app('⚙️ Training       ', training)
app.add_app('📹 Detection      ', detection)
app.add_app('👁‍🗨 Validation    ', validation)
app.add_app('💬 Report         ', report)
app.add_app('👨‍⚖️ Account Setting', account)
app.add_app('🔒 Logout         ', logout)

app.run()
