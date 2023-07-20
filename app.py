import time
import cv2
import os
import io
import numpy as np
import shutil
import pandas as pd
from PIL import Image
from utils import make_folder, make_folder_only, label_name, \
    check_email, check_account, update_json, replace_json, computer_vision as cs

# Package for Streamlit
import streamlit as st
from streamlit_multipage import MultiPage
from datetime import datetime
import pytz
import pytesseract

# Package for Machine Learning
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
PATH = Path(Path(__file__).resolve()).parent
logger = logging.getLogger(__name__)

app = MultiPage()


def sign_up(st, **state):
    placeholder = st.empty()

    with placeholder.form('Sign Up'):
        image = Image.open(f'{PATH}/images/logo_yeomine.png')
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
            image = Image.open(f'{PATH}/images/logo_yeomine.png')
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
    image = Image.open(f'{PATH}/images/logo_yeomine.png')
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

    tab1, tab2, tab3 = st.tabs(['Train Model', 'Dashboard Model', 'Validating Result'])

    with tab1:
        try:
            kind_object = st.selectbox('Please select the kind of object detection do you want',
                                       ['General Detection',
                                        'Coal Detection',
                                        'Seam Detection',
                                        'Core Detection',
                                        'Smart-HSE'],
                                       key='kind-object-training-1')

            path_object = {'General Detection': 'general-detect',
                           'Coal Detection': 'front-coal',
                           'Seam Detection': 'seam-gb',
                           'Core Detection': 'core-logging',
                           'Smart-HSE': 'hse-monitor'}

            path_file = st.text_input('Please input your path data YAML', 'data/front-coal.yaml')
            list_model = os.listdir(f'{PATH}/weights/petrained-model')
            kind_model = st.selectbox('Please select the petrained model',
                                      list_model,
                                      key='kind-model-training-1')
            st4, st5 = st.columns(2)

            with st4:
                epochs = st.number_input('Number of Epochs',
                                         format='%i',
                                         value=10,
                                         key='epochs-training-1')
                imgsz = st.number_input('Number of Image Size',
                                        format='%i',
                                        value=640,
                                        key='imgsz-training-1')
                batch = st.number_input('Number of Batch Size',
                                        format='%i',
                                        value=10,
                                        key='batch-training-1')

            with st5:
                lr_rate = st.number_input('Number of Learning Rate',
                                          format='%f',
                                          value=0.05,
                                          key='lr-rate-training-1')
                momentum = st.number_input('Number of Size Rate',
                                           format='%f',
                                           value=0.05,
                                           key='momentum-training-1')
                weight_decay = st.number_input('Number of Weight Decay',
                                               format='%f',
                                               value=0.05,
                                               key='weight-decay-training-1')

            next_train = st.radio('Are you sure to train model with the parameter above?',
                                  ['Yes', 'No'],
                                  index=1,
                                  key='next-train-training-1')

            if next_train == 'Yes':
                # shutil.rmtree(f'{PATH}/results/{path_object[kind_object]}')

                if torch.cuda.is_available():
                    st.success(
                        f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
                    device = 0
                else:
                    st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
                    device = 'cpu'

                # Load a model
                model = YOLO(
                    f'{PATH}/weights/petrained-model/{kind_model}')
                model.train(data=path_file,
                            device=device,
                            epochs=int(epochs),
                            batch=int(batch),
                            imgsz=int(imgsz),
                            lrf=lr_rate,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            project='results',
                            name=path_object[kind_object])

                src = f'{PATH}/results/{path_object[kind_object]}/weights/best.pt'
                dest = f'{PATH}/weights/{path_object[kind_object]}/{path_object[kind_object]}-000.pt'

                shutil.copyfile(src, dest)

                st.success('The model have been successfully saved!', icon='✅')
        except:
            with st.spinner('Wait a moment..'):
                time.sleep(100)

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
            pass

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
            pass


def detection(st, **state):
    # Title
    image = Image.open(f'{PATH}/images/logo_yeomine.png')
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

    tab1, tab2, tab3 = st.tabs(['Dataset by Admin', 'Upload File', 'Streaming'])

    with tab1:
        kind_object = st.selectbox('Please select the kind of object detection do you want',
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
                         value=60,
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
            type_camera = st.radio('Do you want to use Integrated Webcam?',
                                   ['Yes', 'No'],
                                   index=1,
                                   key='camera-detection-1')

        st6, st7 = st.columns(2)

        with st6:
            if custom == 'Yes':
                option_model = f'{PATH}/results/{path_object[kind_object]}/weights/best.pt'
                model = YOLO(option_model)
                st.success('The model have successfully loaded!')
            else:
                list_weights = [weight_file for weight_file in os.listdir(f'weights/{path_object[kind_object]}')]
                option_model = st.selectbox('Please select model do you want!',
                                            list_weights,
                                            key='option-model-detection-1')
                model = YOLO(f'{PATH}/weights/{path_object[kind_object]}/{option_model}')

        with st7:
            if type_camera == 'Yes':
                source = st.text_input('Please input your Webcam link', 'Auto')
                if source == 'Auto':
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoStream(source).start()
            else:
                list_files = [file for file in os.listdir(f'datasets/{path_object[kind_object]}/predict')]
                sample_video = st.selectbox('Please select sample video do you want',
                                            list_files,
                                            key='sample-video-detection-1')
                source = f'{PATH}/datasets/{path_object[kind_object]}/predict/{sample_video}'
                cap = cv2.VideoCapture(source)

        if torch.cuda.is_available():
            st.success(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
            device = 0
        else:
            st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
            device = 'cpu'

        next_detect = st.radio('Are you sure to detect image/video using parameter above?',
                               ['Yes', 'No'],
                               index=1,
                               key='next-detect-detection-1')

        if next_detect == 'Yes':
            st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' '
                        'stroke=\'black\''
                        'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)

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
                        caption = f'The frame image-{count} generated at {time_JKT}'

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
                            np.savetxt(name_annotate, df2.values, fmt='%.2f')

                        count += 1
                        time.sleep(0.5)

                    else:
                        st.error('Image is not found')

            st.success('Your all images have successfully saved')

    with tab2:
        kind_object = st.selectbox('Please select the kind of object detection do you want',
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
                         value=60,
                         key='confidence-detection-2')

        st8, st9 = st.columns(2)

        with st8:
            custom = st.radio('Do you want to use custom model that has trained?',
                              ['Yes', 'No'],
                              index=1,
                              key='custom-detection-2')
        with st9:
            extension_file = st.radio('What is the kind of file that you want to upload?',
                                      ['Image', 'Video'],
                                      index=0,
                                      key='extension-file-detection-1')

        if custom == 'Yes':
            option_model = f'{PATH}/results/{path_object[kind_object]}/weights/best.pt'
            model = YOLO(option_model)
            st.success('The model have successfully loaded!')
        else:
            list_weights = [weight_file for weight_file in os.listdir(f'weights/{path_object[kind_object]}')]
            option_model = st.selectbox('Please select model do you want!',
                                        list_weights,
                                        key='select-model-detection-2')
            model = YOLO(f'{PATH}/weights/{path_object[kind_object]}/{option_model}')

        if torch.cuda.is_available():
            st.success(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
            device = 0
        else:
            st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
            device = 'cpu'

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

        def save_photo(path_images, func, img_file, annotate_file):
            directory = f'{PATH}/detections/custom-data/{path_object[kind_object]}'
            make_folder_only(directory)

            num_img = len(os.listdir(f'{directory}/images'))

            image_name = f'{directory}/images/{label_name(num_img, 10000)}.png'
            cv2.imwrite(image_name, img_file)

            annotate_name = f'{directory}/annotations/{label_name(num_img, 10000)}.txt'
            try:
                df = pd.DataFrame(annotate_file)
                np.savetxt(annotate_name, df.values, fmt='%.2f')
            except:
                df = pd.DataFrame([0, 0, 0, 0],
                                  columns=['id', 'x', 'y', 'w', 'h'])
                np.savetxt(annotate_name, df.values, fmt='%.2f')

            next_photo(path_images, func)

        if extension_file == 'Image':
            st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' '
                        'stroke=\'black\''
                        'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)

            uploaded_files = st.file_uploader("Upload your image",
                                              type=['jpg', 'jpeg', 'png'],
                                              accept_multiple_files=True)

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
                    st13.button("Back Image ⏭️",
                                on_click=next_photo,
                                args=([image_files, 'back']),
                                key='back-photo-detection-1')
                with st14:
                    btn = st14.button('Save Image ⏭️',
                                      on_click=save_photo,
                                      args=([image_files, 'save', photo_detect, annotate]),
                                      key='save-photo-detection-1')

                with st15:
                    st15.button('Next Image ⏭️',
                                on_click=next_photo,
                                args=([image_files, 'next']),
                                key='next-photo-detection-1')

                if btn:
                    st.success('Now, you can download image with annotation in the button bellow')

                    st17, st18, st19, st20, st21 = st.columns(5)

                    with st18:
                        path_images = f'{PATH}/detections/custom-data/{path_object[kind_object]}/images'
                        num_img = len(os.listdir(path_images))
                        image_name = f'{path_images}/{label_name(num_img - 1, 10000)}.png'

                        with open(image_name, 'rb') as file:
                            st.download_button(label='Image (.png)',
                                               data=file,
                                               file_name=f'{label_name(num_img - 1, 10000)}.png',
                                               mime="image/png")

                    with st20:
                        path_annotate = f'{PATH}/detections/custom-data/{path_object[kind_object]}/annotations'
                        num_annotate = len(os.listdir(path_annotate))
                        annotate_name = f'{path_annotate}/{label_name(num_annotate- 1, 10000)}.txt'

                        with open(annotate_name, 'rb') as file:
                            st.download_button(label='Text (.txt)',
                                               data=file,
                                               file_name=f'{label_name(num_annotate - 1, 10000)}.txt',
                                               mime="text/plain")

            except:
                st.error('Please upload your images or video first!')
    with tab3:
        st.write('Coming Soon!')


def validation(st, **state):
    # Title
    image = Image.open(f'{PATH}/images/logo_yeomine.png')
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

    kind_object = st.selectbox('Please select the kind of object detection do you want',
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
            st2.button("Back Image ⏭️",
                       on_click=next_photo,
                       args=([path_files, 'back']),
                       key='back-photo-validation-1')
        with st3:
            st3.button("Delete Image ⏭️",
                       on_click=delete_photo,
                       args=([path_files, 'delete']),
                       key='delete-photo-validation-1')
        with st4:
            st4.button("Next Image ⏭️",
                       on_click=next_photo,
                       args=([path_files, 'next']),
                       key='next-photo-validation-1')

        st.success('Now, you can download image with annotation in the button bellow')

        st6, st7, st8, st9, st10 = st.columns(5)

        with st7:
            with open(photo, 'rb') as file:
                st.download_button(label='Image (.png)',
                                   data=file,
                                   file_name=f'{photo.split("/")[-1]}',
                                   mime="image/png")

        with st9:
            annotate_path = f'{PATH}/detections/{path_object[kind_object]}/annotations/' + \
                            photo.split("/")[-1].split(".")[0] + '.txt'
            with open(annotate_path, 'rb') as file:
                st.download_button(label='Text (.txt)',
                                   data=file,
                                   file_name=f'{photo.split("/")[-1].split(".")[0]}.txt',
                                   mime="text/plain")
    except:
        st.error('Please go to the menu Detection first!')



def report(st, **state):
    # Title
    image = Image.open(f'{PATH}/images/logo_yeomine.png')
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
    image = Image.open(f'{PATH}/images/logo_yeomine.png')
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
    image = Image.open(f'{PATH}/images/logo_yeomine.png')

    st.success('Your account has been log out from this app')
    MultiPage.save({'login': False})


app.st = st

app.navbar_name = 'Menu'
app.navbar_style = 'VerticalButton'

app.hide_menu = False
app.hide_navigation = True

app.add_app('Sign Up', sign_up)
app.add_app('Login', login)
app.add_app('Training', training)
app.add_app('Detection', detection)
app.add_app('Validation', validation)
app.add_app('Report', report)
app.add_app('Account Setting', account)
app.add_app('Logout', logout)

app.run()
