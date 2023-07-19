import time
import cv2
import os
import numpy as np
import pandas as pd
import shutil

import pandas as pd
from PIL import Image
from utils import check_email, check_account, update_json, replace_json, computer_vision as cs

# Package for Streamlit
import streamlit as st
from streamlit_multipage import MultiPage
from datetime import datetime
import pytz
import pytesseract

# Package for Machine Learning
import torch
from ultralytics import YOLO

import warnings

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

app = MultiPage()


def sign_up(st, **state):
    placeholder = st.empty()

    with placeholder.form('Sign Up'):
        image = Image.open('images/logo_yeomine.png')
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
                        'login': 'True',
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
            image = Image.open('images/logo_yeomine.png')
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
                            'login': 'True',
                            'path_file': 'source/front-coal'})

        elif submit and status == 'wrong password':
            st.error('Login failed because your password is wrong!')

        elif submit and status == 'not register':
            st.error("You haven't registered to this app! Please sign up your account!")

        else:
            pass

    except:
        st.error('Please login with your registered email!')


def train_model(st, **state):
    # Title
    image = Image.open('images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
                'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
    st.markdown('<h3 style=\'text-align:center;\'>Train Custom Model</h3>', unsafe_allow_html=True)

    restriction = state['login']

    if 'login' not in state or restriction == 'False':
        st.warning('Please login with your registered email!')
        return

    tab1, tab2, tab3 = st.tabs(['Train Model', 'Dashboard Model', 'Validating Result'])

    with tab1:
        try:
            kind_object = st.selectbox('Please select the kind of object detection do you want',
                                       ['General Detection', 'Coal Detection', 'Seam Detection', 'Core Detection',
                                        'Smart-HSE'])

            path_object = {'General Detection': 'general-detect',
                           'Coal Detection': 'front-coal',
                           'Seam Detection': 'seam-gb',
                           'Core Detection': 'core-logging',
                           'Smart-HSE': 'hse-monitor'}

            path_file = st.text_input('Please input your path data YAML', 'data/front-coal.yaml')
            list_model = os.listdir(f'weights/petrained-model')
            kind_model = st.selectbox('Please select the petrained model',
                                      list_model)
            st4, st5 = st.columns(2)

            with st4:
                epochs = st.number_input('Number of Epochs', format='%i', value=10, key='epochs')
                imgsz = st.number_input('Number of Image Size', format='%i', value=640, key='imgsz')
                batch = st.number_input('Number of Batch Size', format='%i', value=10, key='batch')

            with st5:
                lr_rate = st.number_input('Number of Learning Rate', format='%f', value=0.05, key='lr_rate')
                momentum = st.number_input('Number of Size Rate', format='%f', value=0.05, key='momentum')
                weight_decay = st.number_input('Number of Weight Decay', format='%f', value=0.05, key='weight_decay')

            next_train = st.radio('Are you sure to train model with the parameter above?',
                                  ['Yes', 'No'], index=1)

            if next_train == 'Yes':
                shutil.rmtree(f'results/{path_object[kind_object]}')

                if torch.cuda.is_available():
                    st.success(
                        f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
                    device = 0
                else:
                    st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
                    device = 'cpu'

                # Load a model
                model = YOLO(
                    f'weights/petrained-model/{kind_model}')  # load a pretrained model (recommended for training)
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

                src = f'results/{path_object[kind_object]}/weights/best.pt'
                dest = f'weights/{path_object[kind_object]}/{path_object[kind_object]}-000.pt'

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
                                  list_visual)

            if visual == 'Summary':
                visual = 'results'
            elif visual == 'Confusion Matrix':
                visual = 'confusion_matrix_normalized'

            st.image(f'results/{path_object[kind_object]}/{visual}.png', caption=f'The image of {visual}')
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
                                  list_visual)

            st.image(f'results/{path_object[kind_object]}/{visual}.jpg', caption=f'The image of {visual}')
        except:
            pass


def detection(st, **state):
    # Title
    image = Image.open('images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
                'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
    st.markdown('<h3 style=\'text-align:center;\'>Detection Model</h3>', unsafe_allow_html=True)

    restriction = state['login']
    path_file = state['path_file']

    if 'login' not in state or restriction == 'False':
        st.warning('Please login with your registered email!')
        return

    kind_object = st.selectbox('Please select the kind of object detection do you want',
                               ['General Detection',
                                'Coal Detection',
                                'Seam Detection',
                                'Core Detection',
                                'Smart HSE'])

    path_object = {'General Detection': 'general-detect',
                   'Coal Detection': 'front-coal',
                   'Seam Detection': 'seam-gb',
                   'Core Detection': 'core-logging',
                   'Smart-HSE': 'hse-monitor'}

    conf = st.slider('Number of Confidence (%)', min_value=0, max_value=100, step=1, value=60)

    st4, st5 = st.columns(2)

    with st4:
        custom = st.radio('Do you want to use custom model that has trained?',
                          ['Yes', 'No'], index=1)
    with st5:
        type_camera = st.radio('Do you want to use Integrated Webcam?',
                               ['Yes', 'No'], index=1)

    st6, st7 = st.columns(2)

    with st6:
        if custom == 'Yes':
            option_model = f'results/{path_object[kind_object]}/weights/best.pt'
            model = YOLO(option_model)
            st.success('The model have successfully loaded!')
        else:
            list_weights = [weight_file for weight_file in os.listdir(f'weights/{path_object[kind_object]}')]
            option_model = st.selectbox('Please select model do you want!',
                                        list_weights)
            model = YOLO(f'weights/{path_object[kind_object]}/{option_model}')

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
                                        list_files)
            source = f'datasets/{path_object[kind_object]}/predict/{sample_video}'
            cap = cv2.VideoCapture(source)

    if torch.cuda.is_available():
        st.success(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
        device = 0
    else:
        st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
        device = 'cpu'

    show_label = st.checkbox('Show label predictions', value=True, key='show-label')
    save_annotate = st.checkbox('Save annotate and images', value=False, key='save-annotate')

    count = 0
    placeholder = st.empty()
    colors = cs.generate_label_colors(model.names)
    data_annotations = pd.DataFrame(columns=['label', 'score', 'x1', 'y1', 'x2', 'y2'])

    try:
        shutil.rmtree(f'detections/{path_object[kind_object]}/images/')
        shutil.rmtree(f'detections/{path_object[kind_object]}/videos/')
        shutil.rmtree(f'detections/{path_object[kind_object]}/annotations/')

        os.makedirs(f'detections/{path_object[kind_object]}/images/')
        os.makedirs(f'detections/{path_object[kind_object]}/videos/')
        os.makedirs(f'detections/{path_object[kind_object]}/annotations/')
    except:
        pass

    # Detection Model
    while cap.isOpened():
        with placeholder.container():
            stop_program = st.checkbox("Do you want to stop this program?", value=False, key=f'stop-program-{count}')

            if not stop_program:
                ret, img = cap.read()

                if ret:
                    tz_JKT = pytz.timezone('Asia/Jakarta')
                    time_JKT = datetime.now(tz_JKT).strftime('%d-%m-%Y %H:%M:%S')
                    caption = f'The frame image-{count} generated at {time_JKT}'

                    img, parameter = cs.draw_image(model, device, img, conf / 100, colors, time_JKT)
                    st.image(img, caption=caption)

                    if save_annotate:
                        path_name = f'detections/{path_object[kind_object]}/images/frame-{count}.png'
                        cv2.imwrite(path_name, img)

                    df = pd.DataFrame(parameter)
                    data_annotations = pd.concat([data_annotations, df], ignore_index=True)

                    if show_label:
                        st.table(df)

                    count += 1
                    time.sleep(0.5)
                else:
                    print('Image is not found')

            else:
                if save_annotate:
                    data_annotations.to_excel(f'detections/{path_object[kind_object]}/annotations/annotate.xlsx',
                                              engine='openpyxl')
                break
    st.success("Your program has been successfully stopped")


def report(st, **state):
    # Title
    image = Image.open('images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
                'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
    st.markdown('<h3 style=\'text-align:center;\'>Messages Report</h3>', unsafe_allow_html=True)

    restriction = state['login']

    if 'login' not in state or restriction == 'False':
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
    image = Image.open('images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
                'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
    st.markdown('<h3 style=\'text-align:center;\'>Account Setting</h3>', unsafe_allow_html=True)

    restriction = state['login']
    password = state['password']

    if ('login' not in state or restriction == 'False') or ('password' not in state):
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
    st.success('Your account has been log out from this app')
    MultiPage.save({'login': 'False'})


app.st = st

app.navbar_name = 'Menu'
app.navbar_style = 'VerticalButton'

app.hide_menu = False
app.hide_navigation = True

app.add_app('Sign Up', sign_up)
app.add_app('Login', login)
app.add_app('Model Training', train_model)
app.add_app('Detection', detection)
app.add_app('Report', report)
app.add_app('Account Setting', account)
app.add_app('Logout', logout)

app.run()
