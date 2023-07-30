import os
from PIL import Image
import cv2
from utils import make_zip, label_name

# Package for Streamlit
from streamlit import session_state as state
import streamlit as st

st.set_page_config(
    page_title="Validation | Yeomine App",
    page_icon="🔎‍",
)

if 'PATH' not in state.keys():
    state['PATH'] = '.'

PATH = state['PATH']

image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
st1, st2, st3 = st.columns(3)

with st2:
    st.image(image)

st.markdown('<svg width=\'705\' height=\'5\'><line x1=\'0\' y1=\'2.5\' x2=\'705\' y2=\'2.5\' stroke=\'black\' '
            'stroke-width=\'4\' fill=\'black\' /></svg>', unsafe_allow_html=True)
st.markdown('<h3 style=\'text-align:center;\'>Validation Result</h3>', unsafe_allow_html=True)

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

    try:
        kind_file = state['kind-file']
        kind_object = state[f'object-{kind_file}']

        def next_photo(path_files, func):
            path_images = [str(path_files + '/' + img_file) for img_file in os.listdir(path_files)]
            path_images.sort()

            if func == 'next':
                state.counter += 1
                if state.counter >= len(path_images):
                    state.counter = 0
            elif func == 'back':
                state.counter -= 1
                if state.counter >= len(path_images):
                    state.counter = 0
                elif state.counter < 0:
                    state.counter = len(path_images) - 1

        def delete_photo(path_files, func):
            path_images = [str(path_files + '/' + img_file) for img_file in os.listdir(path_files)]
            path_images.sort()
            photo = path_images[state.counter]
            text = f'{PATH}/detections/{kind_file}/{path_object[kind_object]}/annotations/' + \
                   photo.split("/")[-1].split(".")[0] + '.txt'

            os.remove(photo)
            os.remove(text)

            next_photo(path_files, func)

        path_files = f'{PATH}/detections/{kind_file}/{path_object[kind_object]}/images'

        if 'counter' not in state:
            state.counter = 0

        path_images = [str(path_files + '/' + img_file) for img_file in os.listdir(path_files)]
        path_images.sort()

        try:
            photo = path_images[state.counter]
            name_photo = photo.split("/")[-1].split(".")[0]
        except (Exception,):
            state.counter = 0
            photo = path_images[state.counter]

        # img_photo = cv2.imread(photo)
        # img_photo = cv2.cvtColor(img_photo, cv2.COLOR_BGR2RGB)

        st.image(photo,
                 channels='RGB',
                 use_column_width='always',
                 caption=f'image-{name_photo}')

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
            st.success(f'Now, you can download the image-{name_photo} with annotation '
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
                annotate_path = f'{PATH}/detections/{kind_file}/{path_object[kind_object]}/annotations/' + \
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
            path_folder = f'{PATH}/detections/{kind_file}/{path_object[kind_object]}'
            name = path_object[kind_object]
            make_zip(path_folder, name)

            with open(f'{path_folder}/{name}.zip', "rb") as fp:
                st.download_button(label="🔗 Download All Files (.zip)",
                                   data=fp,
                                   use_container_width=True,
                                   file_name=f'detection_{name}.zip',
                                   mime="application/zip",
                                   key='download-zip-1')
    except (Exception,):
        st.error('Please go to the menu Detection first!', icon='❎')
