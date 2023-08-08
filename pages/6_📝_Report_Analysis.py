import os
import pandas as pd
from PIL import Image
from streamlit import session_state as state
import streamlit as st
from utils import check_email, computer_vision as cs
import plotly.express as px
from ultralytics import YOLO

st.set_page_config(
    page_title="Report Analysis | Yeomine App",
    page_icon="📝",
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
st.markdown('<h3 style=\'text-align:center;\'>Report Analysis</h3>', unsafe_allow_html=True)

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
        # try:
        kind_file = 'videos'
        kind_object = state['object-videos']
        kind_model = state['model-videos']
        path_folder = f'{PATH}/detections/{kind_file}/{path_object[kind_object]}/annotations'

        model = YOLO(kind_model)

        dataset = pd.DataFrame(columns=['Label', 'X', 'Y', 'Weight', 'Height'])
        for file in os.listdir(path_folder):
            data = pd.read_fwf(f'{path_folder}/{file}',
                               names=['Label', 'X', 'Y', 'Weight', 'Height'])

            dataset = pd.concat([dataset, data])

        dataset.dropna(inplace=True)
        dataset['Label'] = dataset['Label'].astype(int).replace(model.names)
        dataset['X'] = dataset['X'].astype(float)
        dataset['Y'] = dataset['X'].astype(float)
        dataset['Weight'] = dataset['Weight'].astype(float)
        dataset['Height'] = dataset['Height'].astype(float)
        dataset = dataset.reset_index(drop=True)
        dataset['ID'] = cs.count_label(dataset)

        st.table(dataset)

        # except (Exception,):
        #     st.error('Please go to the menu Detection (sub-menu video) first!', icon='❎')

    with tab2:
        try:
            kind_file = 'pictures'
            kind_object = state['object-pictures']
            kind_model = state['model-pictures']
            path_folder = f'{PATH}/detections/{kind_file}/{path_object[kind_object]}/annotations'

        except (Exception,):
            st.error('Please go to the menu Detection (sub-menu picture) first!', icon='❎')
