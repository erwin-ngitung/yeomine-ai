import os
import pandas as pd
from PIL import Image
from streamlit import session_state as state
import streamlit as st
from utils import check_email, computer_vision as cs, create_ppt as cp
from pptx import Presentation
from datetime import datetime as dt
import datetime
import pytz
import plotly.express as px
from ultralytics import YOLO
from io import BytesIO
import shutil

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
st.markdown('<h3 style=\'text-align:center;\'>Checking Data</h3>', unsafe_allow_html=True)

try:
    restriction = state['login']
except (Exception,):
    state['login'] = False
    restriction = state['login']

if not restriction:
    st.warning('Please login with your registered email!')
else:
    tz_JKT = pytz.timezone('Asia/Jakarta')
    day_JKT = dt.now(tz_JKT).strftime('%A')
    date_JKT = dt.now(tz_JKT).strftime('%d-%m-%Y')
    time_JKT = dt.now(tz_JKT).strftime('%H:%M:%S')
    all_JKT = dt.now(tz_JKT).strftime('%A, %d-%m-%Y at %H:%M:%S')

    st.markdown(f'<h3 style=\'text-align:center;\'>{all_JKT}</h3>', unsafe_allow_html=True)

    dataset = pd.read_csv('data/dataset/data_monitoring_fleet.csv')

    unit = dataset['unit'].unique()
    day = dataset['days'].unique()
    shift = dataset['shift'].unique()
    cap_dt = dataset['cap_dt'].unique()
    material = dataset['material'].unique()
    front = dataset['front'].unique()
    road = dataset['road'].unique()
    disposal = dataset['disposal'].unique()
    weather = dataset['weather'].unique()
    tot_rain = dataset['tot_rain'].unique()
    working_hour = dataset['working_hour'].unique()
    slippery = dataset['slippery'].unique()

    st1, st2 = st.columns(2)

    with st1:
        data_unit = st.selectbox('Unit',
                                 unit)
        data_cap_dt = st.selectbox('cap_dt',
                                   cap_dt)
        data_road = st.selectbox('road',
                                 road)


    with st2:
        data_shift = st.selectbox('shift',
                                  shift)
        data_material = st.selectbox('material',
                                     material)
        data_disposal = st.selectbox('material',
                                     material)

    # st.table(dataset)
