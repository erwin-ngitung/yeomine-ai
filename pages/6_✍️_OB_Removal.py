import os
import pandas as pd
from PIL import Image
from streamlit import session_state as state
import streamlit as st
from datetime import datetime as dt
import pytz
from utils import machine_learning as ml
import json

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
    label_days = json.load(open(f'{PATH}/data/dataset/label_days.json', 'rb'))
    label_data = json.load(open(f'{PATH}/data/dataset/label_json.json', 'rb'))

    data_days = label_days[day_JKT]
    unit = dataset['unit'].unique()
    shift = dataset['shift'].unique()
    cap_dt = dataset['cap_dt'].unique()
    material = dataset['material'].unique()
    front = dataset['front'].unique()
    road = dataset['road'].unique()
    disposal = dataset['disposal'].unique()
    weather = dataset['weather'].unique()

    placeholder = st.empty()

    st1, st2 = st.columns(2)

    with st1:
        data_unit = st1.selectbox('Unit Loader',
                                  unit)
        data_cap_dt = st1.selectbox('Capacity Dump Truck',
                                    cap_dt)
        data_road = st1.selectbox('Road',
                                  road)
        data_weather = st1.selectbox('Weather',
                                     weather)
        data_working = st1.number_input('Working Hour')
        data_distance = st1.number_input('Distance')

    with st2:
        data_shift = st2.selectbox('Shift',
                                   shift)
        data_material = st2.selectbox('Material',
                                      material)
        data_front = st2.selectbox('Front',
                                   front)
        data_disposal = st2.selectbox('Disposal',
                                      disposal)
        data_rain = st2.number_input('Total Rain')
        data_slippery = st2.number_input('Slippery')

    process = st.button('Process Data',
                        key='process-data-1',
                        use_container_width=True)

    if process:
        predict = [label_data['unit'][data_unit],
                   int(data_distance),
                   label_data['days'][data_days],
                   label_data['shift'][data_shift],
                   label_data['cap_dt'][data_cap_dt],
                   label_data['material'][data_material],
                   label_data['front'][data_front],
                   label_data['road'][data_road],
                   label_data['disposal'][data_disposal],
                   label_data['weather'][data_weather],
                   data_rain,
                   int(data_working),
                   data_slippery]

        file = f'{PATH}/data/dataset/data_monitoring_fleet.csv'
        predict_prod = ml.random_forest_model(file, predict)

        st.write('Based on the data and machine learning calculation, the removal ob production that we can get is ')
        st.markdown(f'<h3 style=\'text-align:center;\'>{predict_prod[0]} bcm</h3>', unsafe_allow_html=True)

        state['removal_ob'] = {'Unit': data_unit,
                               'Distance': data_distance,
                               'Days': data_days,
                               'Shift': data_shift,
                               'Capacity Dump Truck': data_cap_dt,
                               'Material': data_material,
                               'Front': data_front,
                               'Road': data_road,
                               'Disposal': data_disposal,
                               'Weather': data_weather,
                               'Rain': data_rain,
                               'Working Hours': data_working,
                               'Slippery': data_slippery,
                               "Production": f'{predict_prod[0]} bcm'}
