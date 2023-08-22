import os
import pandas as pd
from PIL import Image
from streamlit import session_state as state
import streamlit as st
from utils import check_email, computer_vision as cs, create_ppt as cp
from pptx import Presentation
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
st.markdown('<h3 style=\'text-align:center;\'>Removal OB</h3>', unsafe_allow_html=True)

try:
    restriction = state['login']
except (Exception,):
    state['login'] = False
    restriction = state['login']

if not restriction:
    st.warning('Please login with your registered email!')
else:
    st.write('Yes')
