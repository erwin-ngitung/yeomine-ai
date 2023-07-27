from streamlit import session_state as state
import streamlit as st
from PIL import Image
import shutil
import os

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

st.success('Your account has been log out from this app')

important = ['name', 'username', 'email', 'password']

# Delete all the items in Session state
for key in state.keys():
    if key not in important:
        del [key]

if os.path.exists(f'{PATH}/detections/'):
    shutil.rmtree(f'{PATH}/detections/')

state['login'] = False
