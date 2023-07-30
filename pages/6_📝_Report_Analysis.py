from PIL import Image
from streamlit import session_state as state
import streamlit as st
from utils import check_email

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
        try:
            kind_object = state['object-videos']
            path_folder = f'{PATH}/detections/videos/{path_object[kind_object]}/annotations'

        except (Exception,):
            st.error('Please go to the menu Detection (sub menu Video) first!', icon='❎')

    with tab2:
        try:
            kind_object = state['object-pictures']
            path_folder = f'{PATH}/detections/pictures/{path_object[kind_object]}/annotations'

        except (Exception,):
            st.error('Please go to the menu Detection (sub-menu Picture) first!', icon='❎')

