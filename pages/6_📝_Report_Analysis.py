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

