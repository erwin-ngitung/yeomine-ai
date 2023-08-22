from streamlit import session_state as state
import streamlit as st
from PIL import Image
from pathlib import Path
import logging

st.set_page_config(
    page_title="Home | Yeomine App",
    page_icon="🏠",
)

PATH = '.'
# PATH = Path(Path(__file__).resolve()).parent
# logger = logging.getLogger(__name__)

state['login'] = False
state['PATH'] = PATH

image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
st1, st2, st3 = st.columns(3)

with st2:
    st.image(image)

st.write("# Welcome to Yeomine! 👋")

st.image('data/images/brochure_1.png')

st.image('data/images/brochure_2.png')

st.markdown(
    """
    ### Want to learn more and purchase it?
    - Check out [Yeomine Website] (https://erwin-ngitung.github.io/yeomine-ai/)
    - Jump into our [Documentation] (https://erwin-ngitung.github.io/yeomine-ai/#gallery)
    - Ask a question in our [Company] (yeomine-digital-services@gmail.com)
    """
)
