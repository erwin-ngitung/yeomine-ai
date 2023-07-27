from streamlit import session_state as state
import streamlit as st
from PIL import Image
from pathlib import Path
import logging

PATH = '.'
# PATH = Path(Path(__file__).resolve()).parent
# logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Home | Yeomine App",
    page_icon="🏠",
)

image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
st1, st2, st3 = st.columns(3)

with st2:
    st.image(image)

st.write("# Welcome to Yeomine! 👋")

st.markdown(
    """
    Yeomine is a product that is built by a web and desktop application using python language as backend and streamlit
     as framework. This is integrated with computer vision technology using YoloV8 Model that is developed 
     with thousands of actual and valid open coal mining data.
    ### Want to learn more and purchase it?
    - Check out [Yeomine Landing Page] (https://erwin-ngitung.github.io/yeomine-ai/)
    - Jump into our [documentation] (https://erwin-ngitung.github.io/yeomine-ai/#gallery)
    - Ask a question in our [company] (yeomine-digital-services@gmail.com)
"""
)

state['login'] = False
state['PATH'] = PATH
