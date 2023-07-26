from streamlit import session_state as state
import streamlit as st
from pathlib import Path
import pytesseract
import logging

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
# PATH = Path(Path(__file__).resolve()).parent
PATH = '.'
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.write("# Welcome to Yeomine! 👋")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)

state['login'] = False
state['PATH'] = PATH