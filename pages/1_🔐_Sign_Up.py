from streamlit import session_state as state
import streamlit as st
from utils import check_email, update_json
from PIL import Image

PATH = state['PATH']
placeholder = st.empty()

with placeholder.form('Sign Up'):
    image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.warning('Please sign up your account!')

    name = st.text_input('Name: ')
    username = st.text_input('Username: ')
    email = st.text_input('Email')
    password = st.text_input('Password', type='password')

    save = st.form_submit_button('Save',
                                 use_container_width=True)

if save and check_email(email) == 'valid email':
    placeholder.empty()
    st.success('Hello ' + name + ', your profile has been save successfully. Please go to the menu login.')

    state['name'] = name
    state['username'] = username
    state['email'] = email
    state['password'] = password
    state['login'] = False

    update_json(name, username, email, password)

elif save and check_email(email) == 'duplicate email':
    st.success('Hello ' + name + ", your profile hasn't been save successfully because your email same with other!")

elif save and check_email(email) == 'invalid email':
    st.success('Hello ' + name + ", your profile hasn't been save successfully because your email invalid!")
else:
    pass


