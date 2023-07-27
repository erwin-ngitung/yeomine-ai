from streamlit import session_state as state
import streamlit as st
from PIL import Image
import numpy as np
from utils import check_email, replace_json

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
st.markdown('<h3 style=\'text-align:center;\'>Account Setting</h3>', unsafe_allow_html=True)

try:
    restriction = state['login']
except (Exception,):
    state['login'] = False
    restriction = state['login']

if not restriction:
    st.warning('Please login with your registered email!')
else:
    placeholder = st.empty()

    st.write('Do you want to edit your account?')
    edited = st.button('Edit')
    state['edit'] = np.invert(edited)

    old_email = state['email']
    password = state['password']

    with placeholder.form('Account'):
        name_ = state['name'] if 'name' in state else ''
        name = st.text_input('Name', placeholder=name_, disabled=state['edit'])

        username_ = state['username'] if 'username' in state else ''
        username = st.text_input('Username', placeholder=username_, disabled=state['edit'])

        email_ = state['email'] if 'email' in state else ''
        email = st.text_input('Email', placeholder=email_, disabled=state['edit'])

        if edited:
            current_password = st.text_input('Old Password', type='password', disabled=state['edit'])
        else:
            current_password = password

        new_password = st.text_input('New Password', type='password', disabled=state['edit'])

        save = st.form_submit_button('Save',
                                     use_container_width=True)

    if save and current_password == password:
        st.success('Hi ' + name + ', your profile has been update successfully')

        del state['name']
        del state['username']
        del state['email']
        del state['password']
        del state['edit']

        state['name'] = name
        state['username'] = username
        state['email'] = email
        state['password'] = password
        state['edit'] = True

        replace_json(name, username, old_email, email, new_password)

    elif save and current_password != password:
        st.success(
            'Hi ' + name + ", your profile doesn't successfully update because your current password doesn't match!")

    elif save and check_email(email) == 'invalid email':
        st.success('Hi ' + name + ", your profile hasn't been update successfully because your email invalid!")

    else:
        pass
