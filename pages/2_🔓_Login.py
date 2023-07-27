from streamlit import session_state as state
import streamlit as st
from utils import check_account
from PIL import Image

PATH = state['PATH']

st.snow()
placeholder = st.empty()

with placeholder.form(key='Form Login'):
    image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
    st1, st2, st3 = st.columns(3)

    with st2:
        st.image(image)

    st.markdown('#### Login Yeomine Application')
    email = st.text_input('Email')
    password = st.text_input('Password', type='password')
    submit = st.form_submit_button('Login',
                                   use_container_width=True)

    st.write("Are you ready registered account in this app? If you don't yet, please sign up your account!")

name, username, status = check_account(email, password)

if submit and status == 'register':
    placeholder.empty()
    st.success('Login successful')

    state['name'] = name
    state['username'] = username
    state['email'] = email
    state['password'] = password
    state['edit'] = True
    st.write(state)

    if 'login' in state.keys():
        del state['login']
        state['login'] = True
    else:
        state['login'] = True

elif submit and status == 'wrong password':
    st.error('Login failed because your password is wrong!')

elif submit and status == 'not register':
    st.error("You haven't registered to this app! Please sign up your account!")

else:
    pass

