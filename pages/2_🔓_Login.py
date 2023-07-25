from streamlit import session_state as state
import streamlit as st
from utils import check_account
from PIL import Image

PATH = state['PATH']

st.snow()
# Create an empty container
placeholder = st.empty()

try:
    # Insert a form in the container
    with placeholder.form('login'):
        image = Image.open(f'{PATH}/data/images/logo_yeomine.png')
        st1, st2, st3 = st.columns(3)

        with st2:
            st.image(image)

        st.markdown('#### Login Yeomine Application')
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        submit = st.form_submit_button('Login')

        st.write("Are you ready registered account in this app? If you don't yet, please sign up your account!")

        name, username, status = check_account(email, password)

    if submit and status == 'register':
        # If the form is submitted and the email and password are correct,
        # clear the form/container and display a success message
        placeholder.empty()
        st.success('Login successful')

        state['name'] = name
        state['username'] = username
        state['email'] = email
        state['password'] = password
        state['login'] = True
        state['edit'] = True

    elif submit and status == 'wrong password':
        st.error('Login failed because your password is wrong!')

    elif submit and status == 'not register':
        st.error("You haven't registered to this app! Please sign up your account!")

    else:
        pass

except:
    st.error('Please login with your registered email!')
