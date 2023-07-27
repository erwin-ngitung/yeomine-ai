from PIL import Image
from streamlit import session_state as state
import streamlit as st
from utils import check_email

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
st.markdown('<h3 style=\'text-align:center;\'>Messages Report</h3>', unsafe_allow_html=True)

try:
    restriction = state['login']
except (Exception,):
    state['login'] = False
    restriction = state['login']

if not restriction:
    st.warning('Please login with your registered email!')
else:
    placeholder = st.empty()

    with placeholder.form('Message'):
        email = st.text_input('Email', value=state['email'])
        text = st.text_area('Messages')
        submit = st.form_submit_button('Send',
                                       use_container_width=True)

    if submit:
        placeholder.empty()
        st.success('Before your message will be send, please confirm your messages again!')
        vals = st.write("<form action= 'https://formspree.io/f/xeqdqdon' "
                        "method='POST'>"
                        "<label> Email: <br> <input type='email' name='email' value='" + str(email) +
                        "'style='width:705px; height:50px;'></label>"
                        "<br> <br>"
                        "<label> Message: <br> <textarea name='Messages' value='" + str(text) +
                        "'style='width:705px; height:200px;'></textarea></label>"
                        "<br> <br>"
                        "<button type='submit'>Confirm</button>"
                        "</form>", unsafe_allow_html=True)

        if vals is not None:
            st.success('Your messages has been send successfully!')

    elif submit and check_email(email) == 'invalid email':
        st.success("Your message hasn't been send successfully because email receiver not in list")

    else:
        pass
