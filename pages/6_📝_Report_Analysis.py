import os
import pandas as pd
from PIL import Image
from streamlit import session_state as state
import streamlit as st
from utils import check_email, computer_vision as cs, create_ppt as cp
from pptx import Presentation
import plotly.express as px
from ultralytics import YOLO
from io import BytesIO

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
        # try:
        kind_file = 'videos'
        kind_object = state['object-videos']
        kind_model = state['model-videos']
        path_folder = f'{PATH}/detections/{kind_file}/{path_object[kind_object]}/annotations'

        model = YOLO(kind_model)

        dataset = cs.converter_dataset(path_folder, model)

        st.markdown('<h4 style=\'text-align:center;\'>Coordinate Object</h4>', unsafe_allow_html=True)

        fig1 = px.scatter(dataset,
                          x='X',
                          y='Y',
                          color='ID',
                          range_x=[0, 1],
                          range_y=[1, 0],
                          hover_name='Label')

        st.plotly_chart(fig1, theme='streamlit', use_container_width=True)
        fig1.write_image(f'{PATH}/reports/{path_object[kind_object]}/coordinate-object.png')

        st.markdown('<h4 style=\'text-align:center;\'>Graph Count Label</h4>', unsafe_allow_html=True)

        data_label = []
        data_count = []

        for label in dataset['Label'].unique():
            dataset_init = dataset[dataset['Label'] == label]

            data_label.append(label)
            data_count.append(dataset_init['ID'].nunique())

        dataset_true = pd.DataFrame(columns=['Label', 'Count'])
        dataset_true['Label'] = data_label
        dataset_true['Count'] = data_count

        fig2 = px.histogram(dataset_true,
                            x="Label",
                            y='Count')

        st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
        fig2.write_image(f'{PATH}/reports/{path_object[kind_object]}/graph-count-object.png')

        # st.markdown('<h4 style=\'text-align:center;\'>Object Dataset</h4>', unsafe_allow_html=True)
        # st.table(dataset)

        # Download Button
        path_model_accuracy = f'{PATH}/reports/{path_object[kind_object]}'
        path_model_validation = f'{PATH}/results/{path_object[kind_object]}'
        ppt_template = f'{PATH}/data/template/format_model-analysis_yeomine.pptx'

        prs1 = cp.model_analysis(path_model_validation, ppt_template)

        model_output1 = BytesIO()
        prs1.save(model_output1)

        prs2 = cp.report_analysis(path_model_accuracy, ppt_template)

        model_output2 = BytesIO()
        prs2.save(model_output2)

        st1, st2 = st.columns(2)

        with st1:
            st1.download_button(label='🔗 Download Report Model',
                                data=model_output1.getvalue(),
                                file_name='report-model.pptx',
                                use_container_width=True)
        with st2:
            st2.download_button(label='🔗 Download Report Analysis',
                                data=model_output2.getvalue(),
                                file_name='report-analysis.pptx',
                                use_container_width=True)

        # except (Exception,):
        #     st.error('Please go to the menu Detection (sub-menu video) first!', icon='❎')

    with tab2:
        try:
            kind_file = 'pictures'
            kind_object = state['object-pictures']
            kind_model = state['model-pictures']
            path_folder = f'{PATH}/detections/{kind_file}/{path_object[kind_object]}/annotations'

            model = YOLO(kind_model)

            dataset = pd.DataFrame(columns=['Label', 'X', 'Y', 'Weight', 'Height'])
            
            for file in os.listdir(path_folder):
                data = pd.read_fwf(f'{path_folder}/{file}',
                                   names=['Label', 'X', 'Y', 'Weight', 'Height'])

                dataset = pd.concat([dataset, data])

            dataset.dropna(inplace=True)
            dataset['Label'] = dataset['Label'].astype(int).replace(model.names)
            dataset['X'] = dataset['X'].astype(float)
            dataset['Y'] = dataset['X'].astype(float)
            dataset['Weight'] = dataset['Weight'].astype(float)
            dataset['Height'] = dataset['Height'].astype(float)
            dataset = dataset.reset_index(drop=True)
            dataset['ID'] = cs.count_label(dataset)

            st.table(dataset)

        except (Exception,):
            st.error('Please go to the menu Detection (sub-menu picture) first!', icon='❎')
