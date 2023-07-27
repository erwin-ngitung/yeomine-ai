from streamlit import session_state as state
import streamlit as st
import os
import shutil
import torch
from PIL import Image
from ultralytics import YOLO
from utils import make_zip_only, label_name

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
st.markdown('<h3 style=\'text-align:center;\'>Train Custom Model</h3>', unsafe_allow_html=True)

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

    tab1, tab2, tab3, tab4 = st.tabs(['⌚ Training Model',
                                      '📊 Dashboard Model',
                                      '🎭 Validating Model',
                                      '📦 Download Model'])

    with tab1:
        with st.form("form-training", clear_on_submit=True):
            kind_object = st.selectbox('Please select the kind of object detection do you want.',
                                       ['General Detection',
                                        'Coal Detection',
                                        'Seam Detection',
                                        'Core Detection',
                                        'Smart-HSE'],
                                       key='kind-object-training-1')

            list_model = os.listdir(f'{PATH}/weights/petrained-model')
            kind_model = st.selectbox('Please select the petrained model.',
                                      list_model,
                                      key='kind-model-training-1')
            st4, st5 = st.columns(2)

            with st4:
                epochs = st4.number_input('Number of Epochs',
                                          min_value=1,
                                          max_value=200,
                                          step=1,
                                          key='epochs-training-1')
                imgsz = st4.number_input('Size of Image',
                                         min_value=50,
                                         max_value=1500,
                                         step=5,
                                         key='imgsz-training-1')
                batch = st4.number_input('Number of Batch Size',
                                         min_value=1,
                                         max_value=200,
                                         step=1,
                                         key='batch-training-1')

            with st5:
                lr_rate = st5.number_input('Number of Learning Rate',
                                           min_value=0.01,
                                           max_value=1.0,
                                           step=0.01,
                                           key='lr-rate-training-1')
                momentum = st5.number_input('Number of Size Rate',
                                            min_value=0.01,
                                            max_value=1.0,
                                            step=0.01,
                                            key='momentum-training-1')
                weight_decay = st5.number_input('Number of Weight Decay',
                                                min_value=0.01,
                                                max_value=1.0,
                                                step=0.01,
                                                key='weight-decay-training-1')

            list_yaml = os.listdir(f'{PATH}/data-yaml/{path_object[kind_object]}')
            path_yaml = st.selectbox('Please select your data YAML.',
                                     list_yaml,
                                     key='data-yaml-1')
            next_train = st.form_submit_button("Process",
                                               use_container_width=True)

        if next_train:
            if torch.cuda.is_available():
                st.success(
                    f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
                device = 0
            else:
                st.success(f"Setup complete. Using torch {torch.__version__} (CPU)")
                device = 'cpu'

            shutil.rmtree(f'{PATH}/results/{path_object[kind_object]}')

            # Load a model
            model = YOLO(f'{PATH}/weights/petrained-model/{kind_model}')
            model.train(data=f'{PATH}/data-yaml/{path_object[kind_object]}/{path_yaml}',
                        device=device,
                        epochs=int(epochs),
                        batch=int(batch),
                        imgsz=int(imgsz),
                        lrf=lr_rate,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        project='results',
                        name=path_object[kind_object])

            num_weights = len(os.listdir(f'{PATH}/weights/{path_object[kind_object]}'))
            src = f'{PATH}/results/{path_object[kind_object]}/weights/best.pt'
            dest = f'{PATH}/weights/{path_object[kind_object]}/{path_object[kind_object]}-' \
                   f'{label_name(num_weights, 10000)}.pt'

            shutil.copyfile(src, dest)

            st.success('The model have been successfully saved. Now you can download model in the button bellow',
                       icon='✅')

            path_folder = f'{PATH}/datasets/{path_object[kind_object]}/weights'
            name = f'{path_object[kind_object]}-{label_name(num_weights, 10000)}'
            make_zip_only(path_folder, src, name)

            with open(f'{path_folder}/{name}.zip', "rb") as fp:
                st.download_button(label="🔗 Download Weights Model (.zip)",
                                   data=fp,
                                   use_container_width=True,
                                   file_name=f'weight_{name}.zip',
                                   mime="application/zip",
                                   key='download-zip-1')

    with tab2:
        try:
            list_visual = ['Confusion Matrix',
                           'F1_curve',
                           'P_curve',
                           'PR_curve',
                           'R_curve',
                           'Summary']

            visual = st.selectbox('Please choose the curve of training model',
                                  list_visual,
                                  key='visual-training-1')

            if visual == 'Summary':
                visual = 'results'
            elif visual == 'Confusion Matrix':
                visual = 'confusion_matrix_normalized'

            st.image(f'{PATH}/results/{path_object[kind_object]}/{visual}.png',
                     use_column_width='always',
                     caption=f'The image of {visual}')
        except (Exception,):
            st.error('Please measure that you have trained model in the sub-menu training model.')

    with tab3:
        try:
            list_visual = ['labels',
                           'train_batch0',
                           'train_batch1',
                           'train_batch2',
                           'val_batch0_labels',
                           'val_batch0_pred']

            visual = st.selectbox('Please choose the validation image!',
                                  list_visual,
                                  key='visual-training-2')

            st.image(f'{PATH}/results/{path_object[kind_object]}/{visual}.jpg',
                     use_column_width='always',
                     caption=f'The image of {visual}')
        except (Exception,):
            st.error('Please measure that you have trained model in the sub-menu training model.')

    with tab4:
        try:
            kind_object = st.selectbox('Please select the kind of object detection that you want.',
                                       ['General Detection',
                                        'Coal Detection',
                                        'Seam Detection',
                                        'Core Detection',
                                        'Smart-HSE'],
                                       key='kind-object-training-2')

            list_weights = [weight_file for weight_file in os.listdir(f'{PATH}/weights/{path_object[kind_object]}')]
            option_model = st.selectbox('Please select model do you want.',
                                        list_weights,
                                        key='option-model-detection-1')

            path_folder = f'{PATH}/datasets/{path_object[kind_object]}/weights'
            src = f'{PATH}/weights/{path_object[kind_object]}/{option_model}'
            name = f'{option_model.split(".")[0]}'
            make_zip_only(path_folder, src, name)

            with open(f'{path_folder}/{name}.zip', "rb") as fp:
                st.download_button(label="🔗 Download Weights Model (.zip)",
                                   data=fp,
                                   use_container_width=True,
                                   file_name=f'weight_{name}.zip',
                                   mime="application/zip",
                                   key='download-zip-2')
        except (Exception,):
            st.error('Please measure that you have trained model in the sub-menu training model.')
