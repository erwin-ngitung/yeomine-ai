from pptx import Presentation
from pptx.util import Inches, Cm, Pt
from pptx.enum.text import PP_PARAGRAPH_ALIGNMENT as PP
from datetime import datetime as dt
from streamlit import session_state as state
import pytz
import copy

path_object = {'General Detection': 'general-detect',
               'Coal Detection': 'front-coal',
               'Seam Detection': 'seam-gb',
               'Core Detection': 'core-logging',
               'Smart-HSE': 'hse-monitor'}

# if 'PATH' not in state.keys():
#     state['PATH'] = '.'
#
# PATH = state['PATH']
PATH = '.'

# kind_object = state['object-videos']

img_path = f'{PATH}/../datasets/front-coal/images/train/0000.jpg'
logo_path = f'{PATH}/../data/images/logo_yeomine.png'
ppt_template = f'{PATH}/../data/template/format_report-analysis_yeomine.pptx'
# save_ppt = f'{PATH}/../reports/{path_object[kind_object]}'

tz_JKT = pytz.timezone('Asia/Jakarta')
time_JKT = dt.now(tz_JKT).strftime('%d-%m-%Y')

prs = Presentation(ppt_template)

slide_0 = prs.slides[0]
slide_1 = prs.slides[1]

slide_layout = prs.slide_layouts[2]
curr_slide = prs.slides.add_slide(slide_layout)

for shp in slide_1.shapes:
    el = shp.element
    newel = copy.deepcopy(el)
    curr_slide.shapes._spTree.insert_element_before(newel, 'p:extLst')

# Slide 0
line_0 = slide_0.shapes[2].text_frame.paragraphs[0]
line_0.text = f'Date: {time_JKT}'
line_0.alignment = PP.CENTER
line_0.font.name = 'Archive Black'
line_0.font.size = Pt(35)
line_0.font.bold = True

# Slide 1
line_1 = slide_1.shapes[3].text_frame.paragraphs[0]
line_1.text = f'Model Analysis'
line_1.alignment = PP.LEFT
line_1.font.name = 'Archive Black'
line_1.font.size = Pt(50)
line_1.font.bold = True

picture_1 = slide_1.shapes
picture_1.add_picture(logo_path,
                      left=Inches(15),
                      top=Inches(1),
                      width=Inches(3),
                      height=Inches(1))
picture_1.add_picture(img_path,
                      left=Inches(2.5),
                      top=Inches(2),
                      width=Inches(15),
                      height=Inches(9))

# Slide 2
line_2 = curr_slide.shapes[2].text_frame.paragraphs[0]
line_2.text = f'Model Analysis'
line_2.alignment = PP.LEFT
line_2.font.name = 'Archive Black'
line_2.font.size = Pt(50)
line_2.font.bold = True

picture_2 = curr_slide.shapes
picture_2.add_picture(logo_path,
                      left=Inches(15),
                      top=Inches(1),
                      width=Inches(3),
                      height=Inches(1))

picture_2.add_picture(img_path,
                      left=Inches(2.5),
                      top=Inches(2),
                      width=Inches(15),
                      height=Inches(9))

prs.save('../test.pptx')
