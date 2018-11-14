import base64
import cv2
from io import BytesIO
from flask import Flask, request, jsonify, Response
from src.func.img_handler import show_img
# from src.apis.normal_api import server
from src.model import predict
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from PIL import Image
from dash.dependencies import Input, Output
from src.logger_setting import setup_logging
from src.func import detect
from src.func import line_detect
from src.model import predict
from src.func import capture_rect
from src import const
from src.func import time_util
from skimage import filters, img_as_bool, img_as_uint, morphology


# import pandas as pd
logger = setup_logging()

server = Flask(__name__)

model = predict.load_model()

application = dash.Dash(url_base_pathname='/ai/image/v1.0/bg_img', server=server)

application.scripts.config.serve_locally = True

application.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='input-image'),

    html.Div(id='output-data-upload')
    # html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'})
])


def parse_contents(contents):
    if contents is None:
        return None
    elif isinstance(contents, list):
        contents = contents[0]
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    decoded = BytesIO(decoded)

    image1 = Image.open(decoded).convert("RGB")
    src = np.array(image1)

    # 通道转换
    img = src[:, :, [2, 1, 0]]

    # keras预分割
    img_binary = predict.predict(model, image1, predict.color_encoding)
    show_img(img_binary, 0)
    gradX = cv2.Sobel(img_binary, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(img_binary, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    im2, contours, hierarchy = cv2.findContours(gradient, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    height = src.shape[0]
    width = src.shape[1]

    std_size = (768, 768)
    src = cv2.resize(src, std_size, interpolation=cv2.INTER_AREA)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    # 亮屏部分二次提取
    rect_img = src[y:y + h, x:x + w]
    show_img(rect_img, 0)

    # 最终ROI提取
    gray_img = rect_img[..., 0]
    gray_img = line_detect.gray_value(gray_img)

    thresh = filters.threshold_otsu(gray_img)
    ret, binary_img = cv2.threshold(gray_img, thresh=thresh - 30, maxval=255, type=cv2.THRESH_BINARY_INV)
    # show_img(binary_img, 0)
    dst = img_as_bool(binary_img)
    dst = morphology.remove_small_objects(dst, min_size=4000, connectivity=1, in_place=True)
    binary_img = img_as_uint(dst)
    show_img(binary_img, 0)
    kernel = np.ones((5, 5), np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    show_img(binary_img, 0)

    gradX = cv2.Sobel(binary_img, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(binary_img, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    show_img(gradient, 0)

    try:
        f_corner_point = capture_rect.find_rect_points(gradient, gradient)
        cp = []
        for i, point in enumerate(f_corner_point):
            cp.append(point[0])
        corner_point = cp
        # print("============corner_point===========")
        # print(corner_point)
        (p1, p2, p3, p4) = capture_rect.re_range(corner_point)
    except Exception as e:
        return Response(response='Unable to Extract the Glucose Image Efficiently! RETAKING PHOTO', status=500)

    # 投影变换(计算实际像素长度)
    dis_width = np.sqrt(np.sum(np.square(p1 - p4)))
    rdis_width = dis_width * (width / 768)
    dis_height = np.sqrt(np.sum(np.square(p1 - p2)))
    rdis_height = dis_height * (height / 768)

    arraySRC = np.array([p1, p2, p3, p4]).astype(np.float32)
    arrayDST = np.array([[0, 0], [255, 0], [255, 255], [0, 255]]).astype(np.float32)
    PerspectiveMatrix = cv2.getPerspectiveTransform(arraySRC, arrayDST)
    trans_img = cv2.warpPerspective(rect_img, PerspectiveMatrix, (256, 256))
    if rdis_height < rdis_width:
        trans_img = np.rot90(trans_img)
    show_img(trans_img, 0)

    bg_res = detect.detect(trans_img)

    return bg_res


def resize_img(contents):
    if contents is None:
        return None
    elif isinstance(contents, list):
        contents = contents[0]
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    decoded = BytesIO(decoded)

    try:
        image1 = Image.open(decoded)
        image1 = np.array(image1)
        result = np.resize(image1, (1024, 1024))
    except Exception as e:
        print(e)
        return [], 0
    result = base64.b64encode(result)
    result = result.decode('utf-8')
    result = content_type + ',' + result
    # print(type(content_type), type(result))

    # result = bytes(content_type, 'utf-8') + bytes(',', 'utf-8') + bytes(result, 'utf-8')
    return result


@application.callback(Output('input-image', 'children'),
              [Input('upload-data', 'contents'),
              # Input('upload-data', 'filename'),
              # Input('upload-data', 'last_modified')
              ])
def show_input(img_content):
    # print('img content', type(img_content))
    # resized_img = resize_img(img_content)
    return html.Img(src=[img_content],
                    style={
                        'width': '60%',
                        'height': '60%',
                        'textAlign': 'center',
                        'margin': '10px'
                        },
                    )

@application.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               # Input('upload-data', 'filename'),
               # Input('upload-data', 'last_modified')
              ])
def update_output(img_content):
    bg_res = parse_contents(img_content)

    if bg_res is None:
        return dcc.Textarea(
            placeholder='请上传一张血糖仪图片',
            value='',
            style={'width': '100%'}
        )

    img_type = bg_res.get('type')
    if img_type == const.BgImageType.NOT_BG:
        return dcc.Textarea(
            placeholder='请重新上传一张血糖仪图片',
            value='',
            style={'width': '100%'}
        )
    result = bg_res.get('data')
    if img_type == const.BgImageType.DAILY:
        start_hour = 0
        end_hour = 24
    else:
        time_start = bg_res.get('startTime')
        end_time = bg_res.get('endTime')

        start_hour = time_util.timestamp_toHour(time_start)
        end_hour = time_util.timestamp_toHour(end_time)

    y = [i.get('value') for i in result]
    if start_hour < end_hour:
        x = np.linspace(start_hour, end_hour, len(result))
    else:
        zero_clock = 24
        x1_num = int(len(result) * ((zero_clock - start_hour) / 8))
        x1 = np.linspace(start_hour, zero_clock, x1_num) - zero_clock
        x2_num = len(result) - x1_num + 1
        x2 = np.linspace(0, end_hour, x2_num)
        x = np.concatenate([x1[:-1], x2])

    logger.info('Detected')
    return dcc.Graph(
        id='basic-interactions',
        figure={
            'data': [
                {
                    'x': x,
                    'y': y,
                    # 'text': ['a', 'b', 'c', 'd'],
                    # 'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                    'name': 'Trace 1',
                    'mode': 'lines+markers',
                    'marker': {'size': 5}
                }
            ],
            'layout': {
                'yaxis': {
                    'range': [0, 20]
                }
            }
        },
        style={'width': '60%'}
    )

application.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
