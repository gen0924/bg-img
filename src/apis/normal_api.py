from flask import Flask, request, jsonify, Response
from src.func.img_handler import show_img
from PIL import Image
import numpy as np
from src.model import predict
from src.func import line_detect
from src.func import detect
from src.func import capture_rect
from skimage import filters, img_as_bool, img_as_uint, morphology
import time
import cv2


# server = app.server
server = Flask(__name__)

model = predict.load_model()


@server.route('/ai/image/v1.0/bg_img_detect', methods=['POST'])
def my_detect():
    start = time.time()
    file = request.files.get('image')
    if file is None:
        return {}
    input_time = request.values.get('input_time')
    if input_time is None:
        input_time = time.time()
    else:
        input_time = int(input_time)
    raw_image_bytes = file.stream

    image = Image.open(raw_image_bytes).convert('RGB')
    src = np.array(image)

    # ENet模型预分割
    img_binary = predict.predict(model, image, predict.color_encoding)
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
    # show_img(rect_img, 0)

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

    gradX = cv2.Sobel(binary_img, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(binary_img, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # show_img(gradient, 0)

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
    # show_img(trans_img, 0)

    result = detect.detect(trans_img, input_time)
    end = time.time()
    print('Running Time : %s Sec ' % (end - start))
    print("================result===============")
    print(result)
    return jsonify(result)
