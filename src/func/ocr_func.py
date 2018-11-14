from PIL import Image
from io import BytesIO
import requests
import base64
from collections import Counter
import re
from flask import Response


def ocr_find_time_api(rect_img):
    default_start = 8
    default_location_start = 22
    default_location_end = 191
    res = ocr_detect(rect_img)
    words = res.get('words_result')
    # print("==========OCR_Result==========")
    # print(words)
    time_start = []
    time_location_start = []
    time_location_mid = []
    time_location_end = []

    for w in words:
        left = w.get('location').get('left')
        width = w.get('location').get('width')
        w_word = w.get('words')

        if left < 40:   # 第一个时间点
            location = left + width / 2
            time_location_start.append(int(location))
            t_start = time_hr(w_word)
            if t_start is not None:
                time_start.append(t_start)
        elif 40 <= left < 130:
            location = left + width / 2
            time_location_mid.append(int(location))
            t_start = time_hr(w_word)
            if t_start is not None:
                if 0 <= t_start < 4:
                    time_start.append(t_start + 20)
                if t_start >= 4:
                    time_start.append(t_start - 4)
        else:
            location = left + width / 2
            time_location_end.append(int(location))
            t_start = time_hr(w_word)
            if t_start is not None:
                if 0 <= t_start < 8:
                    time_start.append(t_start + 16)
                if t_start >= 8:
                    time_start.append(t_start - 8)

    if not time_start:
        return Response(response='OCR Failed! RETAKING PHOTO', status=500)
        # return default_start, default_location_start, default_location_end

    if time_location_start and time_location_end:
        location_start = time_location_start[0]
        location_end = time_location_end[0]

    if time_location_start and (not time_location_end):
        location_start = time_location_start[0]
        if time_location_mid:
            location_end = location_start + time_location_mid[0]
        if not time_location_mid:
            location_end = default_location_end

    if (not time_location_start) and time_location_end:
        location_end = time_location_end[0]
        if time_location_mid:
            location_start = location_end - time_location_mid[0]
        if not time_location_mid:
            location_start = default_location_start
    if location_start < 40:
        location_start = location_start
    else:
        location_start = default_location_start

    if location_end > 180:
        location_end = location_end
    else:
        location_end = default_location_end

    t_s = Counter(time_start).most_common(1)[0][0]
    # print("========起始时间==========")
    # print(t_s)

    return t_s, location_start, location_end


def ocr_find_time_api1(rect_img):
    # n = 230
    # m = 70
    default_start = 8
    # patch = rect_img[n:256, 0:256]
    # t0 = time.time()
    # file_name = '{}.jpg'.format(str(t0))
    # cv2.imwrite(file_name, patch)
    # return 8
    res = ocr_detect(rect_img)
    # res = {'words_result': []}
    # print(res)
    # show_img(patch, 0)
    words = res.get('words_result')
    time_start = []
    for w in words:
        left = w.get('location').get('left')
        w_word = w.get('words')
        if 0 < left < 40:   # 第一个时间点
            t_start = time_hr(w_word)
            if t_start is not None:
                time_start.append(t_start)
        elif 40 <= left < 130:
            t_start = time_hr(w_word)
            if t_start is not None:
                time_start.append(t_start - 4)
        else:
            t_start = time_hr(w_word)
            if t_start is not None:
                time_start.append(t_start - 8)

    if not time_start:
        return default_start

    t_s = Counter(time_start).most_common(1)[0][0]
    # print("=========t_s==========")
    # print(t_s)
    return t_s


def ocr_detect(img):
    img = Image.fromarray(img)
    access_token = '24.2c6f89853cea007a44148bffa601943b.2592000.1543557595.282335-14386924'
    # access_token = '24.b73d7aa3bf90560b8edf5f401a3f2c01.2592000.1539144983.282335-11187721'
    # access_token = '24.db9f9c5f910aff9a9cb745ca4e2474af.2592000.1536370174.282335-11187721'
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general?access_token={}'.format(access_token)

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    image_bytes_for_tensor = BytesIO()
    img.save(image_bytes_for_tensor, 'JPEG')
    img_str = base64.b64encode(image_bytes_for_tensor.getvalue())
    data = {'image': img_str}
    res = requests.post(url, data, headers=headers)
    return res.json()


def time_hr(words):
    time_pattern1 = '(\d\d:00)'
    time_pattern2 = '(\d\d.00)'
    res1 = re.findall(time_pattern1, words)
    res2 = re.findall(time_pattern2, words)
    if not res1:
        if res2:
            res = re.split("[..a]", words)[0]
        if not res2:
            return None
    res = re.split("[:.a]", words)[0]
    if not res.isdigit():
        return None
    res = int(res)
    return res
