from src.func import ocr_func
from src.func import line_detect
# from src.func.img_handler import show_img
from src.model.bg_result import BgResult
from src.func import image_classify
import operator
from src import const
import cv2
import numpy as np
from skimage import filters


def detect(img, input_time=None):
    res = BgResult(input_time=input_time)

    # step 1: 血糖图像类型识别
    img_rect, img_type = image_classify.decide_orientation(img)
    print(img_type)
    # step 2: 定量分析—血糖曲线检测
    if img_type == const.BgImageType.NORMAL:
        src_height, src_width, _ = np.shape(img_rect)
        src = img_rect[int(src_height / 2) + 10: int(src_height) - 15, 10: int(src_width) - 20]
        height, width, _ = np.shape(src)

        thresh = filters.threshold_otsu(src[..., 2])
        ret, binary_img = cv2.threshold(src[..., 2], thresh=thresh, maxval=255, type=cv2.THRESH_BINARY_INV)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded = cv2.erode(binary_img, element)
        # show_img(eroded, 0)
        dilated = cv2.dilate(eroded, element)

        gradX = cv2.Sobel(dilated, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradY = cv2.Sobel(dilated, ddepth=cv2.CV_32F, dx=0, dy=1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        # show_img(gradient, 0)

        # 直线检测—检测标准band
        std_lower, std_upper, std_theta = line_detect.normal_filter_band(gradient)
        # print("=============色带信息=========")
        # print(std_lower)
        # print(std_upper)
        # print(std_theta)

        # 时间起点终点识别—OCR
        ocr_img = img_rect[int(src_height) - 20: int(src_height), 10: int(src_width) - 20]
        # show_img(ocr_img, 0)
        time_start, location_start, location_end = ocr_func.ocr_find_time_api(ocr_img)
        # print("==========OCR==========")
        # print(time_start)
        # print(location_start)
        # print(location_end)

        # 血糖曲线检测
        line_dict = line_detect.find_bg_line_normal(src, location_start, location_end)
        # print("===========血糖曲线坐标=============")
        # print(line_dict)

        # 血糖值计算预测
        result = {}
        for index, (x, y) in line_dict.items():
            bg_value = line_detect.get_bg_value_normal(x, y, std_upper=std_upper,
                                                std_lower=std_lower,
                                                std_theta=std_theta)

            result[index] = bg_value
        result = sorted(result.items(), key=operator.itemgetter(0), reverse=False)
        # print("==========血糖值==========")
        # print(result)
        # result
        res.type = const.BgImageType.NORMAL
        res.add_start_time(time_start)
        res.add_raw_data(result, time_start)
        res = res.to_dict(time_start)
        print("===============血糖预测值===============")
        print(res)
        return res

    elif img_type == const.BgImageType.DAILY:
        height, width, _ = np.shape(img_rect)
        src = img_rect[int(height / 2) - 60: int(height) - 90, 30: int(width) - 15]
        height, width, _ = np.shape(src)
        # show_img(src, 0)

        thresh = filters.threshold_otsu(src[..., 2])
        ret, binary_img = cv2.threshold(src[..., 2], thresh=thresh, maxval=255, type=cv2.THRESH_BINARY_INV)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded = cv2.erode(binary_img, element)
        # show_img(eroded, 0)
        dilated = cv2.dilate(eroded, element)

        gradX = cv2.Sobel(dilated, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradY = cv2.Sobel(dilated, ddepth=cv2.CV_32F, dx=0, dy=1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        # show_img(gradient, 0)

        # 直线检测—标准色带band
        std_lower, std_upper, std_theta = line_detect.daily_filter_band(gradient)
        # print(std_upper)
        # print(std_lower)
        # print(std_theta)

        # 血糖曲线检测
        line_dict = line_detect.find_bg_line_daily(src)

        # 血糖值计算预测
        result = {}
        for index, (x, y) in line_dict.items():
            bg_value = line_detect.get_bg_value_daily(x, y, std_upper=std_upper,
                                                std_lower=std_lower,
                                                std_theta=std_theta)
            result[index] = bg_value
        result = sorted(result.items(), key=operator.itemgetter(0), reverse=False)
        # print(result)
        res.type = const.BgImageType.DAILY
        res.add_start_time(0)
        res.add_raw_data(result, 0)
        # print("====================血糖值检测值===============")
        # print(res.to_dict())
        return res.to_dict(0)

    else:
        return res.to_dict(0)


def refine_result(time_start, result, input_time):
    if len(result) == 0:
        return {}
    time_start = int(time_start)
    std_day_start = std_dateTime(input_time)
    time_start = std_day_start + 3600 * time_start
    time_end = time_start + 3600 * 8
    delta_time = 3600 * 8 / len(result)
    res = []
    for i, r in enumerate(result):
        temp = {'time': int(time_start + i * delta_time),
                'value': r[1]}
        res.append(temp)
    res = {
        'data': res,
        'startTime': time_start,
        'endTime': time_end
    }
    return res


def std_dateTime(time_stamp):
    if isinstance(time_stamp, str):
        time_stamp = int(time_stamp)
    unit = 3600 * 24
    # day_start = (time_stamp - (time_stamp % unit)) - (8 * 3600)
    day_start = (time_stamp - ((time_stamp + 8 * 3600) % unit))
    return day_start
