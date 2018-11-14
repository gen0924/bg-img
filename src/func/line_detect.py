import cv2
import numpy as np
from src.func import Shape
from skimage import filters, img_as_bool, img_as_uint, morphology
from src.func.img_handler import show_img


def normal_filter_band(img,
                mid_line=72,
                default_std_upper=81,
                default_std_lower=63):
    lines = normal_line_detect(img)

    # 根据统计经验，72可以作为分割两条线的中点
    # 根据经验值，band的宽度大概是18pixel
    default_width = 18
    theta_list = []
    upper_line = []
    lower_line = []
    if lines is not None:
        for line in lines:
            rho = line[0]
            theta = line[1]
            if rho < mid_line:
                lower_line.append(rho)
            else:
                upper_line.append(rho)
            theta_list.append(theta)

    # 确认两条直线的theta值（默认是平行的，所以共用一个theta)
    if not theta_list:
        std_theta = np.pi / 2     # 如果没有的话，默认是与底面平行
    else:
        std_theta = np.mean(theta_list)

    if not upper_line and not lower_line:        # 都没有检测出来，给默认统计值 200 和 220
        std_upper = default_std_upper
        std_lower = default_std_lower
    elif not upper_line:
        std_lower = np.mean(lower_line)
        if 60 < std_lower < 65:
            std_lower = std_lower
        else:
            std_lower = default_std_lower
        std_upper = std_lower + default_width
    elif not lower_line:
        std_upper = np.mean(upper_line)
        if 78 < std_upper < 83:
            std_upper = std_upper
        else:
            std_upper = default_std_upper
        std_lower = std_upper - default_width
    else:
        std_upper = np.mean(upper_line)
        std_lower = np.mean(lower_line)
        if 78 < std_upper < 83:
            std_upper = std_upper
        else:
            std_upper = default_std_upper
        if 60 < std_lower < 65:
            std_lower = std_lower
        else:
            std_lower = default_std_lower

    return std_upper, std_lower, std_theta


def daily_filter_band(img,
                mid_line=71,
                default_std_upper=85,
                default_std_lower=56):

    lines = daily_line_detect(img)

    # 根据统计经验，71可以作为分割两条线的中点
    # 根据经验值，band的宽度大概是29pixel
    default_width = 29
    theta_list = []
    upper_line = []
    lower_line = []
    if lines is not None:
        for line in lines:
            rho = line[0]
            theta = line[1]
            if rho < mid_line:
                lower_line.append(rho)
            else:
                upper_line.append(rho)
            theta_list.append(theta)

    # 确认两条直线的theta值（默认是平行的，所以共用一个theta)
    if not theta_list:
        std_theta = np.pi / 2  # 如果没有的话，默认是与底面平行
    else:
        std_theta = np.mean(theta_list)

    if not upper_line and not lower_line:  # 都没有检测出来，给默认统计值 200 和 220
        std_upper = default_std_upper
        std_lower = default_std_lower
    elif not upper_line:
        std_lower = np.mean(lower_line)
        if 53 < std_lower < 58:
            std_lower = std_lower
        else:
            std_lower = default_std_lower
        std_upper = std_lower + default_width
    elif not lower_line:
        std_upper = np.mean(upper_line)
        if 82 < std_upper < 87:
            std_upper = std_upper
        else:
            std_upper = default_std_upper
        std_lower = std_upper - default_width
    else:
        std_upper = np.mean(upper_line)
        std_lower = np.mean(lower_line)
        if 82 < std_upper < 87:
            std_upper = std_upper
        else:
            std_upper = default_std_upper

        if 53 < std_lower < 58:
            std_lower = std_lower
        else:
            std_lower = default_std_lower

    return std_upper, std_lower, std_theta


# 对binary图片进行直线检测
def normal_line_detect(img, bottom=50, top=90):
    lines = cv2.HoughLines(img, 1, np.pi / 180, 100)
    # lines = cv2.HoughLines(img, 1, np.pi / 360, 50)
    if lines is None:
        return None

    # 把不在目标范围内的直线去掉
    result_lines = []
    for line in lines:
        if top > line[0][0] > bottom:
            result_lines.append(line[0])
    if len(result_lines) == 0:
        return None

    return result_lines


# 对每日图片类型的直线检测
def daily_line_detect(img, bottom=45, top=95):
    # lines = cv2.HoughLines(img, 1, np.pi / 180, 50, min_theta=(0.9 * np.pi / 2), max_theta=(1.1 * np.pi / 2))
    lines = cv2.HoughLines(img, 1, np.pi / 180, 50)
    if lines is None:
        return None

    # 把不在目标范围内的直线去掉
    result_lines = []
    for line in lines:
        if top > line[0][0] > bottom:
            result_lines.append(line[0])
    if len(result_lines) == 0:
        return None
    return result_lines


# 检测血糖曲线—8H
def find_bg_line_normal(img, location_start, location_end):
    # 灰度拉伸归一化0—255
    cv2.imwrite('/home/liu/icx/kmeans-seg/test5.png', img)
    img[..., 0] = gray_value(img[..., 0])
    show_img(img[..., 0])
    # 阈值分割:defult = thresh-50
    thresh = filters.threshold_otsu(img[..., 0])
    ret, binary_img = cv2.threshold(img[..., 0], thresh=thresh-50, maxval=255, type=cv2.THRESH_BINARY_INV)
    show_img(binary_img, 0)
    # 去除坐标轴竖线
    lines = cv2.HoughLinesP(binary_img, 1, np.pi / 180, 8, minLineLength=10, maxLineGap=6)
    if lines is not None:
        for l in lines:
            l = tuple(l[0])
            p1 = Shape.Point(l[0], l[1])
            p2 = Shape.Point(l[2], l[3])
            my_line = Shape.Line(p1, p2)

            if my_line.is_vertical:
                x_min, x_max, y_min, y_max = my_line.all_points
                binary_img[y_min: y_max, x_min: x_max] = 0

    height = binary_img.shape[0]
    width = binary_img.shape[1]
    for y in range(location_start):
        for x in range(height):
            binary_img[x, y] = 0

    for y in range(location_end+3, width):
        for x in range(height):
            binary_img[x, y] = 0
    show_img(binary_img, 0)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    binary_img = cv2.erode(binary_img, element)
    show_img(binary_img, 0)
    # （3，1）
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))
    edge = cv2.dilate(binary_img, element)
    # show_img(edge, 0)

    dst = img_as_bool(edge)

    dst = morphology.remove_small_objects(dst, min_size=200, connectivity=1, in_place=True)
    dst = img_as_uint(dst)

    show_img(dst, 0)
    #
    # height = dst.shape[0]
    # width = dst.shape[1]
    # for y in range(location_start):
    #     for x in range(height):
    #         dst[x, y] = 0
    #
    # for y in range(location_end + 5, width):
    #     for x in range(height):
    #         dst[x, y] = 0
    # show_img(dst, 0)
    line_dict = cal_values(dst, location_start, location_end)
    return line_dict


# 检测血糖曲线—24H
def find_bg_line_daily(img):
    # 灰度拉伸归一化0—255
    img[..., 0] = gray_value(img[..., 0])
    show_img(img[..., 0], 0)

    # 阈值分割
    thresh = filters.threshold_otsu(img[..., 0])
    ret, binary_img = cv2.threshold(img[..., 0], thresh=thresh-60, maxval=255, type=cv2.THRESH_BINARY_INV)
    # show_img(binary_img, 0)

    # 去除坐标轴竖线
    lines = cv2.HoughLinesP(binary_img, 1, np.pi / 180, 8, minLineLength=10, maxLineGap=6)
    if lines is not None:
        for l in lines:
            l = tuple(l[0])
            p1 = Shape.Point(l[0], l[1])
            p2 = Shape.Point(l[2], l[3])
            my_line = Shape.Line(p1, p2)

            if my_line.is_vertical:
                x_min, x_max, y_min, y_max = my_line.all_points
                binary_img[y_min: y_max, x_min: x_max] = 0

    height = binary_img.shape[0]
    # width = binary_img.shape[1]
    for y in range(16):
        for x in range(height):
            binary_img[x, y] = 0

    # （3，1）
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 2))
    edge = cv2.dilate(binary_img, element)
    dst = img_as_bool(edge)
    dst = morphology.remove_small_objects(dst, min_size=100, connectivity=1, in_place=True)
    dst = img_as_uint(dst)
    # show_img(dst, 0)

    height = dst.shape[0]
    width = dst.shape[1]
    for y in range(16):
        for x in range(height):
            dst[x, y] = 0
    show_img(dst, 0)
    line_dict = cal_values(dst, 16, width - 3)
    return line_dict


def cal_values(edge, location_start, location_end):
    # edge = edge.astype(np.int8)
    bg_curve = np.where(edge > 0)
    if len(bg_curve[0]) == 0:
        return None

    left = location_start
    right = location_end + 3
    line = []
    for idx in range(left, right):
        ys = edge[:, idx]
        ys_value = np.where(ys > 0)

        if len(ys_value) == 0:
            line.append(np.nan)
        else:
            line.append(np.mean(ys_value[0]))

    line = np.array(line)

    f_res = {}
    for idx in range(line.shape[0]):
        f_res[idx] = (idx, line[idx])
    return f_res


# 插值法
def cal_values1(edge):
    bg_curve = np.where(edge > 0)
    if len(bg_curve[0]) == 0:
        return None
    left = int(np.min(bg_curve[1]))
    right = int(np.max(bg_curve[1]))

    line = []
    for idx in range(left, right + 1):
        ys = edge[:, idx]
        ys_value = np.where(ys > 0)

        if len(ys_value) == 0:
            line.append(np.nan)
        else:
            line.append(np.mean(ys_value[0]))

    line = np.array(line)
    # nans = [np.nan] * 10
    # line[10:20] = np.array(nans)

    non_nan = ~np.isnan(line)
    nonnan_x = non_nan.ravel().nonzero()[0]
    nonnan_y = line[non_nan]
    nan_x = np.isnan(line).ravel().nonzero()[0]

    line[np.isnan(line)] = np.interp(nan_x, nonnan_x, nonnan_y)

    f_res = {}
    for idx in range(line.shape[0]):
        f_res[idx] = (idx, line[idx])

    return f_res


# 根据点的位置，以及两条线的信息，判断血糖值
def get_bg_value_normal(width, height, std_upper, std_lower, std_theta):

    # 上线和下线对应的血糖值分别是7.8 和 3.9，求出rho每差1 对应的血糖值的差异
    std_var = (7.8 - 3.9) / (std_lower - std_upper)
    line_rho = height

    # 分别按两条线计算，取均值
    bg_value1 = 3.9 + std_var * (std_lower - line_rho)
    bg_value2 = 7.8 + std_var * (std_upper - line_rho)
    bg_value = (bg_value1 + bg_value2) / 2
    if bg_value < 2.0:
        bg_value = 2.0
    return bg_value


def get_bg_value_daily(width, height, std_upper, std_lower, std_theta):

    # 上线和下线对应的血糖值分别是9.2 和 3.9，求出rho每差1 对应的血糖值的差异
    std_var = (9.2 - 3.9) / (std_lower - std_upper)
    line_rho = height

    # 分别按两条线计算，取均值
    bg_value1 = 3.9 + std_var * (std_lower - line_rho)
    bg_value2 = 9.2 + std_var * (std_upper - line_rho)
    bg_value = (bg_value1 + bg_value2) / 2
    if bg_value < 2.0:
        bg_value = 2.0
    return bg_value


# 灰度图像拉伸—灰度值归一化 0—255
def gray_value(img):
    height = img.shape[0]
    width = img.shape[1]
    max = np.max(img)
    min = np.min(img)

    for y in range(width):
        for x in range(height):
            img[x, y] = (255 / (max - min)) * (img[x, y] - min)
    return img





