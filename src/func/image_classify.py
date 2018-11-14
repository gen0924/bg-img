import numpy as np
import cv2
from skimage import img_as_uint, img_as_bool, morphology
from src.func.img_handler import show_img
from src import const
from skimage import filters
from matplotlib import pyplot as plt


# 区分是否为血糖仪图片
# 简单根据颜色区分，排除掉白色，蓝色和黑色，剩下颜色如果太多的
# 则认为是非血糖仪图片
def verify_bg(trans_img):
    deal_img = trans_img.copy()
    hsv = cv2.cvtColor(deal_img, cv2.COLOR_BGR2HSV)

    # 黑色部分
    lower = np.array([0, 0, 0])
    higher = np.array([180, 255, 81])
    mask2 = cv2.inRange(hsv, lower, higher)
    res1 = cv2.bitwise_and(deal_img, deal_img, mask=mask2)

    # 蓝色部分
    lower = np.array([100, 43, 46])
    higher = np.array([124, 255, 255])
    mask2 = cv2.inRange(hsv, lower, higher)
    res2 = cv2.bitwise_and(deal_img, deal_img, mask=mask2)

    # 白色部分
    lower = np.array([0, 0, 221])
    higher = np.array([180, 30, 255])
    mask2 = cv2.inRange(hsv, lower, higher)
    res3 = cv2.bitwise_and(deal_img, deal_img, mask=mask2)

    # res = res1 + res2 + res3
    # res = np.clip(res * 1.5, 0, 255)
    # res = np.uint8(res)

    gray = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
    ret, res1 = cv2.threshold(gray, thresh=5, maxval=255, type=cv2.THRESH_BINARY_INV)
    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    ret, res2 = cv2.threshold(gray, thresh=5, maxval=255, type=cv2.THRESH_BINARY)
    gray = cv2.cvtColor(res3, cv2.COLOR_BGR2GRAY)
    ret, res3 = cv2.threshold(gray, thresh=5, maxval=255, type=cv2.THRESH_BINARY)
    res = res1 + res2 + res3

    x, y = np.where(res <= 0)
    if len(x) > 5000:
        return False

    return True


# 血糖仪图片识别(24H/8H):依据右上角黄色铅笔
def recog_img_type(trans_img):
    width = trans_img.shape[1]
    img_mid = trans_img[110: 145, 15: int(width) - 15]
    show_img(img_mid, 0)

    mean = np.mean(img_mid)
    std = np.var(img_mid)
    print("mean:", mean)
    print("std:", std)

    # thresh = filters.threshold_otsu(img_mid)
    # ret, binary_img = cv2.threshold(img_mid, thresh=thresh - 10, maxval=255, type=cv2.THRESH_BINARY_INV)
    # show_img(binary_img, 0)
    # # 图像腐蚀操作去除噪声
    # # kernel = np.ones((2, 2), np.uint8)
    # # binary_img = cv2.erode(binary_img, kernel, iterations=1)
    # # show_img(binary_img, 0)
    #
    # # remove the small object
    # dst = img_as_bool(binary_img)
    # dst = morphology.remove_small_objects(dst, min_size=300, connectivity=1, in_place=True)
    # binary_img = img_as_uint(dst)
    # show_img(binary_img, 0)
    #
    # pix = np.sum(binary_img > 0)
    # print(pix)
    if std > 500:
        return const.BgImageType.DAILY

    return const.BgImageType.NORMAL


def decide_orientation(trans_img):
    height = trans_img.shape[0]
    width = trans_img.shape[1]
    img_type = recog_img_type(trans_img[..., 2])
    plt.figure()
    plt.subplot(221)
    plt.title('image')
    plt.imshow(trans_img)
    plt.subplot(222)
    plt.title('First Channel')
    plt.imshow(trans_img[..., 0], cmap='gray')
    plt.subplot(223)
    plt.title('Second Channel')
    plt.imshow(trans_img[..., 1], cmap='gray')
    plt.subplot(224)
    plt.title('Third Channel')
    plt.imshow(trans_img[..., 2], cmap='gray')
    plt.show()
    # print("==========imageType==========")
    # print(img_type)
    if img_type == const.BgImageType.DAILY:
        img_mid = trans_img[..., 2][110: 145, 15: int(width) - 15]
        thresh = filters.threshold_otsu(img_mid)
        ret, bin_mid = cv2.threshold(img_mid, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY_INV)
        # show_img(bin_mid, 0)

        rect_height = img_mid.shape[0]
        rect_width = img_mid.shape[1]
        part_up = bin_mid[0: int(rect_height / 2), 0: int(rect_width)]
        part_down = bin_mid[int(rect_height / 2): rect_height, 0: int(rect_width)]
        # show_img(part_down, 0)

        sum_up = np.sum(part_up)
        sum_down = np.sum(part_down)
        # if sum_up > sum_down:
        #     trans_img = np.rot90(trans_img)
        #     trans_img = np.rot90(trans_img)
        return trans_img, img_type

    if img_type == const.BgImageType.NORMAL:
        img_mid = trans_img[..., 2][int(height / 3): int((2 * height) / 3), 15: int(width) - 15]
        thresh = filters.threshold_otsu(img_mid)
        ret, bin_mid = cv2.threshold(img_mid, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY_INV)
        # show_img(bin_mid)

        # 图像腐蚀操作去处小噪声斑点
        kernel = np.ones((2, 4), np.uint8)
        bin_mid = cv2.erode(bin_mid, kernel, iterations=1)
        # show_img(bin_mid, 0)

        rect_height = img_mid.shape[0]
        rect_width = img_mid.shape[1]
        part_up = bin_mid[0: int(rect_height / 2), 0: int(rect_width)]
        part_down = bin_mid[int(rect_height / 2): rect_height, 0: int(rect_width)]

        sum_up = np.sum(part_up)
        sum_down = np.sum(part_down)
        # if sum_up < sum_down:
        #     trans_img = np.rot90(trans_img)
        #     trans_img = np.rot90(trans_img)
        return trans_img, img_type

