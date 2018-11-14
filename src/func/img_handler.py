import os
# from bg_image import img_lib
# from bg_image import Amy_lib as img_lib
import cv2
import numpy as np

std_size = (1024, 1024)

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


# def read_img(pid):
#     pic_path = os.path.dirname(img_lib.__file__) + '\\p' + str(pid) + '.jpg'
#     color_img = cv2.imread(pic_path)
#     # print(np.shape(color_img), pid)
#     color_img = cv2.resize(color_img, std_size)
#     return color_img


def show_img(img, time=3000):
    cv2.imshow('p1', img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def draw_contour(img, contours, hierarchy=None, time=2000):
    new_img = img.copy()
    if hierarchy is None:
        c_img = cv2.drawContours(new_img, contours, -1, GREEN, 2)
    else:
        c_img = cv2.drawContours(new_img, contours, -1, GREEN, 2,
                                 hierarchy=hierarchy, maxLevel=1)
    # show_img(c_img, time)


def draw_rectangle(img, point1=None, point2=None, top=0, left=0, width=0, height=0, time=2000):
    # point1 = rects[0:2]
    # point2 = rects[2:4]
    if point1 is not None and point2 is not None:
        point1 = point1
        point2 = point2
    else:
        point1 = (left, top)
        point2 = (left + width, top + height)
    img = cv2.rectangle(img, point1, point2, (0, 0, 255), 2)
    # show_img(img, time=time)


def calcAndDrawHist(image, color=[0, 0, 255]):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

    return histImg


def trackbar(in_img):
    cv2.namedWindow('p1')
    img_height, img_width, _ = np.shape(in_img)
    img_area = img_height * img_width
    def func(arg):
        gray_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.medianBlur(gray_img, 7)
        kernel_size = cv2.getTrackbarPos('track', 'p1')
        print(kernel_size)
        ret, binary_img = cv2.threshold(gray_img, thresh=np.round(100), maxval=255, type=cv2.THRESH_BINARY)

        kernel1 = np.ones((kernel_size, kernel_size), np.uint8)
        kernel2 = np.ones((28, 28), np.uint8)
        closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel1)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
        canny_img = closing
        # show_img(opening, 0)

    cv2.createTrackbar('track', 'p1', 0, 50, func)
    func(0)
    cv2.waitKey(0)


def refine_rect(img, rotate_rect):
    angle = rotate_rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    # img_rot = cv2.warpAffine(img, M, (cols, rows))
    box = cv2.boxPoints(rotate_rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    # pts[pts < 0] = 0
    # print(pts)
    points = [tuple(pts[0]), tuple(pts[2])]
    return points


def image_generator():
    # for pid in range(1, 18):
    #     print('pid', pid)
    #     img = read_img(pid)
    #     yield img
    pass


if __name__ == '__main__':
    for img in image_generator():
        trackbar(img)
        # show_img(img)
