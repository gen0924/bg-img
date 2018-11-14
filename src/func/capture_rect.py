import cv2
import numpy as np
from src.func import detect
from src.func import Shape
from src import const

std_size = (1024, 1024)


def capture_rect(img):
    img_height, img_width, _ = np.shape(img)
    img_area = img_width * img_height
    # show_img(img, 0)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (9, 9), 0)
    # gray_img = cv2.medianBlur(gray_img, 7)
    # gray_img = cv2.equalizeHist(gray_img)
    # gradX = cv2.Sobel(gray_img, ddepth=cv2.CV_32F, dx=1, dy=0)
    # gradY = cv2.Sobel(gray_img, ddepth=cv2.CV_32F, dx=0, dy=1)
    #
    # gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.convertScaleAbs(gradient)
    # show_img(gradient, 0)

    img_mean = 100

    # show_img(gray_img, 0)

    # gray_img[gray_img > 250] = 0
    # show_img(hist_img, 0)
    # show_img(gray_img, 0)

    # ret, binary_img = cv2.threshold(gray_img, thresh=250, maxval=255, type=cv2.THRESH_BINARY_INV)
    # 阈值分割
    ret, binary_img = cv2.threshold(gray_img, thresh=np.round(img_mean), maxval=255, type=cv2.THRESH_BINARY)

    # kernel1 = np.ones((22, 22), np.uint8)
    kernel1 = np.ones((10, 10), np.uint8)

    kernel2 = np.ones((28, 28), np.uint8)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel1)
    # show_img(opening, 0)

    gradX = cv2.Sobel(opening, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(opening, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # show_img(gradient, 0)


    # canny_img = opening
    # show_img(opening, 0)
    # contour_img, contour, hierarchy = cv2.findContours(opening, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contour_img, contour, hierarchy = cv2.findContours(gradient, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    candi_rects = []
    for cnt in contour:
        area = cv2.contourArea(cnt)
        # 0.1
        if area < img_area * 0.1:
            continue
        candi_rects.append(cnt)
        # print(np.shape(cnt))

    # rect_img = cv2.resize(img, std_size)
    # rect_img = None

    max_area = 0
    f_corner_point = []
    for idx, cnt in enumerate(candi_rects):
        corner_point = cv2.approxPolyDP(cnt, 10, True)
        # draw_contour(img, cnt, time=0)

        # 把不是四边形的过滤掉
        if len(corner_point) != 4:
            continue
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            f_corner_point = corner_point

    if len(f_corner_point) == 0:
        new_gray = cv2.equalizeHist(gray_img)
        ret, binary_img = cv2.threshold(new_gray, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
        # kernel1 = np.ones((22, 22), np.uint8)
        # kernel2 = np.ones((28, 28), np.uint8)
        # closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel1)
        # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
        # show_img(binary_img, 0)
        contour_img, contour, hierarchy = cv2.findContours(binary_img, mode=cv2.RETR_EXTERNAL,
                                                           method=cv2.CHAIN_APPROX_NONE)
        max_area = 0
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.1:
                continue
            corner_point = cv2.approxPolyDP(cnt, 10, True)
            if len(corner_point) != 4:
                continue
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                f_corner_point = corner_point

    if len(f_corner_point) == 0:
        g_img = gray_img.copy()
        f_corner_point = find_rect_points(opening, g_img, img.copy())

    cp = []
    for i, point in enumerate(f_corner_point):
        cp.append(point[0])
    corner_point = cp
    if len(corner_point) == 0:
        return img.copy()

    # 投影变换—左上，左下，右下，右上
    (p1, p2, p3, p4) = re_range(corner_point)
    dis_width = np.sqrt(np.sum(np.square(p1 - p4)))
    rdis_width = dis_width * (detect.g_imgWidth/1024)
    dis_height = np.sqrt(np.sum(np.square(p1 - p2)))
    rdis_height = dis_height * (detect.g_imgHeight/1024)

    arraySRC = np.array([p1, p2, p3, p4]).astype(np.float32)
    arrayDST = np.array([[0, 0], [255, 0], [255, 255], [0, 255]]).astype(np.float32)
    PerspectiveMatrix = cv2.getPerspectiveTransform(arraySRC, arrayDST)
    trans_img = cv2.warpPerspective(img, PerspectiveMatrix, (256, 256))
    if rdis_height < rdis_width:
        trans_img = np.rot90(trans_img)

    return trans_img


# 处理四边形四个角点的方法
# 1. 找到左上，左下，右下，右上
def re_range(corner_points):       # 默认是4个点
    # print(type(corner_points))
    p1 = corner_points[0]
    p2 = corner_points[1]
    p3 = corner_points[2]
    p4 = corner_points[3]

    # 其实应该是两条对角线的交点，但是这里近似用一条对角线的中点代替了
    (x0, y0) = ((p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2)
    left_up = None
    left_bottom = None
    right_bottom = None
    right_up = None
    p = p1
    for p in corner_points:
        if p[0] < x0 and p[1] < y0:
            left_up = p
        # elif p[0] < x0 and p[1] > y0:
        #     left_bottom = p
        elif p[0] > x0 and p[1] < y0:
            left_bottom = p

        elif p[0] > x0 and p[1] > y0:
            right_bottom = p
        else:
            right_up = p
    if left_up is None:
        left_up = p
    if left_bottom is None:
        left_bottom = p
    if right_bottom is None:
        right_bottom = p
    if right_up is None:
        right_up = p

    left_up, left_bottom, right_bottom, right_up = \
        map(lambda x: np.array(x), [left_up, left_bottom, right_bottom, right_up])

    return left_up, left_bottom, right_bottom, right_up


# 通过搜索直线相交的方式，找出目标区域的四个点
def find_rect_points(input_img, g_img, real_img=None):
    # show_img(input_img, 0)
    img_height, img_width = np.shape(input_img)
    # kernel2 = np.ones((30, 30), np.uint8)
    # closing = cv2.morphologyEx(input_img, cv2.MORPH_CLOSE, kernel2)
    # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    # # canny_img = cv2.Canny(input_img, 120, 100)
    # gradX = cv2.Sobel(opening, ddepth=cv2.CV_32F, dx=1, dy=0)
    # gradY = cv2.Sobel(opening, ddepth=cv2.CV_32F, dx=0, dy=1)
    # gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.convertScaleAbs(gradient)
    # show_img(gradient, 0)

    # lines = cv2.HoughLinesP(gradient, 1, np.pi / 180, 80, minLineLength=250, maxLineGap=200)
    lines = cv2.HoughLinesP(input_img, 1, np.pi / 180, 100, minLineLength=150, maxLineGap=200)

    # 直线拟合求交点
    # lines = cv2.HoughLinesP(gradient, 1, np.pi/180, 100, minLineLength=150, maxLineGap=200)
    lines1 = lines[:, 0, :]
    for x1, y1, x2, y2 in lines1[:]:
        cv2.line(input_img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    hori_lines = []
    verti_lines = []

    if lines is None:
        return []

    for l in lines:
        l = tuple(l[0])
        # canny_img = cv2.line(canny_img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2)
        if l[0] + l[1] < l[2] + l[3]:
            p1 = Shape.Point(l[0], l[1])
            p2 = Shape.Point(l[2], l[3])
        else:
            p2 = Shape.Point(l[0], l[1])
            p1 = Shape.Point(l[2], l[3])
        my_line = Shape.Line(p1, p2)
        l_k = np.abs(my_line.k)
        if l_k < 0.5:
            break_tag = False   # 去掉过于相近的线
            for h_l in hori_lines:
                if abs(h_l.top - my_line.top) < 10:
                    break_tag = True
                    break
            if not break_tag:
                hori_lines.append(my_line)
        elif l_k > 1:
            break_tag = False
            for v_l in verti_lines:
                if abs(v_l.left - my_line.left) < 10:
                    break_tag = True
                    break
            if not break_tag:
                verti_lines.append(my_line)

    if not hori_lines or not verti_lines:
        return []

    # show_img(canny_img, 0)
    up_line = None
    bottom_line = None
    left_line = None
    right_line = None

    hori_lines = sorted(hori_lines, key=lambda line: line.top)
    if len(hori_lines) < 0:
        return []
    elif len(hori_lines) == 1:
        h_l = hori_lines[0]
        if h_l.top < img_height / 2:
            up_line = h_l
        else:
            bottom_line = h_l
    elif len(hori_lines) == 2:
        up_line = hori_lines[0]
        bottom_line = hori_lines[1]
    else:
        up_line = hori_lines[0]
        bottom_line = hori_lines[-1]
        for b_l in hori_lines[::-1]:
            leng = 3
            mark_part1 = input_img[b_l.top - leng:b_l.top, b_l.p1.x:b_l.p2.x]
            mark_part2 = input_img[b_l.top:b_l.top + leng, b_l.p1.x:b_l.p2.x]
            if np.average(mark_part1) > 100 and \
                    b_l.top > img_height / 2 and \
                    np.average(mark_part2) < 100:
                bottom_line = b_l
                # break
        for b_l in hori_lines:
            leng = 3
            mark_part1 = input_img[b_l.top - leng:b_l.top, b_l.p1.x:b_l.p2.x]
            mark_part2 = input_img[b_l.top:b_l.top + leng, b_l.p1.x:b_l.p2.x]
            if np.average(mark_part1) < 100 and \
                            b_l.top < img_height / 2 and \
                            np.average(mark_part2) > 100:
                up_line = b_l
                # break

        for index, l in enumerate(hori_lines):
            if bottom_line.top - l.top < 350 and index >= 1:
                up_line = hori_lines[index - 1]
                break

    verti_lines = sorted(verti_lines, key=lambda line: line.left)
    # temp_verti_line = []
    # pre_line = verti_lines[0]
    # for v_l in verti_lines[1:]:
    #     if v_l.left - pre_line.left > 3:
    #         temp_verti_line.append(pre_line)
    #         pre_line = v_l
    #     else:

    if len(verti_lines) < 0:
        return []
    elif len(verti_lines) == 1:
        v_l = verti_lines[0]
        if v_l.left < img_width / 2:
            left_line = v_l
        else:
            right_line = v_l
    elif len(verti_lines) == 2:
        left_line = verti_lines[0]
        right_line = verti_lines[1]
    else:
        left_line = verti_lines[0]
        right_line = verti_lines[-1]
        # for index, l in enumerate(verti_lines):
        #     if right_line.left - left_line.left < 350 and index >= 1:
        #         left_line = verti_lines[index - 1]
        #         break
        # i_img = input_img.copy()
        for b_l in verti_lines[::-1]:
            leng = 3
            mark_part1 = input_img[b_l.p1.y: b_l.p2.y, b_l.left - leng: b_l.left]
            mark_part2 = input_img[b_l.p1.y: b_l.p2.y, b_l.left: b_l.left + leng]
            # show_img(mark_part1)
            a = np.average(mark_part1)
            b = np.average(mark_part2)
            if np.average(mark_part1) > 200 and \
                            b_l.left > img_width / 2 and \
                            np.average(mark_part2) < 200:
                right_line = b_l
                # break
        for b_l in verti_lines:
            leng = 1
            mark_part1 = input_img[b_l.p1.y: b_l.p2.y, b_l.left - leng: b_l.left]
            mark_part2 = input_img[b_l.p1.y: b_l.p2.y, b_l.left: b_l.left + leng]
            if np.average(mark_part1) < 200 and \
                            b_l.left < img_width / 2 and \
                            np.average(mark_part2) > 200:
                left_line = b_l

        for v_l in verti_lines:
            if v_l.lbp(g_img) == const.LEFT and v_l.left < img_width / 2:
                left_line = v_l
            elif v_l.lbp(g_img) == const.RIGHT and v_l.left > img_width / 2:
                right_line = v_l
                # break

    # for l in lines:
    #     l = l[0]
    #     real_img = cv2.line(real_img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2)
    # real_img = cv2.line(real_img, left_line.point1, left_line.point2, (0,0,255), 1)
    # show_img(real_img, 0)

    if left_line is not None and up_line is not None:
        left_up = left_line.inter_point(up_line)
    elif left_line is None and up_line is None:
        left_up = Shape.Point(0, 0)
    elif up_line is None:
        left_up = left_line.p1
    else:
        left_up = up_line.p1

    if left_line is not None and bottom_line is not None:
        left_bottom = left_line.inter_point(bottom_line)
    elif left_line is None and bottom_line is None:
        left_bottom = Shape.Point(img_height, 0)
    elif bottom_line is None:
        left_bottom = left_line.p2
    else:
        left_bottom = bottom_line.p1

    if right_line is not None and bottom_line is not None:
        right_bottom = right_line.inter_point(bottom_line)
    elif right_line is None and bottom_line is None:
        right_bottom = Shape.Point(img_height, img_width)
    elif bottom_line is None:
        right_bottom = right_line.p2
    else:
        right_bottom = bottom_line.p2

    if right_line is not None and up_line is not None:
        right_up = right_line.inter_point(up_line)
    elif right_line is None and up_line is None:
        right_up = Shape.Point(x=img_width, y=img_height)
    elif up_line is None:
        right_up = right_line.p1

    else:
        right_up = up_line.p2

    # left_bottom = left_line.inter_point(bottom_line)
    # right_bottom = right_line.inter_point(bottom_line)
    # right_up = right_line.inter_point(up_line)
    return [left_up.location], [left_bottom.location], [right_bottom.location], [right_up.location]
