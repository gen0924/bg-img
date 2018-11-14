import numpy as np
from src import const

class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    @property
    def location(self):
        return int(self.x), int(self.y)

    @property
    def up_neighbor(self):
        return [(self.x - 1, self.y - 1), (self.x, self.y - 1), (self.x + 1, self.y - 1)]

    @property
    def down_neighbor(self):
        return [(self.x - 1, self.y + 1), (self.x, self.y + 1), (self.x + 1, self.y + 1)]

    @property
    def left_neighbor(self):
        return [(self.x - 1, self.y - 1), (self.x - 1, self.y), (self.x - 1, self.y + 1)]

    @property
    def right_neighbor(self):
        return [(self.x + 1, self.y + 1), (self.x + 1, self.y), (self.x + 1, self.y + 1)]


class Line:
    def __init__(self, p1, p2):
        if isinstance(p1, Point) and isinstance(p2, Point):
            self.p1 = p1
            self.p2 = p2
        else:
            self.p1 = Point(p1[0], p1[1])
            self.p2 = Point(p2[0], p2[1])

    @property
    def k(self):
        delta_x = self.p1.x - self.p2.x
        if delta_x == 0:
            return np.inf
        delta_y = self.p1.y - self.p2.y
        return delta_y / delta_x

    @property
    def b(self):
        y = self.p1.y
        kx = self.k * self.p1.x
        b = y - kx
        return b

    @property
    def point1(self):
        x = self.p1.x
        y = self.p1.y
        return x, y

    @property
    def point2(self):
        x = self.p2.x
        y = self.p2.y
        return x, y

    @property
    def left(self):
        left = self.p1.x
        if self.p2.x < left:
            left = self.p2.x
        return left

    @property
    def top(self):
        top = self.p1.y
        if self.p2.y < top:
            top = self.p2.y
        return top

    @property
    def is_vertical(self):
        if self.k > 6 or self.k < -6:
            return True
        return False

    @property
    def all_points(self):
        # points = []
        x_min = np.min([self.p1.x, self.p2.x]) - 1
        x_max = np.max([self.p1.x, self.p2.x]) + 1
        y_min = np.min([self.p1.y, self.p2.y]) - 1
        y_max = np.max([self.p1.y, self.p2.y]) + 1
        # for x in range(x_min, x_max + 1):
        #     for y in range(y_min, y_max + 1):
        #         points.append((x, y))
        return x_min, x_max, y_min, y_max

    # 求与另一条直线的夹角
    def angle_with(self, line2):
        k1 = self.k
        k2 = line2.k
        if k1 == 0 and k2 == np.inf:
            return np.pi / 2
        if k2 == 0 and k1 == np.inf:
            return np.pi / 2

        if k1 == np.inf:
            tan_theta = k2
        elif k2 == np.inf:
            tan_theta = k1
        else:
            tan_theta = (k1 - k2) / (1 + k1 * k2)
        theta = np.arctan(tan_theta)
        return theta

    # 求与另一条直线的交点
    def inter_point(self, line2):
        if line2.k - self.k == 0:
            return self.p1

        if self.k == np.inf:
            x = self.p1.x
            y = line2.p1.y
        elif line2.k == np.inf:
            x = line2.p1.x
            y = self.p1.y
        else:
            x = (self.b - line2.b) / (line2.k - self.k)
            y = self.k * x + self.b

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        p = Point(x, y)
        return p

    # 给定一张图片，根据直线上5个点，求出直线的模式（利用LBP方法）
    # 先设定只有五种模式，上边界，下边界，左边界，右边界和其他
    def lbp(self, image):
        img_height, img_width = np.shape(image)
        sample_points = []
        samples = 5
        if self.k == np.inf:
            delta_y = (self.p2.y - self.p1.y) // samples
            for i in range(0, samples):
                temp_p = Point(self.p1.x, self.p1.y + i * delta_y)
                sample_points.append(temp_p)
        elif self.k == 0:
            delta_x = (self.p2.x - self.p1.x) // samples
            for i in range(0, samples):
                temp_p = Point(self.p1.x + i * delta_x, self.p1.y)
                sample_points.append(temp_p)
        else:
            delta_x = (self.p2.x - self.p1.x) // samples
            for i in range(0, samples):
                x = self.p1.x + i * delta_x
                y = self.k * x + self.b
                temp_p = Point(x, y)
                sample_points.append(temp_p)

        patterns = []
        for p in sample_points:
            lbp = []
            for u_p in p.up_neighbor:
                if u_p[0] < 0 or \
                        u_p[0] >= img_width or \
                        u_p[1] < 0 or \
                        u_p[1] >= img_height:
                    lbp.append(0)
                elif image[int(u_p[0]), int(u_p[1])] > image[p.x, p.y]:
                    lbp.append(1)
                else:
                    lbp.append(0)
            for u_p in p.down_neighbor:
                if u_p[0] < 0 or \
                                u_p[0] >= img_width or \
                                u_p[1] < 0 or \
                                u_p[1] >= img_height:
                    lbp.append(0)
                elif image[u_p[0], u_p[1]] > image[p.x, p.y]:
                    lbp.append(1)
                else:
                    lbp.append(0)
            for u_p in p.left_neighbor:
                if u_p[0] < 0 or \
                                u_p[0] >= img_width or \
                                u_p[1] < 0 or \
                                u_p[1] >= img_height:
                    lbp.append(0)
                elif image[u_p[0], u_p[1]] > image[p.x, p.y]:
                    lbp.append(1)
                else:
                    lbp.append(0)
            for u_p in p.right_neighbor:
                if u_p[0] < 0 or \
                                u_p[0] >= img_width or \
                                u_p[1] < 0 or \
                                u_p[1] >= img_height:
                    lbp.append(0)
                elif image[u_p[0], u_p[1]] > image[p.x, p.y]:
                    lbp.append(1)
                else:
                    lbp.append(0)

            if lbp[:3] == [1, 1, 1] and lbp[3:6] == [0, 0, 0]:    # 上白，下黑
                patterns.append(const.DOWN)
            elif lbp[:3] == [0, 0, 0] and lbp[3:6] == [1, 1, 1]:
                patterns.append(const.UP)
            elif lbp[6:9] == [1, 1, 1] and lbp[9:12] == [0, 0, 0]:   # 左白，右黑
                patterns.append(const.RIGHT)
            elif lbp[6:9] == [0, 0, 0] and lbp[9:12] == [1, 1, 1]:
                patterns.append(const.LEFT)
            else:
                patterns.append(const.OTHER)

        if patterns.count(const.UP) > samples / 2:
            return const.UP
        if patterns.count(const.DOWN) > samples / 2:
            return const.DOWN
        if patterns.count(const.LEFT) > samples / 2:
            return const.LEFT
        if patterns.count(const.RIGHT) > samples / 2:
            return const.RIGHT
        return const.OTHER



if __name__ == '__main__':
    p1 = Point(0, 0)
    p2 = Point(100, 130)
    p3 = Point(0, 130)
    p4 = Point(100, 0)
    l1 = Line(p1, p2)
    l2 = Line(p3, p4)
    p = l1.inter_point(l2)
    # print(angle)
    print(p.location)