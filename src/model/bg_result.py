from src import const
import time
from src.func import time_util

class BgResult:

    # Input time 用于确认日期
    def __init__(self, start_time=None, input_time=None):
        if input_time is None:
            input_time = time.time()
        if start_time is None:
            start_time = 0

        self.start_time = int(start_time)
        self.std_day_start = std_dateTime(input_time)
        # start_time = std_day_start + 3600 * start_time

        self.type = const.BgImageType.NOT_BG
        # self.end_time = 0
        self.data = []

    def add_start_time(self, start_time):
        self.start_time = self.std_day_start + 3600 * start_time
        return

    @property
    def end_time(self):
        if self.type == const.BgImageType.NOT_BG:
            return 0
        if self.type == const.BgImageType.NORMAL:
            end_time = self.start_time + 3600 * 8
            return end_time
        if self.type == const.BgImageType.DAILY:
            end_time = self.start_time + 3600 * 24
            return end_time
        return 0

    @property
    def sample_numbers(self):
        if self.type == const.BgImageType.DAILY:
            points = 24 // const.time_interval
        else:
            end_hour = time_util.timestamp_toHour(self.end_time)
            start_hour = time_util.timestamp_toHour(self.start_time)
            # points = (end_hour - start_hour) // const.time_interval
            points = 8 // const.time_interval
        points = int(points)
        return points

    def add_raw_data(self, result, ocr_start_time):
        delta_time = (self.end_time - self.start_time) / self.sample_numbers

        point_delta = len(result) / self.sample_numbers
        tmp_result = []
        for i in range(0, self.sample_numbers):
            tmp_result.append(result[int(i * point_delta)])
        result = tmp_result

        res = []
        for i, r in enumerate(result):
            if ocr_start_time > 15:
                temp = {'time': int(self.start_time - 3600*24 + i * delta_time), 'value': r[1]}
                res.append(temp)
            else:
                temp = {'time': int(self.start_time + i * delta_time), 'value': r[1]}
                res.append(temp)
        self.data = res
        return

    def to_dict(self, ocr_start_time):
        if ocr_start_time > 15:
            res = {
                'type': self.type,
                'data': self.data,
                'startTime': int(self.start_time - 3600*24),
                'endTime': int(self.end_time - 3600*24)
            }
        else:
            res = {
                'type': self.type,
                'data': self.data,
                'startTime': int(self.start_time),
                'endTime': int(self.end_time)
            }
        return res


def std_dateTime(time_stamp):
    if isinstance(time_stamp, str):
        time_stamp = int(time_stamp)
    unit = 3600 * 24
    # day_start = (time_stamp - (time_stamp % unit)) - (8 * 3600)
    day_start = (time_stamp - ((time_stamp + 8 * 3600) % unit))
    return day_start