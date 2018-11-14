import time
from datetime import datetime


# 把datetime转成字符串
def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d-%H")


# 把字符串转成datetime
def string_toDatetime(string):
    try:
        result = datetime.strptime(string, "%Y-%m-%d %H:%M")
    except ValueError:
        result = datetime.strptime(string, "%Y-%m-%d")

    return result


# 把字符串转成时间戳形式
def string_toTimestamp(strTime):
    return time.mktime(string_toDatetime(strTime).timetuple())


# 把时间戳转成字符串形式
def timestamp_toString(stamp):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))


# 把datetime类型转外时间戳形式
def datetime_toTimestamp(dateTim):
    return time.mktime(dateTim.timetuple())


# 今日日期
def today_date():
    t = time.time()
    date = time.strftime("%Y-%m-%d", time.localtime(t))
    return date


def now_time():
    t = time.time()
    time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    time_str = time_str.split()[1]
    return time_str


# 时间戳到小时
def timestamp_toHour(stamp):
    time_str = timestamp_toString(stamp)
    times = time_str.split(' ', 1)
    hour = times[1].split(':')[0]
    return int(hour)


if __name__ == '__main__':
    stamp1 = time.time()
    hour = timestamp_toHour(stamp1)
    data = time.strftime("%Y-%m-%d", time.localtime(stamp1))
    stamp2 = stamp1 - 3600*24
    hour1 = timestamp_toHour(stamp2)
    data1 = time.strftime("%Y-%m-%d", time.localtime(stamp2))
    print(data)
    print(hour)
    print(data1)
    print(hour1)
