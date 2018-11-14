# 项目介绍
提供血糖仪图片，识别图片中血糖曲线的血糖值，以及开始时间

## 版本
v1.0

## 此项目提供了接口和网页前端，两种交互方式
1. 接口交互：
url：119.29.248.56/ai/image/v1.0/bg_img_detect
输入：
image：（图片文件）
input_time: 11位时间戳（用于确认当前日期）（optional，如果没有，则默认为当前时间）

输出：
data：血糖曲线值，是个dict，从第一个点到第N个点的值
time_start: 血糖仪上初始显示时间（每张图片都默认是持续8小时）

2. 前端交互：
url：https://119.29.248.56/ai/image/v1.0/bg_img
直接登陆后，按提示操作即可。结果会用折线图形式画出。

## 截图示例：
接口调用：
输入：
![Image text](https://raw.githubusercontent.com/tanmc123/Image_lib/master/p3.jpg)

输出：
![Image text](https://raw.githubusercontent.com/tanmc123/Image_lib/master/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20180803120515.png)

输出示例：
```json
{
    "data": [
        {
            "time": 1533243600,
            "value": 3.4124999999999996
        },
        {
            "time": 1533243756,
            "value": 3.4124999999999996
        },
        {
            "time": 1533243913,
            "value": 3.4124999999999996
        },
        {
            "time": 1533244069,
            "value": 3.4124999999999996
        },
        {
            "time": 1533244226,
            "value": 3.4124999999999996
        },
        {
            "time": 1533244382,
            "value": 3.4124999999999996
        },
        {
            "time": 1533244539,
            "value": 3.4124999999999996
        },
        {
            "time": 1533244695,
            "value": 3.4124999999999996
        },
        {
            "time": 1533244852,
            "value": 3.4124999999999996
        },
        {
            "time": 1533245008,
            "value": 3.4124999999999996
        },
        {
            "time": 1533245165,
            "value": 3.4124999999999996
        },
        {
            "time": 1533245321,
            "value": 3.4124999999999996
        },
        {
            "time": 1533245478,
            "value": 3.4124999999999996
        },
        {
            "time": 1533245634,
            "value": 3.4124999999999996
        },
        {
            "time": 1533245791,
            "value": 3.4124999999999996
        },
        {
            "time": 1533245947,
            "value": 3.4124999999999996
        },
        {
            "time": 1533246104,
            "value": 3.4124999999999996
        },
        {
            "time": 1533246260,
            "value": 3.4124999999999996
        },
        {
            "time": 1533246417,
            "value": 3.4124999999999996
        },
        {
            "time": 1533246573,
            "value": 3.4124999999999996
        },
        {
            "time": 1533246730,
            "value": 3.4124999999999996
        },
        {
            "time": 1533246886,
            "value": 3.4124999999999996
        },
        {
            "time": 1533247043,
            "value": 3.4124999999999996
        },
        {
            "time": 1533247200,
            "value": 3.4124999999999996
        },
        {
            "time": 1533247356,
            "value": 3.3149999999999995
        },
        {
            "time": 1533247513,
            "value": 3.3149999999999995
        },
        {
            "time": 1533247669,
            "value": 3.4124999999999996
        },
        {
            "time": 1533247826,
            "value": 3.3149999999999995
        },
        {
            "time": 1533247982,
            "value": 3.4124999999999996
        },
        {
            "time": 1533248139,
            "value": 3.4124999999999996
        },
        {
            "time": 1533248295,
            "value": 3.4124999999999996
        },
        {
            "time": 1533248452,
            "value": 3.4124999999999996
        },
        {
            "time": 1533248608,
            "value": 3.4124999999999996
        },
        {
            "time": 1533248765,
            "value": 3.4124999999999996
        },
        {
            "time": 1533248921,
            "value": 3.4124999999999996
        },
        {
            "time": 1533249078,
            "value": 3.4124999999999996
        },
        {
            "time": 1533249234,
            "value": 3.4124999999999996
        },
        {
            "time": 1533249391,
            "value": 3.4124999999999996
        },
        {
            "time": 1533249547,
            "value": 3.4124999999999996
        },
        {
            "time": 1533249704,
            "value": 3.4124999999999996
        },
        {
            "time": 1533249860,
            "value": 3.4124999999999996
        },
        {
            "time": 1533250017,
            "value": 3.4124999999999996
        },
        {
            "time": 1533250173,
            "value": 3.4124999999999996
        },
        {
            "time": 1533250330,
            "value": 3.4124999999999996
        },
        {
            "time": 1533250486,
            "value": 3.4124999999999996
        },
        {
            "time": 1533250643,
            "value": 3.4124999999999996
        },
        {
            "time": 1533250800,
            "value": 3.4124999999999996
        },
        {
            "time": 1533250956,
            "value": 3.4124999999999996
        },
        {
            "time": 1533251113,
            "value": 3.4124999999999996
        },
        {
            "time": 1533251269,
            "value": 3.3149999999999995
        },
        {
            "time": 1533251426,
            "value": 3.3149999999999995
        },
        {
            "time": 1533251582,
            "value": 3.3149999999999995
        },
        {
            "time": 1533251739,
            "value": 3.3149999999999995
        },
        {
            "time": 1533251895,
            "value": 3.3149999999999995
        },
        {
            "time": 1533252052,
            "value": 3.3149999999999995
        },
        {
            "time": 1533252208,
            "value": 3.3149999999999995
        },
        {
            "time": 1533252365,
            "value": 3.3149999999999995
        },
        {
            "time": 1533252521,
            "value": 3.3149999999999995
        },
        {
            "time": 1533252678,
            "value": 3.4124999999999996
        },
        {
            "time": 1533252834,
            "value": 3.4124999999999996
        },
        {
            "time": 1533252991,
            "value": 3.51
        },
        {
            "time": 1533253147,
            "value": 3.51
        },
        {
            "time": 1533253304,
            "value": 3.51
        },
        {
            "time": 1533253460,
            "value": 3.51
        },
        {
            "time": 1533253617,
            "value": 3.51
        },
        {
            "time": 1533253773,
            "value": 3.6075
        },
        {
            "time": 1533253930,
            "value": 3.6075
        },
        {
            "time": 1533254086,
            "value": 3.6075
        },
        {
            "time": 1533254243,
            "value": 3.6075
        },
        {
            "time": 1533254400,
            "value": 3.6075
        },
        {
            "time": 1533254556,
            "value": 3.6075
        },
        {
            "time": 1533254713,
            "value": 3.6075
        },
        {
            "time": 1533254869,
            "value": 3.6075
        },
        {
            "time": 1533255026,
            "value": 3.6075
        },
        {
            "time": 1533255182,
            "value": 3.6075
        },
        {
            "time": 1533255339,
            "value": 3.6075
        },
        {
            "time": 1533255495,
            "value": 3.6075
        },
        {
            "time": 1533255652,
            "value": 3.6075
        },
        {
            "time": 1533255808,
            "value": 3.6075
        },
        {
            "time": 1533255965,
            "value": 3.51
        },
        {
            "time": 1533256121,
            "value": 3.51
        },
        {
            "time": 1533256278,
            "value": 3.6075
        },
        {
            "time": 1533256434,
            "value": 3.6075
        },
        {
            "time": 1533256591,
            "value": 3.51
        },
        {
            "time": 1533256747,
            "value": 3.6075
        },
        {
            "time": 1533256904,
            "value": 3.6075
        },
        {
            "time": 1533257060,
            "value": 3.6075
        },
        {
            "time": 1533257217,
            "value": 3.6075
        },
        {
            "time": 1533257373,
            "value": 3.6075
        },
        {
            "time": 1533257530,
            "value": 3.51
        },
        {
            "time": 1533257686,
            "value": 3.51
        },
        {
            "time": 1533257843,
            "value": 3.51
        },
        {
            "time": 1533258000,
            "value": 3.6075
        },
        {
            "time": 1533258156,
            "value": 3.6075
        },
        {
            "time": 1533258313,
            "value": 3.6075
        },
        {
            "time": 1533258469,
            "value": 3.6075
        },
        {
            "time": 1533258626,
            "value": 3.6075
        },
        {
            "time": 1533258782,
            "value": 3.705
        },
        {
            "time": 1533258939,
            "value": 3.6075
        },
        {
            "time": 1533259095,
            "value": 3.705
        },
        {
            "time": 1533259252,
            "value": 3.705
        },
        {
            "time": 1533259408,
            "value": 3.705
        },
        {
            "time": 1533259565,
            "value": 3.705
        },
        {
            "time": 1533259721,
            "value": 3.705
        },
        {
            "time": 1533259878,
            "value": 3.705
        },
        {
            "time": 1533260034,
            "value": 3.6075
        },
        {
            "time": 1533260191,
            "value": 3.6075
        },
        {
            "time": 1533260347,
            "value": 3.4124999999999996
        },
        {
            "time": 1533260504,
            "value": 3.4124999999999996
        },
        {
            "time": 1533260660,
            "value": 3.4124999999999996
        },
        {
            "time": 1533260817,
            "value": 3.4124999999999996
        },
        {
            "time": 1533260973,
            "value": 3.4124999999999996
        },
        {
            "time": 1533261130,
            "value": 3.3149999999999995
        },
        {
            "time": 1533261286,
            "value": 3.3149999999999995
        },
        {
            "time": 1533261443,
            "value": 3.4124999999999996
        },
        {
            "time": 1533261600,
            "value": 3.3149999999999995
        },
        {
            "time": 1533261756,
            "value": 3.4124999999999996
        },
        {
            "time": 1533261913,
            "value": 3.3149999999999995
        },
        {
            "time": 1533262069,
            "value": 3.3149999999999995
        },
        {
            "time": 1533262226,
            "value": 3.51
        },
        {
            "time": 1533262382,
            "value": 3.6075
        },
        {
            "time": 1533262539,
            "value": 3.6075
        },
        {
            "time": 1533262695,
            "value": 3.705
        },
        {
            "time": 1533262852,
            "value": 3.705
        },
        {
            "time": 1533263008,
            "value": 3.8999999999999995
        },
        {
            "time": 1533263165,
            "value": 4.095
        },
        {
            "time": 1533263321,
            "value": 4.289999999999999
        },
        {
            "time": 1533263478,
            "value": 4.484999999999999
        },
        {
            "time": 1533263634,
            "value": 4.68
        },
        {
            "time": 1533263791,
            "value": 4.875
        },
        {
            "time": 1533263947,
            "value": 4.875
        },
        {
            "time": 1533264104,
            "value": 5.07
        },
        {
            "time": 1533264260,
            "value": 5.265
        },
        {
            "time": 1533264417,
            "value": 5.3625
        },
        {
            "time": 1533264573,
            "value": 5.557499999999999
        },
        {
            "time": 1533264730,
            "value": 5.46
        },
        {
            "time": 1533264886,
            "value": 5.46
        },
        {
            "time": 1533265043,
            "value": 5.265
        },
        {
            "time": 1533265200,
            "value": 5.1675
        },
        {
            "time": 1533265356,
            "value": 4.9725
        },
        {
            "time": 1533265513,
            "value": 4.875
        },
        {
            "time": 1533265669,
            "value": 4.5825
        },
        {
            "time": 1533265826,
            "value": 4.387499999999999
        },
        {
            "time": 1533265982,
            "value": 4.1925
        },
        {
            "time": 1533266139,
            "value": 3.9974999999999996
        },
        {
            "time": 1533266295,
            "value": 3.9974999999999996
        },
        {
            "time": 1533266452,
            "value": 3.8999999999999995
        },
        {
            "time": 1533266608,
            "value": 3.8024999999999998
        },
        {
            "time": 1533266765,
            "value": 3.705
        },
        {
            "time": 1533266921,
            "value": 3.6075
        },
        {
            "time": 1533267078,
            "value": 3.51
        },
        {
            "time": 1533267234,
            "value": 3.51
        },
        {
            "time": 1533267391,
            "value": 3.705
        },
        {
            "time": 1533267547,
            "value": 3.705
        },
        {
            "time": 1533267704,
            "value": 3.705
        },
        {
            "time": 1533267860,
            "value": 3.8999999999999995
        },
        {
            "time": 1533268017,
            "value": 3.8999999999999995
        },
        {
            "time": 1533268173,
            "value": 4.095
        },
        {
            "time": 1533268330,
            "value": 4.095
        },
        {
            "time": 1533268486,
            "value": 4.289999999999999
        },
        {
            "time": 1533268643,
            "value": 4.484999999999999
        },
        {
            "time": 1533268800,
            "value": 4.5825
        },
        {
            "time": 1533268956,
            "value": 4.68
        },
        {
            "time": 1533269113,
            "value": 4.875
        },
        {
            "time": 1533269269,
            "value": 5.07
        },
        {
            "time": 1533269426,
            "value": 5.265
        },
        {
            "time": 1533269582,
            "value": 5.3625
        },
        {
            "time": 1533269739,
            "value": 5.557499999999999
        },
        {
            "time": 1533269895,
            "value": 5.557499999999999
        },
        {
            "time": 1533270052,
            "value": 5.654999999999999
        },
        {
            "time": 1533270208,
            "value": 5.654999999999999
        },
        {
            "time": 1533270365,
            "value": 5.7524999999999995
        },
        {
            "time": 1533270521,
            "value": 5.85
        },
        {
            "time": 1533270678,
            "value": 5.7524999999999995
        },
        {
            "time": 1533270834,
            "value": 5.654999999999999
        },
        {
            "time": 1533270991,
            "value": 5.46
        },
        {
            "time": 1533271147,
            "value": 5.265
        },
        {
            "time": 1533271304,
            "value": 5.3625
        },
        {
            "time": 1533271460,
            "value": 5.265
        },
        {
            "time": 1533271617,
            "value": 5.07
        },
        {
            "time": 1533271773,
            "value": 5.07
        },
        {
            "time": 1533271930,
            "value": 5.135000000000002
        },
        {
            "time": 1533272086,
            "value": 4.550000000000002
        },
        {
            "time": 1533272243,
            "value": 4.550000000000002
        }
    ],
    "endTime": 1533272400,
    "startTime": 1533243600
}
```