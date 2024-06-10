# -*- coding: utf-8 -*-

import os
import string
import math


EARTH_RADIUS = 6378.137  # 地球半径，单位为km


def coordinate2distance(lng1, lat1, lng2, lat2):
    '''
    计算两点间的经纬度坐标转换为东西向、南北向和直线距离
    '''
    #perimetre = 2 * math.pi * radius
    perimetre = 2 * math.pi * EARTH_RADIUS
    distance_unit_lat = perimetre/360
    lat_diff = abs(lat2 - lat1)
    distance_lat = distance_unit_lat * lat_diff

    distance_unit_lng = perimetre * math.cos(math.pi*abs(lat1)/180)/360
    lng_diff = abs(lng2-lng1)
    distance_lng = distance_unit_lng * lng_diff

    distance = math.sqrt(math.pow(distance_lat, 2) +
                         math.pow(distance_lng, 2))

    return [distance_lat, distance_lng, distance]


def rad(d=0.0):
    f = 180.0
    return d * math.pi / f


def get_distance(lng1, lat1, lng2, lat2):
    '''
    计算两点间距离, 这个算法对现在坐标系不准确
    '''
    lat_diff = lat1 - lat2
    lng_diff = rad(lng1) - rad(lng2)
    diff = 2 * math.asin(math.sqrt(math.pow(math.sin(lat_diff / 2), 2)
                                   + math.cos(lat1) * math.cos(lat2) * math.pow(math.sin(lng_diff / 2), 2)))
    dis = diff * EARTH_RADIUS
    dis = round(dis * 10000) / 10000
    return dis


def move_cordinate(lng, lat):
    '''
    坐标向右上角漂移大约300公里
    '''
    lng = float(lng) + 4.0
    lat = float(lat) + 2.5
    if lng >= 180.0:
        lng -= 4.0

    if lat >= 90.0:
        lat -= 2.5
    return lng, lat


if __name__ == '__main__':

    lat1 = 73.0470458
    lng1 = 26.2808841

    #lat2 = 77.0844152
    #lng2 = 28.5549767

    lat2 = 75.5470458
    lng2 = 30.2808841

    dis1, dis2, dis = coordinate2distance(lng1, lat1, lng2, lat2)
    print(dis1, dis2, dis)

    lng3, lat3 = move_cordinate(lng1, lat1)
    print(lng3, lat3)

    dis1, dis2, dis = coordinate2distance(lng1, lat1, lng2, lat2)
    print(dis1, dis2, dis)

    lat11 = 22.3876902
    lng11 = 114.1973341

    lat22 = 22.3877652
    lng22 = 114.1967807

    dis1, dis2, dis = coordinate2distance(lng11, lat11, lng22, lat22)

    print(dis1, dis2, dis, type(dis))

    distance = get_distance(lng1, lat1, lng2, lat2)
    print(distance)

    distance = get_distance(lng11, lat11, lng22, lat22)
    print(distance)
