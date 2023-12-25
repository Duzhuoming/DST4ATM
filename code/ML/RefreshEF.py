import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, sin, cos, degrees, atan2, pi, asin, sqrt
# from geopy import distance

def getDegree(latA, lonA, latB, lonB):
    """
    Args:
        point p1(latA, lonA)
        point p2(latB, lonB)
    Returns:
        bearing between the two GPS points,
        default: the basis of heading direction is north
    """
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng

#
# def calDist(a, b, c, d):
#     dist = distance.distance((a, b), (c, d)).km
#     return dist
#
#
# vcalDist = np.vectorize(calDist)


currentPath = r'E:\离场_气象数据\201911\201911'
abst = pd.read_csv(os.path.join(currentPath, 'abs_efeedadsb_supDEP_201911.csv'), header=None)
abst[21] = ''


ARP = np.array([23.391388888888887, 113.30972222222222])

'离港点'
for i in range(len(abst)):
    brng = getDegree(ARP[0], ARP[1], abst.iloc[i, 16], abst.iloc[i, 15])
    if brng > 33 and brng <= 133:
        abst.iloc[i, 21] = '033_133_A'
    elif brng > 133 and brng <= 192:
        abst.iloc[i, 21] = '133_192_B'
    elif brng > 192 and brng <= 270:
        abst.iloc[i, 21] = '192_270_C'
    elif brng > 270 and brng <= 327:
        abst.iloc[i, 21] = '270_327_D'
    elif brng > 327 and brng <= 350:
        abst.iloc[i, 21] = '327_350_E'
    else:
        abst.iloc[i, 21] = '350_033_F'

abst.to_csv(os.path.join(currentPath, 'abs_efeedadsb_supDEP_201911.csv'), header=False, index=False)



# plt.figure(1)
# np_wp1 = np.array(waypoint1)
# for i in np.unique(np_wp1):
#     i_abst = abst[np_wp1 == i, 15:17]
#     plt.plot(i_abst[:, 0], i_abst[:, 1], '.')

'进港点'
abst = pd.read_csv(os.path.join(currentPath, 'abs_efeedadsb_supARR_201911.csv'), header=None)
abst[21] = ''
for i in range(len(abst)):
    brng = getDegree(ARP[0], ARP[1], abst.iloc[i, 14], abst.iloc[i, 13])
    if brng > 0 and brng <= 36:
        abst.iloc[i, 21] = '000_036_ATAGA'
    elif brng > 36 and brng <= 89:
        abst.iloc[i, 21] = '036_089_IGONO'
    elif brng > 89 and brng <= 112:
        abst.iloc[i, 21] = '089_112_P270'
    elif brng > 112 and brng <= 180:
        abst.iloc[i, 21] = '112_180_IDUMA'
    elif brng > 180 and brng <= 300:
        abst.iloc[i, 21] = '180_300_GAOYAO'
    else:
        abst.iloc[i, 21] = '300_360_P71'


abst.to_csv(os.path.join(currentPath, 'abs_efeedadsb_supARR_201911.csv'), header=False, index=False)

# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot()
# ax1.set_xlabel('Longitude')  # 经度
# ax1.set_ylabel('Latitude')  # 维度
# ax1.set_aspect('equal', 'box')
#
# for i in filelist:
#     tra = pd.read_csv(os.path.join(dataPath, i), header=None)
#     tra = np.array(tra)
#     ax1.plot(tra[:, 2], tra[:, 3], c='red', alpha=0.1)