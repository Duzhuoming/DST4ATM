import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


# 读11、12月摘要
# 摘要abs_efeedadsb_supDEP/ARR.csv：
# 0 A列-航班号
# 1	B列-机型
# 2	C列-尾流等级
# 3	D列-起飞机场
# 4	E列-着陆机场
# 5	F列-轨迹起始时间
# 6	G列-轨迹终止时间
# 7	H列-eto
# 8	I列-ato
# 9	J列-eta
# 10	K列-轨迹标号
# 11	L列-跑道
# 12	M列-进离场类型（DEP/ARR）
# 13 	N列-轨迹起始点的经度
# 14	O列-轨迹起始点的纬度
# 15	P列-轨迹终止点的经度
# 16	Q列-轨迹起始点的纬度
# 17	R列-飞行时间（s）
# 18	S列-飞行距离（km）
# 19	T列-进离场类型（进场：0 离场：1）
# 20	U列-运行方向（北向：0 南向：1）
# 21	V列-离港点

# 轨迹数据命名为航班号（摘要项第A列）_轨迹唯一表示（摘要项第K列）：
# 0       	A列-时间戳
# 1	B列-高度
# 2	C列-经度
# 3	D列-纬度
# 4	E列-selected_altitude
# 5	F列-tas
# 6	G列-ias
# 7	H列-轨迹标号
# 8	I列-待飞距离（km）
# 9	J列-地速（km/h）
# 10	K列-航向
# "E:\离场_气象数据\201911\201911\abs_efeedadsb_supARR_201911.csv"

# 特征经纬度高度速度航向，小时数，uv风，数量

# sns.displot(arr11[17])
# plt.show()
# # 计算21列元素出现的次数
# print(arr11[21].value_counts())
# 轨迹所在文件夹：E:\离场_气象数据\201911\201911\201911arr
#

# 读取气象数据
def getuv(ac, tra):
    U = []
    V = []
    alt = tra[1][0]

    ARP = np.array([23.391388888888887, 113.30972222222222])
    # E:\离场_气象数据\meteorologicalData\meteorologicalData
    y = ac['time'].year
    m = ac['time'].month
    d = ac['time'].day
    h = ac['time'].hour
    if m != 11 and m != 12:
        print(m, d)
        m = 11
        d = 1

    if d < 10:
        d = '0' + str(d)
    if h < 6:
        h = '00'
    elif 6 <= h < 12:
        h = '06'
    elif 12 <= h < 18:
        h = '12'
    else:
        h = '18'
    # 备选高度：540，761，988，1457，1948，2466，3012，3590，4206，4865，5574，6343，7185，8117，
    array = [540, 761, 988, 1457, 1948, 2466, 3012, 3590, 4206, 4865, 5574, ]
    nums = [540, 1500, 2400, 3000, 3600, 4200, 5400]
    for num in nums:
        h1, h2 = find_interval(array, num)
        wea1 = pd.read_csv(fr'E:\离场_气象数据\meteorologicalData\meteorologicalData\{y}{m}{d}\{h}_level_{h1}.csv',
                           header=None)
        points = np.array(wea1.iloc[:, 0:2])
        values = np.array(wea1.iloc[:, 2:4])
        (u, v) = griddata(points, values, ARP, method='linear')[0]

        if h2 != None:
            wea2 = pd.read_csv(fr'E:\离场_气象数据\meteorologicalData\meteorologicalData\{y}{m}{d}\{h}_level_{h2}.csv',
                               header=None)
            points = np.array(wea2.iloc[:, 0:2])
            values = np.array(wea2.iloc[:, 2:4])
            (u2, v2) = griddata(points, values, ARP, method='linear')[0]
            u = u + (u2 - u) / (h2 - h1) * (num - h1)
            v = v + (v2 - v) / (h2 - h1) * (num - h1)
        U.append(u)
        V.append(v)
    U.extend(V)
    #     pdseire, index u1~u7,v`~v7
    s = pd.Series(U, index=[f'u{i}' for i in range(1, 8)] + [f'v{i}' for i in range(1, 8)])
    return s


def find_interval(arr, num):
    # 确保数组是递增的
    arr = sorted(arr)

    # 处理边界情况
    if num < arr[0]:
        return None, arr[0]
    elif num > arr[-1]:
        return arr[-1], None

    # 在数组中找到正确的区间
    for i in range(len(arr) - 1):
        if arr[i] == num:
            return arr[i], None
        elif arr[i] < num < arr[i + 1]:
            return arr[i], arr[i + 1]


arr11 = pd.read_csv(r"E:\离场_气象数据\201911\201911\abs_efeedadsb_supARR_201911.csv",header=None)
# arr11 = pd.read_excel(r'D:\nuaadzm\PycharmProjects\detour\database\新建文件夹\12adsbcdm\12arr.xlsx')
arr11[21] = arr11[21].str.replace('000_036_ATAGA', '1') \
    .replace('180_300_GAOYAO', '5') \
    .replace('089_112_P270', '3') \
    .replace('036_089_IGONO', '2') \
    .replace('112_180_IDUMA', '4') \
    .replace('300_360_P71', '6')
# 1是ATAGA，2是IGONO， 3是P270，4是IDUMA，5是GYA，6是P71
# name = ['ATAGA', 'IGONO', 'P270', 'IDUMA', 'GYA', 'P71']
arr11['time'] = pd.to_datetime(arr11[5], unit='s') #+ pd.Timedelta(hours=8)  # ,11月的时间戳被加过了8h
arr11['end_time'] = pd.to_datetime(arr11[6], unit='s') #+ pd.Timedelta(hours=8)
# 获取hour of day
arr11['hour'] = arr11['time'].dt.hour
# hour displot
sns.displot(arr11['hour'])
plt.show()
feature = arr11.loc[:, [0, 13, 14, 21, 'hour']]

label = arr11.iloc[:, 17]
#
# fea=[]
# ap=[]#arrival pressure
# Wea=[]
#
# for idx,ac in arr11[0:3].iterrows():
#     tra=pd.read_csv(fr'E:\离场_气象数据\201911\201911\201911arr\{ac[0]}_{ac[10]}.csv',header=None)
#     alt=tra[1][0]
#     fea.append([tra[10][0],tra[9][0],alt])#mh,gs,alt
#
#     ap.append(arr11[(arr11[6] > ac[5]) & (arr11[5] < ac[5])][21].value_counts())
#     wea=getuv(ac,tra)
#     Wea.append(wea)
# fea=pd.DataFrame(fea,columns=['mh','gs','alt'])
# ap = pd.DataFrame(ap).fillna(0).sort_index(axis=1).reset_index(drop=True)
# Wea=pd.DataFrame(Wea)
# feature=pd.concat([feature,fea,ap,Wea],axis=1)
#
# # 加入feature
# from joblib import Parallel, delayed
# import pandas as pd
#
from joblib import Parallel, delayed
import pandas as pd


def process_row(ac, arr11):
    # tra = pd.read_csv(fr'E:\离场_气象数据\201912arr\{ac[0]}_{ac[10]}.csv', header=None)
    # alt = tra[1][0]
    # fea = [tra[10][0], tra[9][0], alt]  # mh, gs, alt
    mt0 = arr11[(arr11['end_time'] < ac['time']) & (arr11['end_time'] > (ac['time'] + pd.Timedelta(minutes=-15)))][
        17].mean()

    mt1 = arr11[(arr11['end_time'] < ac['time']) & (arr11['end_time'] > (ac['time'] + pd.Timedelta(minutes=-30)))][
        17].mean()
    mt2 = arr11[(arr11['end_time'] < ac['time']) & (arr11['end_time'] > (ac['time'] + pd.Timedelta(minutes=-15))) & (
                arr11[21] == ac[21])][17].mean()

    mt3 = arr11[(arr11['end_time'] < ac['time']) & (arr11['end_time'] > (ac['time'] + pd.Timedelta(minutes=-30))) & (
                arr11[21] == ac[21])][17].mean()

    ap0 = arr11[(arr11[6] > ac[5]) & (arr11[5] < ac[5])][21].value_counts()
    # ap1= arr11[(arr11[6] > ac[5]) & (arr11[5] < ac[5])][21].value_counts()[ac[21]]

    ap2 = len(arr11[(arr11[6] > ac[5]) & (arr11[5] < ac[5])])
    mt = [mt0, mt1, mt2, mt3,ap2]

    ap = ap0
    # wea = getuv(ac, tra)

    return mt, ap  # , weafea,


results = Parallel(n_jobs=-1)(delayed(process_row)(ac, arr11) for idx, ac in arr11.iterrows())

# 从results中提取fea, ap, Wea
# fea_list, ap_list, Wea_list = zip(*results)
mt_list, ap_list = zip(*results)

# 转换为DataFrame
# fea_df = pd.DataFrame(fea_list, columns=['mh', 'gs', 'alt'])
mt_df = pd.DataFrame(mt_list, columns=['mt15', 'mt30', 'mtef15', 'mtef30','aptot']).fillna(0)
ap_df = pd.DataFrame(ap_list).fillna(0).sort_index(axis=1).reset_index(drop=True)
# Wea_df = pd.DataFrame(Wea_list)

# 合并特征
# feature = pd.concat([feature, fea_df, ap_df, Wea_df], axis=1)
feature = pd.concat([feature, mt_df, ap_df], axis=1)

feature['apef'] = feature.apply(lambda row: row[row[21]], axis=1)

#
allset = pd.concat([feature, label], axis=1)
allset['date'] = arr11['time']
allset['rwy']=arr11[11]
# 时间增序
allset = allset.sort_values(by='date')

# rename = {0:'callsign',13:'lon',14:'lat',21:'entryfix',17:'flttime'}
# allset.rename(columns=rename,inplace=True)
# allset.to_csv(r'D:\nuaadzm\PycharmProjects\detour\ML\set12.csv', index=False)
#
# sns.boxplot(x="variable", y="value", data=pd.melt(feature[[f'u{i}' for i in range(1,8)]+[f'v{i}' for i in range(1,8)]]))
#
# # 显示图形
# plt.show()
# arr11.to_excel(r'D:\nuaadzm\PycharmProjects\detour\ML\arrtemp.xlsx', index=False)