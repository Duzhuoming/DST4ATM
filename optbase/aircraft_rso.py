
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from .fltpred import *
from pathlib import Path
current_directory = Path().cwd()

parent_directory = current_directory.parent.parent.parent

db_cdo_gdsj = pd.read_csv(fr'{parent_directory}\LocalData\sa_cdo_gdsj.csv', header=None)

map = pd.read_excel(fr'{parent_directory}\LocalData\gatemap_enhance.xlsx')
# feaset=pd.read_excel(fr'{parent_directory}\LocalData\allset1112.xlsx')

# 1st column type: str
map['gate'] = map['gate'].astype(str)

tb = 1575360000 - 1000


# 对应
# rwy  :01 02L 02R
# index:0  1   2

class Aircraft():
    # SI unit
    def __init__(self, callsign, sepclass, ad, entryfix, time, init_entrytime, ind, gate,
                 init_pushback_time):
        self.callsign = callsign
        self.sepclass = sepclass
        self.ad = ad
        self.entryfix = entryfix
        self.ymd=time
        self.gate = gate
        self.visio = map['visio'][map['gate'] == self.gate].values[0]
        self.init_entrytime = init_entrytime

        self.index = ind
        self.init_pushback_time = init_pushback_time

        self.t = []
        self.r = []
        self.x = []
        self.seq = []
        self.septime = []

        self.obte = None
        self.obtl = None
        self.obtt = None

        self.ldte = None
        self.ldtl = None
        self.ldtt = None

        self.utt = None
        self.add = None

    def computeflight(self):
        if self.ad == 'a':
            # self.getfea()
            t0 = self.init_entrytime
            eldt2 = 0
            lldt2 = 0
            tldt2 = 0
            eldt1 = 0
            lldt1 = 0
            tldt1 = 0
            if self.entryfix not in ['P270', 'IDUMA']:
                rwy = '01'
                eldt1 = db_cdo_gdsj[(db_cdo_gdsj[0] == self.callsign) & (db_cdo_gdsj[2] == rwy) & (
                        db_cdo_gdsj[3] == 10)].dropna(
                    axis=1, how='all').iloc[:, 4:].values.tolist()[0][-1]
                lldt1 = db_cdo_gdsj[(db_cdo_gdsj[0] == self.callsign) & (db_cdo_gdsj[2] == rwy) & (
                        db_cdo_gdsj[3] == 1)].dropna(
                    axis=1, how='all').iloc[:, 4:].values.tolist()[0][-1]
                tldt1 = db_cdo_gdsj[(db_cdo_gdsj[0] == self.callsign) & (db_cdo_gdsj[2] == rwy) & (
                        db_cdo_gdsj[3] == 5)].dropna(
                    axis=1, how='all').iloc[:, 4:].values.tolist()[0][-1]
                if lldt1 - eldt1 < 60:
                    lldt1 += 10
                    self.add = 10
                    print(self.index, self.entryfix, '<60,add60')

                elif lldt1 - eldt1 < 100:
                    lldt1 += 30
                    self.add = 30
                    print(self.index, self.entryfix, '<100,add30')
            if self.entryfix != 'GYA':
                rwy = '02R'
                eldt2 = db_cdo_gdsj[(db_cdo_gdsj[0] == self.callsign) & (db_cdo_gdsj[2] == rwy) & (
                        db_cdo_gdsj[3] == 10)].dropna(
                    axis=1, how='all').iloc[:, 4:].values.tolist()[0][-1]
                lldt2 = db_cdo_gdsj[(db_cdo_gdsj[0] == self.callsign) & (db_cdo_gdsj[2] == rwy) & (
                        db_cdo_gdsj[3] == 1)].dropna(
                    axis=1, how='all').iloc[:, 4:].values.tolist()[0][-1]
                tldt2 = db_cdo_gdsj[(db_cdo_gdsj[0] == self.callsign) & (db_cdo_gdsj[2] == rwy) & (
                        db_cdo_gdsj[3] == 5)].dropna(
                    axis=1, how='all').iloc[:, 4:].values.tolist()[0][-1]
                if lldt2 - eldt2 < 60:
                    lldt2 += 10
                    self.add = 10
                    print(self.index, self.entryfix, '<60,add60')

                elif lldt2 - eldt2 < 100:
                    lldt2 += 30
                    self.add = 30
                    print(self.index, self.entryfix, '<100,add30')

            self.ldte = np.array([eldt1 + t0 - tb, 0, eldt2 + t0 - tb])
            self.ldtl = np.array([lldt1 + t0 - tb, 0, lldt2 + t0 - tb])
            self.ldtt = np.array([tldt1 + t0 - tb, 0, tldt2 + t0 - tb])
            self.ete = t0 - tb - 60
            self.ett = t0 - tb
            self.etl = t0 - tb + 600

        if self.ad == 'd':
            self.obte = self.init_pushback_time - tb - 60
            self.obtt = self.init_pushback_time - tb
            self.obtl = self.init_pushback_time - tb + 600

            utt1 = map['01d'][map['gate'] == self.gate].values[0]
            utt2 = map['02Ld'][map['gate'] == self.gate].values[0]
            self.utt = [utt1 * 60, utt2 * 60, 0]

    def getfea(self):
        con=(feaset['callsign']==self.callsign) &(feaset['date']==self.ymd)
        fea=feaset[con].iloc[:, 1:-2]
        lab=feaset[con].iloc[:, -2]
        self.fea=fea
        self.lab=lab
        fltid=fea.index[0]
        # fltid = 21730
        a,b= getfset(fltid)
        self.fset=(a,b)


def draw(numd, numa, ac_list, res=False):
    rind = {0: '01', 1: '02L', 2: '02R'}
    plt.figure(figsize=(10, 20))
    # ytick set int
    plt.yticks(np.arange(0, numd + numa, 1.0))
    for i, ac in enumerate(ac_list):
        if ac.ad == 'd':
            plt.plot([ac.obte, ac.obtt, ac.obtl], [i, i, i], color='k', marker='o', markersize=6,
                     mfc='none')
            plt.text(ac.obte, i - 0.1, ac.obte, rotation=-25, horizontalalignment='left',
                     verticalalignment='top', rotation_mode='anchor')
            plt.text(ac.obtt, i - 0.1, ac.obtt, rotation=-25, horizontalalignment='left',
                     verticalalignment='top', rotation_mode='anchor')
            plt.text(ac.obtl, i - 0.1, ac.obtl, rotation=-25, horizontalalignment='left',
                     verticalalignment='top', rotation_mode='anchor')
        if ac.ad == 'a':
            if ac.entryfix != 'P270' and ac.entryfix != 'IDUMA':
                plt.plot([ac.ldte[0], ac.ldtt[0], ac.ldtl[0]], [i, i, i], color='k', marker='o',
                         markersize=6, mfc='b')
            if ac.entryfix != 'GYA':
                plt.plot([ac.ldte[2], ac.ldtt[2], ac.ldtl[2]], [i - 0.2, i - 0.2, i - 0.2],
                         color='k', marker='o',
                         markersize=6, mfc='g')

    plt.tight_layout()
    # plt.savefig(fr'./timewindow.svg')
    plt.show()


def compute_parameters(dt):
    # Manually transcribe the data from the image into a DataFrame
    # Based on the visual inspection, we'll create a dictionary with the data

    # Define the column headers and row labels as seen in the image
    columns = ['A-H', 'A-M', 'A-L', 'D-H', 'D-M', 'D-L']
    rows = ['A-H', 'A-M', 'A-L', 'D-H', 'D-M', 'D-L']

    # Define the data as seen in the image
    data = [
        [96, 157, 207, 60, 60, 60],  # Data for A-H
        [60, 69, 123, 60, 60, 60],  # Data for A-M
        [60, 69, 82, 60, 60, 60],  # Data for A-L
        [60, 60, 60, 96, 120, 120],  # Data for D-H
        [60, 60, 60, 60, 60, 60],  # Data for D-M
        [60, 60, 60, 60, 60, 60]  # Data for D-L
    ]

    # Create a pandas DataFrame with the data
    sepclass = pd.DataFrame(data, index=rows, columns=columns)

    aclistsample = pd.read_csv(fr'{parent_directory}\LocalData\ad191203.csv')  # {datadate}
    ac_list = []
    cuttime = 1575360000

    A, D = 0, 0
    ind = []

    for index, row in aclistsample.iterrows():
        if (row['ad'] == 'a' and row['entrytime'] < (cuttime + dt)):
            A += 1
            ac_list.append(
                Aircraft(row['callsign'], row['class'], row['ad'], row['entryfix'], row['time'], row['entrytime'],
                         index, row['gate'], row['pushback_time']))
            ind.append(index)

        elif (row['ad'] == 'd' and row['pushback_time'] < (cuttime + dt)):
            ac_list.append(
                Aircraft(row['callsign'], row['class'], row['ad'], row['entryfix'], row['gaterwy'], row['entrytime'],
                         index, row['gate'], row['pushback_time']))
            ind.append(index)

            D += 1
    df = aclistsample.loc[ind]

    ldte = []
    ldtl = []
    ldtt = []
    ete = []
    ett = []
    etl = []
    obte = []
    obtt = []
    obtl = []
    utt = []
    zerotime = []
    for ac in ac_list:
        ac.computeflight()
        if ac.ad == 'a':
            ldte.append(ac.ldte)
            ldtl.append(ac.ldtl)
            ldtt.append(ac.ldtt)
            ete.append(ac.ete)
            ett.append(ac.ett)
            etl.append(ac.etl)
            zerotime.append(ac.init_entrytime - tb)
        else:
            obte.append(ac.obte)
            obtt.append(ac.obtt)
            obtl.append(ac.obtl)
            utt.append(ac.utt)
    ldte = np.array(ldte)
    ldtl = np.array(ldtl)
    ldtt = np.array(ldtt)
    ete = np.array(ete)
    ett = np.array(ett)
    etl = np.array(etl)
    obte = np.array(obte)
    obtl = np.array(obtl)
    utt = np.array(utt)
    zerotime = np.array(zerotime)
    R = 3
    ALL = len(ac_list)
    # draw(D, A, ac_list)
    sep = np.zeros((ALL, ALL))
    for idxi, aci in enumerate(ac_list):
        for idxj, acj in enumerate(ac_list):
            sepi = aci.ad.upper() + '-' + aci.sepclass.upper()
            sepj = acj.ad.upper() + '-' + acj.sepclass.upper()
            sep[idxi, idxj] = sepclass[sepi][sepj]
    df['ldte01'] = np.concatenate((np.zeros(D), ldte[:, 0]))
    # df['ldte02L']=np.concatenate((np.zeros(D),ldte[:,1]))
    df['ldte02R'] = np.concatenate((np.zeros(D), ldte[:, 2]))
    df['ldtl01'] = np.concatenate((np.zeros(D), ldtl[:, 0]))
    # df['ldtl02L']=np.concatenate((np.zeros(D),ldtl[:,1]))
    df['ldtl02R'] = np.concatenate((np.zeros(D), ldtl[:, 2]))
    df['ldtt01'] = np.concatenate((np.zeros(D), ldtt[:, 0]))
    # df['ldtt02L']=np.concatenate((np.zeros(D),ldtt[:,1]))
    df['ldtt02R'] = np.concatenate((np.zeros(D), ldtt[:, 2]))
    df['obte'] = np.concatenate((obte, np.zeros(A)))
    df['obtl'] = np.concatenate((obtl, np.zeros(A)))
    df['utt01'] = np.concatenate((utt[:, 0], np.zeros(A)))
    df['utt02L'] = np.concatenate((utt[:, 1], np.zeros(A)))
    df['zerotime'] = np.concatenate((np.zeros(D), zerotime))
    return (ldte, ldtl, ldtt), (ete, etl, ett), (obte, obtt,obtl, utt), ac_list, sep, (A, D, R, ALL), df


def saveres(D, A, ac_list, t, y, x, r, S, delta, ds, df, k):
    columns = ['A-H', 'A-M', 'A-L', 'D-H', 'D-M', 'D-L']
    rows = ['A-H', 'A-M', 'A-L', 'D-H', 'D-M', 'D-L']

    # Define the data as seen in the image
    data = [
        [96, 157, 207, 60, 60, 60],  # Data for A-H
        [60, 69, 123, 60, 60, 60],  # Data for A-M
        [60, 69, 82, 60, 60, 60],  # Data for A-L
        [60, 60, 60, 96, 120, 120],  # Data for D-H
        [60, 60, 60, 60, 60, 60],  # Data for D-M
        [60, 60, 60, 60, 60, 60]  # Data for D-L
    ]

    # Create a pandas DataFrame with the data
    sepclass = pd.DataFrame(data, index=rows, columns=columns)
    sepclass = sepclass * k
    df['01'] = y[:, 0]
    df['02L'] = y[:, 1]
    df['02R'] = y[:, 2]
    for idx, ac in enumerate(ac_list):
        ac.t = t[idx]

    for s in range(S):
        seq01 = []
        seq02L = []
        seq02R = []
        sept = []
        for idx, ac in enumerate(ac_list):
            ac.x.append(x[s][idx])
            ac.r.append(r[s][idx])

            rind = {0: '01', 1: '02L', 2: '02R'}

            ind = np.where(y[idx] > 0.5)[0][0]
            ac.rwy = rind[ind]
            Delta = delta[s].sum(axis=1)
            ac.seq = Delta[idx]

            if ac.rwy == '01':
                seq01.append((ac.seq, idx))
            elif ac.rwy == '02L':
                seq02L.append((ac.seq, idx))
            elif ac.rwy == '02R':
                seq02R.append((ac.seq, idx))
        seq01.sort(key=lambda x: x[0], reverse=True)
        seq02L.sort(key=lambda x: x[0], reverse=True)
        seq02R.sort(key=lambda x: x[0], reverse=True)
        for seqs in [seq01, seq02L, seq02R]:
            for i in range(len(seqs) - 1):
                aci = ac_list[seqs[i][1]]
                acj = ac_list[seqs[i + 1][1]]
                sepi = aci.ad.upper() + '-' + aci.sepclass.upper()
                sepj = acj.ad.upper() + '-' + acj.sepclass.upper()
                aci.septime .append( sepclass[sepi][sepj])
            acj.septime .append( 0)
        for ac in ac_list:
            sept.append(ac.septime[s])
        sept = np.array(sept)
        df[f'septime_s{s}'] = sept


    for s in range(S):
        df[f'ds_s{s}'] =  ds[s]

    for s in range(S):
        df[f'r-s_s{s}'] = r[s]-x[s]
    df['t']=t

    for s in range(S):
        df[f'x_s{s}'] = x[s]
    for s in range(S):
        df[f'r_s{s}'] = r[s]
    for s in range(S):
        Delta = delta[s].sum(axis=1)
        df[f'seq_s{s}'] = Delta

    return ac_list, df


def drawres(numd, numa, ac_list, S, obte, obtl, ete, etl):
    rind = {0: '01', 1: '02L', 2: '02R'}
    plt.figure(figsize=(10, 20))
    # ytick set int
    plt.yticks(np.arange(0, numd + numa, 1.0))
    lb = min(min(obte), min(ete))
    ub = max(max(obtl), max(etl)) + 1800
    for s in range(S):
        plt.text(lb, -1 - s * 3, f'S{s}')
        plt.plot([lb, ub], [-0.5 - s * 3, -0.5 - s * 3])

    for i, ac in enumerate(ac_list):
        if ac.rwy == '01':
            fc = 'b'
            y_pos = -1
        elif ac.rwy == '02L':
            fc = 'r'
            y_pos = -2
        elif ac.rwy == '02R':
            fc = 'g'
            y_pos = -3

        if ac.ad == 'd':
            obts=[ac.obte, ac.obtt, ac.obtl]
            plt.plot(obts, [i, i, i], color='k', marker='o', markersize=6,
                     mfc='none')
            for obt in obts:
                plt.text(obt, i - 0.1, obt, rotation=-25, horizontalalignment='left',
                         verticalalignment='top', rotation_mode='anchor')
            plt.text(400, i, f'{ac.index} {ac.rwy}  {ac.entryfix}')
            plt.scatter(ac.t, i, color='k', marker='o', s=90, facecolors='none')

            for s in range(S):

                plt.scatter(ac.x[s], i, color='k', marker='^', s=90, facecolors='none')
                plt.scatter(ac.r[s],i, color='k', marker='v', s=90, facecolors=fc)
                plt.scatter(ac.r[s], y_pos- s * 3, color='k', marker='v', s=90, facecolors=fc)
                plt.plot([ac.r[s], ac.r[s]], [y_pos- s * 3, i], color='k', ls='--')
                plt.plot([ac.r[s], ac.r[s] + ac.septime[s]], [y_pos - s * 3, y_pos - s * 3], color='k', linewidth=1)
                plt.text(ac.r[s], y_pos- s * 3, f'{ac.index},{round(ac.r[s])},{ac.septime[s]}', rotation=-15,
                         horizontalalignment='left',
                         verticalalignment='top', rotation_mode='anchor')


        if ac.ad == 'a':
            ets = [ac.ete, ac.ett, ac.etl]

            # if ac.entryfix != 'P270' and ac.entryfix != 'IDUMA':
            plt.plot(ets, [i, i, i], color='k', marker='o',
                         markersize=6, mfc='none')
            # if ac.entryfix != 'GYA':
            #     plt.plot([ac.ldte[2], ac.ldtt[2], ac.ldtl[2]], [i - 0.2, i - 0.2, i - 0.2],
            #              color='k', marker='o',
            #              markersize=6, mfc='g')
            plt.text(400, i, f'{ac.index} {ac.rwy}  ')

            # if ac.rwy == '01':
            plt.scatter(ac.t, i, color='k', marker='o', s=90, facecolors='none')
            for et in ets:
                plt.text(et, i , et, rotation=-25, horizontalalignment='left',
                         verticalalignment='top', rotation_mode='anchor')

            # elif ac.rwy == '02R':
            #     plt.scatter(ac.t, i - 0.2, color='k', marker='o', s=90, facecolors='none')
            #     plt.text(ac.ete, i - 0.3, ac.ete, rotation=-25, horizontalalignment='left',
            #              verticalalignment='top', rotation_mode='anchor')
            #     plt.text(ac.ett, i - 0.3, ac.ett, rotation=-25, horizontalalignment='left',
            #              verticalalignment='top', rotation_mode='anchor')
            #     plt.text(ac.etl, i - 0.3, ac.etl, rotation=-25, horizontalalignment='left',
            #              verticalalignment='top', rotation_mode='anchor')

            for s in range(S):
                plt.scatter(ac.r[s], y_pos- s * 3, color='k', marker='v', s=90, facecolors=fc)
                plt.scatter(ac.x[s], i, color='k', marker='^', s=90, facecolors='none')
                plt.plot([ac.x[s],ac.x[s]+500], [i,i], color='k',)

                plt.scatter(ac.r[s],i, color='k', marker='v', s=90, facecolors=fc)
                plt.plot([ac.r[s], ac.r[s]], [y_pos- s * 3, i], color='k', ls='--')

                plt.plot([ac.r[s], ac.r[s] + ac.septime[s]], [y_pos- s * 3, y_pos- s * 3], color='k', linewidth=1)
                plt.text(ac.r[s], y_pos- s * 3, f'{ac.index},{round(ac.r[s])},{ac.septime[s]}', rotation=-15,
                         horizontalalignment='left',
                         verticalalignment='top', rotation_mode='anchor')
    plt.tight_layout()
    # plt.savefig(fr'./timewindow.svg')
    plt.show()


def get_random(randtype, S, A, p1, p2,seed=42):
    np.random.seed(seed)

    if randtype == 'uni':
        return np.random.uniform(p1, p2, (S, A))
    elif randtype == 'norm':
        return np.random.normal(p1, p2, (S, A))

# def set_obj():
