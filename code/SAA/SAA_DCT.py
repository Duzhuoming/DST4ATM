from aircraft_rso import compute_parameters, saveres, drawres, get_random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import gurobipy as gp
from gurobipy import GRB

import warnings
from joblib import Parallel, delayed
# 计时
import time
warnings.filterwarnings('ignore')
# modeltype='DRO'
modeltype = 'SAA'
ub_ua = 800
lb_ua = 500
ub_ud = 300
lb_ud = 0
timerange = 3000

epsilon = 10
k = 1

randtype = 'uni'
p1a = 800
p2a = 1600

p1d = 600
p2d = 800
# randtype = 'norm'
# p1 = 0
# p2 = 30
weight=0.2
S = 10
seed=42


(ldte, ldtl, ldtt), (ete, etl, ett), (obte, obtt, obtl, utt), ac_list, sep, (A, D, R, ALL), df = compute_parameters(
    timerange)
lb_t, ub_t = np.concatenate((obte, ete)), np.concatenate((obtl, etl))
target = np.concatenate((obtt, ett))
sep *= k
ALL = range(ALL)





dsd = get_random(randtype, S, D, p1d, p2d, seed=seed)
dsa = get_random(randtype, S, A, p1a, p2a, seed=seed)
ds = np.concatenate((dsd, dsa), axis=1)
R = range(R)
S = range(S)

m = gp.Model("sync")
# m.setParam('OutputFlag', 0)

t = m.addVars(ALL, vtype=GRB.CONTINUOUS, name="t")
y = m.addVars(ALL, R, vtype=GRB.BINARY, name="y")
z = m.addVars(ALL, ALL, vtype=GRB.BINARY, name="z")
xmax = m.addVars(ALL, vtype=GRB.CONTINUOUS, name="xmax")
xmax2 = m.addVar(vtype=GRB.CONTINUOUS, name="xmax2")
xmax3 = m.addVar(vtype=GRB.CONTINUOUS, name="xmax3")
xmax4 = m.addVar(vtype=GRB.CONTINUOUS, name="xmax4")
alpha = m.addVars(ALL, vtype=GRB.CONTINUOUS, lb=0, name="alpha")
beta = m.addVars(ALL, vtype=GRB.CONTINUOUS, lb=0, name="beta")
Zmax = m.addVar(vtype=GRB.CONTINUOUS, name="Zmax")
Zmin = m.addVar(vtype=GRB.CONTINUOUS, name="Zmin")

x = m.addVars(S, ALL, vtype=GRB.CONTINUOUS, name="x")
r = m.addVars(S, ALL, vtype=GRB.CONTINUOUS, name="r")
delta = m.addVars(S, ALL, ALL, vtype=GRB.BINARY, name="delta")

for i in ALL:
    if ac_list[i].ad == 'd':
        m.addConstr(y[i, 2] == 0)
    else:
        m.addConstr(y[i, 1] == 0)
        if ac_list[i].entryfix == 'P270' or ac_list[i].entryfix == 'IDUMA':
            m.addConstr(y[i, 0] == 0)
        if ac_list[i].entryfix == 'GYA':
            m.addConstr(y[i, 2] == 0)

m.addConstrs((t[i] >= lb_t[i] for i in ALL), "lb")
m.addConstrs((t[i] <= ub_t[i] for i in ALL), "ub")
# 唯一性约束
m.addConstrs((gp.quicksum(y[i, r] for r in R) == 1 for i in ALL), "unique1")
m.addConstrs(z[i, j] >= y[i, r] + y[j, r] - 1 for i in ALL for j in ALL for r in R if i != j)
# m.addConstrs(z[i, j] ==gp.quicksum( y[i, r] * y[j, r] for r in R) for i in ALL for j in ALL  if i > j)

m.addConstrs(z[i, j] == z[j, i] for i in ALL for j in ALL if i > j)
## stage 2
m.addConstrs(x[s, i] == t[i] + ds[s][i] for s in S for i in ALL)
for i in ALL:
    if ac_list[i].ad == 'a':
        m.addConstrs(x[s, i] + lb_ua <= r[s, i] for s in S)
        m.addConstrs(r[s, i] <= x[s, i] + ub_ua for s in S)
    elif ac_list[i].ad == 'd':
        m.addConstrs(x[s, i] + lb_ud <= r[s, i] for s in S)
        m.addConstrs(r[s, i] <= x[s, i] + ub_ud for s in S)
    else:
        print('error')
#
# for i in ALL:
#     for j in ALL:
#         if ac_list[i].entryfix =='LMN' and ac_list[j].entryfix =='LMN' :
#             m.addConstrs(r[s, j] >= r[s, i] + y[i,0]*(1-z[i, j]) * 240 - delta[s, j, i] * 10000 for s in S if i != j)
#         elif ac_list[i].entryfix =='P268' and ac_list[j].entryfix =='P268' :
#             m.addConstrs(r[s, j] >= r[s, i] + y[i,1]*(1-z[i, j]) * 240 - delta[s, j, i] * 10000 for s in S  if i != j)



m.addConstrs(
    r[s, j] >= r[s, i] + z[i, j] * sep[i, j] - delta[s, j, i] * 10000 for s in S for i in ALL for j in ALL if
    i != j)
m.addConstrs(delta[s, j, i] + delta[s, i, j] == 1 for s in S for i in ALL for j in ALL if i > j)

# obj
m.addConstrs((Zmax >= gp.quicksum(y[i, r] for i in ALL)) for r in R)
m.addConstrs((Zmin <= gp.quicksum(y[i, r] for i in ALL)) for r in R)

m.addConstrs(alpha[i] >= target[i] - t[i] for i in ALL)
m.addConstrs(beta[i] >= t[i] - target[i] for i in ALL)
m.addConstrs(alpha[i] <= target[i] - lb_t[i] for i in ALL)
m.addConstrs(beta[i] <= ub_t[i] - target[i] for i in ALL)
m.addConstrs(t[i] == target[i] - alpha[i] + beta[i] for i in ALL)
m.addConstrs((xmax[i] == alpha[i] + beta[i]) for i in ALL)

obj1 = sum(xmax[i] for i in ALL)
obj2 = Zmax - Zmin
obj3 = gp.quicksum(r[s, i] - x[s, i] for s in S for i in ALL)
m.setObjective(obj2 + obj1 * weight + 1 / len(S) * obj3, GRB.MINIMIZE)
# time limit
# m.Params.TimeLimit = 1200
# m.NumVars
# m.NumConstrs
m.optimize()

# 获取t,y,x,r,delta
T = np.array([t[i].x for i in ALL])
Y = np.array([[y[i, r].x for r in R] for i in ALL])
X = np.array([[x[s, i].x for i in ALL] for s in S])
RR = np.array([[r[s, i].x for i in ALL] for s in S])
DELTA = np.array([[[delta[s, i, j].x for j in ALL] for i in ALL] for s in S])
ALPHA = np.array([alpha[i].x for i in ALL])
BETA = np.array([beta[i].x for i in ALL])
# zij
Z = np.array([[z[i, j].x for j in ALL] for i in ALL])
# 获取obj
OBJ = m.objVal
obj1_value = obj1.getValue()
obj2_value = obj2.getValue()
obj3_value = 1 / len(S) *obj3.getValue()


gap = m.getAttr('MIPGap')
# m.NumVars
# m.NumConstrs





S=len(S)
# ddf=pd.DataFrame([obj1_value,obj2_value,obj3_value,gap,t]).T
# ddf.to_csv(fr'D:\nuaadzm\PycharmProjects\detour\rso_model3\saa\ans3.csv')
dsd = get_random(randtype, S, D, p1d, p2d, seed=42)
dsa = get_random(randtype, S, A, p1a, p2a, seed=42)
ds = np.concatenate((dsd, dsa), axis=1)
ac_list, df = saveres(D, A, ac_list, T, Y, X, RR, S, DELTA, ds, df, k)
drawres(D, A, ac_list, S, obte, obtl, ete, etl)


import gurobipy as gp

# Assuming you have a Gurobi model named 'model'
# Initialize flags
is_mip = False
is_qcp = False

# Check for integer or binary variables
if any(var.vType != gp.GRB.CONTINUOUS for var in m.getVars()):
    is_mip = True

# Check for quadratic constraints
if m.getQConstrs():
    is_qcp = True

# Determine the type of model
if is_mip and is_qcp:
    print("This is a Mixed-Integer Quadratically Constrained Program (MIQCP)")
elif is_mip:
    print("This is a Mixed-Integer Linear Program (MILP)")
elif is_qcp:
    print("This is a Quadratically Constrained Program (QCP)")
else:
    print("This is a Linear Program (LP)")