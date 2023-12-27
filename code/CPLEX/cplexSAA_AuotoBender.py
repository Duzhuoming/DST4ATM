from DST4ATM.code.aircraft_rso import compute_parameters, saveres, drawres, get_random
import pandas as pd
import numpy as np

import warnings
from joblib import Parallel, delayed

from docplex.mp.model import Model

# 计时
import time

warnings.filterwarnings('ignore')
# modeltype='DRO'
modeltype = 'SAA'
ub_ua = 800
lb_ua = 500
ub_ud = 300
lb_ud = 0
timerange = 300
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
weight = 1
S = 10
seed = 42

(ldte, ldtl, ldtt), (ete, etl, ett), (obte, obtt, obtl, utt), ac_list, sep, (A, D, R, ALL), df = compute_parameters(
    timerange)
lb_t, ub_t = np.concatenate((obte, ete)), np.concatenate((obtl, etl))
lb_ud = np.array([lb_ud for i in range(D)])
ub_ud = np.array([ub_ud for i in range(D)])
lb_ua = np.array([lb_ua for i in range(A)])
ub_ua = np.array([ub_ua for i in range(A)])
lb_u, ub_u = np.concatenate((lb_ud, lb_ua)), np.concatenate((ub_ud, ub_ua))
target = np.concatenate((obtt, ett))
sep *= k
ALL = range(ALL)

dsd = get_random(randtype, S, D, p1d, p2d, seed=seed)
dsa = get_random(randtype, S, A, p1a, p2a, seed=seed)
ds = np.concatenate((dsd, dsa), axis=1)
R = range(R)
S = range(S)
# with Model("SAA",log_output=True) as m:

m = Model("SAA",log_output=True)
# m.parameters.mip.display.set(2)  # 2 是一种常用的中等详细级别
# m.parameters.mip.tolerances.mipgap = 0.055
t = m.continuous_var_dict(ALL, name="t")
y = m.binary_var_matrix(ALL, R, name="y")
z = m.binary_var_matrix(ALL, ALL, name="z")
xmax = m.continuous_var_dict(ALL, name="xmax")
xmax2 = m.continuous_var(name="xmax2")
xmax3 = m.continuous_var(name="xmax3")
xmax4 = m.continuous_var(name="xmax4")
alpha = m.continuous_var_dict(ALL, name="alpha")
beta = m.continuous_var_dict(ALL, name="beta")
Zmax = m.continuous_var(name="Zmax")
Zmin = m.continuous_var(name="Zmin")

x = m.continuous_var_matrix(S, ALL, name="x")
r = m.continuous_var_matrix(S, ALL, name="r")
delta = m.binary_var_cube(S, ALL, ALL, name="z")

# 创建三维二元变量立方体
# var_cube = {(i, j, k): mdl.binary_var(name=f"var_{i}_{j}_{k}")
#             for i in Dim1 for j in Dim2 for k in Dim3}

for i in ALL:
    if ac_list[i].ad == 'd':
        m.add_constraint(y[i, 2] == 0)
    else:
        m.add_constraint(y[i, 1] == 0)
        if ac_list[i].entryfix == 'P270' or ac_list[i].entryfix == 'IDUMA':
            m.add_constraint(y[i, 0] == 0)
        if ac_list[i].entryfix == 'GYA':
            m.add_constraint(y[i, 2] == 0)

m.add_constraints((t[i] >= lb_t[i] for i in ALL), "lb")
m.add_constraints((t[i] <= ub_t[i] for i in ALL), "ub")
m.add_constraints((m.sum(y[i, r] for r in R) == 1 for i in ALL), "unique1")
m.add_constraints(z[i, j] >= y[i, r] + y[j, r] - 1 for i in ALL for j in ALL for r in R if i != j)
# m.add_constraints(z[i, j] ==m.sum( y[i, r] * y[j, r] for r in R) for i in ALL for j in ALL  if i > j)
m.add_constraints(z[i, j] == z[j, i] for i in ALL for j in ALL if i > j)

# stage 2
m.add_constraints((x[s, i] == t[i] + ds[s, i] for s in S for i in ALL), "x")
m.add_constraints((r[s, i] - x[s, i] >= lb_u[i] for s in S for i in ALL), "lb_u")
m.add_constraints((x[s, i] - r[s, i] >= -ub_u[i] for s in S for i in ALL), "ub_u")
m.add_constraints(
    (r[s, j] >= r[s, i] + z[i, j] * sep[i, j] - delta[s, j, i] * 10000 for s in S for i in ALL for j in ALL if
     i != j), "Wake")
m.add_constraints((delta[s, j, i] + delta[s, i, j] == 1 for s in S for i in ALL for j in ALL if i > j), "delta")

m.add_constraints((xmax[i] == alpha[i] + beta[i]) for i in ALL)
m.add_constraints((alpha[i] >= target[i] - t[i] for i in ALL))
m.add_constraints((beta[i] >= t[i] - target[i] for i in ALL))
m.add_constraints((alpha[i] <= target[i] - lb_t[i] for i in ALL))
m.add_constraints((beta[i] <= ub_t[i] - target[i] for i in ALL))
m.add_constraints((t[i] == target[i] - alpha[i] + beta[i] for i in ALL))
# obj
m.add_constraints((Zmax >= m.sum(y[i, r] for i in ALL)) for r in R)
m.add_constraints((Zmin <= m.sum(y[i, r] for i in ALL)) for r in R)

obj1 = m.sum(xmax[i] for i in ALL)
obj2 = Zmax - Zmin
obj3 = m.sum(r[s, i] - x[s, i] for s in S for i in ALL)
m.minimize(weight * obj1 + obj2 + 1 / len(S) * obj3)
m.print_information()
m.add_kpi(obj3,'2nd cost')

solution = m.solve()
m.report()
m.report_kpis()
if solution:
    print("求解成功")
else:
    print("求解失败")
    print(m.solve_details)


T = np.array([t[i].sv for i in ALL])
Y = np.array([[y[i, r].sv for r in R] for i in ALL])
X = np.array([[x[s, i].sv for i in ALL] for s in S])
RR = np.array([[r[s, i].sv for i in ALL] for s in S])
DELTA = np.array([[[delta[s, j, i].sv for i in ALL] for j in ALL] for s in S])
ALPHA = np.array([alpha[i].sv for i in ALL])
BETA = np.array([beta[i].sv for i in ALL])
Z= np.array([z[i, j].sv for i in ALL for j in ALL])

S=len(S)

dsd = get_random(randtype, S, D, p1d, p2d, seed=42)
dsa = get_random(randtype, S, A, p1a, p2a, seed=42)
ds = np.concatenate((dsd, dsa), axis=1)
ac_list, df = saveres(D, A, ac_list, T, Y, X, RR, S, DELTA, ds, df, k)
drawres(D, A, ac_list, S, obte, obtl, ete, etl)
