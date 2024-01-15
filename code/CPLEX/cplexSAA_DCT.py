from DST4ATM.optbase.aircraft_rso import Parameters, saveres, drawres
import numpy as np
import warnings
from docplex.mp.model import Model

warnings.filterwarnings('ignore')
modeltype = 'SAA'
timerange =1000
S = 10

weight = 1
parm = Parameters(timerange, S)
parm.compute_parameters()
# 解析parm
ldte, ldtl, ldtt=parm.ldte, parm.ldtl, parm.ldtt
ete, etl, ett=parm.ete, parm.etl, parm.ett
obte, obtt, obtl, utt=parm.obte, parm.obtt, parm.obtl, parm.utt
ac_list, sep, A, D, R, ALL, df=parm.ac_list, parm.sep, parm.A, parm.D, parm.R, parm.ALL, parm.df
lb_t, ub_t,lb_u, ub_u  = parm.lb_t, parm.ub_t,parm.lb_u, parm.ub_u
target = parm.target
ds = parm.ds
S,k=parm.S,parm.k


m = Model("SAA",log_output=True)

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
m.add_kpi(weight * obj1 + obj2 ,'1st cost')

m.add_kpi(1 / len(S) * obj3,'2nd cost')

solution = m.solve()
m.report()
# m.report_kpis()
if solution:
    print("求解成功")
else:
    print("求解失败")
    print(m.solve_details)
# print gap
print("gap:", m.solve_details.mip_relative_gap)

T = np.array([t[i].sv for i in ALL])
Y = np.array([[y[i, r].sv for r in R] for i in ALL])
X = np.array([[x[s, i].sv for i in ALL] for s in S])
RR = np.array([[r[s, i].sv for i in ALL] for s in S])
DELTA = np.array([[[delta[s, j, i].sv for i in ALL] for j in ALL] for s in S])
ALPHA = np.array([alpha[i].sv for i in ALL])
BETA = np.array([beta[i].sv for i in ALL])
Z= np.array([z[i, j].sv for i in ALL for j in ALL])

S=len(S)

#
# ac_list, df = saveres(D, A, ac_list, T, Y, X, RR, S, DELTA, ds, df, k)
# drawres(D, A, ac_list, S, obte, obtl, ete, etl)
