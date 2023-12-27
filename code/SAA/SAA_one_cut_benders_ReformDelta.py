from DST4ATM.head import compute_parameters, get_random
import numpy as np
import gurobipy as gp
from gurobipy import GRB

import warnings

# 计时

warnings.filterwarnings('ignore')
# modeltype='DRO'
modeltype = 'SAA'
ub_ua = 800
lb_ua = 500
ub_ud = 300
lb_ud = 0
timerange = 300

epsilon = 10
k = 2

randtype = 'uni'
p1a = 800
p2a = 1600

p1d = 600
p2d = 800
# randtype = 'norm'
# p1 = 0
# p2 = 30
weight = 0.2
S = 5
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

mp = gp.Model("MP")
mp.setParam('LazyConstraints', 1)

t = mp.addVars(ALL, vtype=GRB.CONTINUOUS, name="t")
y = mp.addVars(ALL, R, vtype=GRB.BINARY, name="y")
z = mp.addVars(ALL, ALL, vtype=GRB.BINARY, name="z")
q = mp.addVar(vtype=GRB.CONTINUOUS, lb=0, name="q")
xmax = mp.addVars(ALL, vtype=GRB.CONTINUOUS, name="xmax")
xmax2 = mp.addVar(vtype=GRB.CONTINUOUS, name="xmax2")
xmax3 = mp.addVar(vtype=GRB.CONTINUOUS, name="xmax3")
xmax4 = mp.addVar(vtype=GRB.CONTINUOUS, name="xmax4")
alpha = mp.addVars(ALL, vtype=GRB.CONTINUOUS, lb=0, name="alpha")
beta = mp.addVars(ALL, vtype=GRB.CONTINUOUS, lb=0, name="beta")
Zmax = mp.addVar(vtype=GRB.CONTINUOUS, name="Zmax")
Zmin = mp.addVar(vtype=GRB.CONTINUOUS, name="Zmin")
delta = mp.addVars(S, ALL, ALL, vtype=GRB.BINARY, name="delta")

for i in ALL:
    if ac_list[i].ad == 'd':
        mp.addConstr(y[i, 2] == 0)
    else:
        mp.addConstr(y[i, 1] == 0)
        if ac_list[i].entryfix == 'P270' or ac_list[i].entryfix == 'IDUMA':
            mp.addConstr(y[i, 0] == 0)
        if ac_list[i].entryfix == 'GYA':
            mp.addConstr(y[i, 2] == 0)

mp.addConstrs((t[i] >= lb_t[i] for i in ALL), "lb")
mp.addConstrs((t[i] <= ub_t[i] for i in ALL), "ub")
# 唯一性约束
mp.addConstrs((gp.quicksum(y[i, r] for r in R) == 1 for i in ALL), "unique1")
mp.addConstrs(z[i, j] >= y[i, r] + y[j, r] - 1 for i in ALL for j in ALL for r in R if i != j)
# mp.addConstrs(z[i, j] ==gp.quicksum( y[i, r] * y[j, r] for r in R) for i in ALL for j in ALL  if i > j)

mp.addConstrs(z[i, j] == z[j, i] for i in ALL for j in ALL if i > j)
mp.addConstrs(delta[s, j, i] + delta[s, i, j] == 1 for s in S for i in ALL for j in ALL if i > j)

# obj
mp.addConstrs((Zmax >= gp.quicksum(y[i, r] for i in ALL)) for r in R)
mp.addConstrs((Zmin <= gp.quicksum(y[i, r] for i in ALL)) for r in R)

mp.addConstrs(alpha[i] >= target[i] - t[i] for i in ALL)
mp.addConstrs(beta[i] >= t[i] - target[i] for i in ALL)
mp.addConstrs(alpha[i] <= target[i] - lb_t[i] for i in ALL)
mp.addConstrs(beta[i] <= ub_t[i] - target[i] for i in ALL)
mp.addConstrs(t[i] == target[i] - alpha[i] + beta[i] for i in ALL)
mp.addConstrs((xmax[i] == alpha[i] + beta[i]) for i in ALL)

obj1 = sum(xmax[i] for i in ALL)
obj2 = Zmax - Zmin
# obj3 = gp.quicksum(r[s, i] - x[s, i] for s in S for i in ALL)
mp.setObjective(obj2 + obj1 * weight + 1 / len(S) * q, GRB.MINIMIZE)
# time limit
mp.Params.TimeLimit = 1200


def makesp(sol_dict={}, relax=True):
    sp = gp.Model("sp")
    sp.setParam('PreCrush', 1)
    sp.setParam('InfUnbdInfo', 1)
    sp.setParam('OutputFlag', 0)

    x = sp.addVars(S, ALL, vtype=GRB.CONTINUOUS, name="x")
    r = sp.addVars(S, ALL, vtype=GRB.CONTINUOUS, name="r")

    sp.addConstrs(x[s, i] == sol_dict[t[i]] + ds[s][i] for s in S for i in ALL)
    sp.addConstrs(r[s, i] - x[s, i] >= lb_u[i] for s in S for i in ALL)
    sp.addConstrs(x[s, i] - r[s, i] >= -ub_u[i] for s in S for i in ALL)
    sp.addConstrs(r[s, j] - r[s, i] >= sep[i, j] * sol_dict[z[i, j]] - 10000 * sol_dict[delta[s, j, i]]
                  for s in S for i in ALL for j in ALL if i != j)

    sp.setObjective(gp.quicksum(r[s, i] - x[s, i] for s in S for i in ALL), GRB.MINIMIZE)

    sp.optimize()
    return sp


def mycallback(model, where):
    if where == GRB.Callback.MIPSOL:
        # 获取主问题的当前整数解
        curr_sol = model.cbGetSolution(model._vars)
        sol_dict = {var: curr_sol[i] for i, var in enumerate(model._vars)}
        print('MP status:', model.status)
        # 构建并求解子问题
        sp = makesp(sol_dict)
        rhs = ([t[i] + ds[s][i] for s in S for i in ALL] +
               [lb_u[i] for s in S for i in ALL] +
               [-ub_u[i] for s in S for i in ALL] +
               [sep[i, j] * z[i, j] - 10000 * delta[s, j, i]for s in S for i in ALL for j in ALL if i != j])
        rhs = np.array(rhs)
        # 根据子问题的结果向主问题添加惰性约束（如适用）and bsp.status == GRB.OPTIMAL
        if sp.status == GRB.OPTIMAL:
            print('optimal cut')

            duals = [constr.Pi for constr in sp.getConstrs()]
            duals = np.array(duals)
            # 遍历约束和对偶变量
            optcut = sum(duals * rhs)
            model.cbLazy(optcut <= q)
            # in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
        elif sp.status == GRB.INFEASIBLE or sp.status == GRB.INF_OR_UNBD:
            print('feasible cut,sp.status:', sp.status)

            farkas_duals = [constr.FarkasDual for constr in sp.getConstrs()]
            optcut = sum(farkas_duals * rhs)
            model.cbLazy(optcut <= 0)

        else:
            print('error')


# 注册回调函数
mp._vars = list(t.values()) + list(y.values()) + list(z.values()) + list(delta.values())

mp.update()
print(mp.NumConstrs)

mp.optimize(mycallback)




def postbsp(t, z,delta):
    sp = gp.Model("sp")
    # sp.setParam('PreCrush', 1)
    # sp.setParam('InfUnbdInfo', 1)
    # sp.setParam('OutputFlag', 0)

    x = sp.addVars(S, ALL, vtype=GRB.CONTINUOUS, name="x")
    r = sp.addVars(S, ALL, vtype=GRB.CONTINUOUS, name="r")

    sp.addConstrs(x[s, i] == t[i] + ds[s][i] for s in S for i in ALL)
    # sp.addConstrs(delta[s, j, i] + delta[s, i, j] == 1 for s in S for i in ALL for j in ALL if i > j)
    sp.addConstrs(r[s, i] - x[s, i] >= lb_u[i] for s in S for i in ALL)
    sp.addConstrs(x[s, i] - r[s, i] >= -ub_u[i] for s in S for i in ALL)

    sp.addConstrs(
        r[s, j] - r[s, i] + delta[s, j, i] * 10000 >= sep[i, j] * z[i, j] for s in S for i in ALL for j in
        ALL if i != j)
    sp.update()
    num_constrs1 = sp.NumConstrs
    sp.setObjective(gp.quicksum(r[s, i] - x[s, i] for s in S for i in ALL), GRB.MINIMIZE)

    sp.optimize()
    X = np.array([[x[s, i].x for i in ALL] for s in S])
    RR = np.array([[r[s, i].x for i in ALL] for s in S])
    return sp, X, RR


# # 获取t,y,x,r,delta
T = np.array([t[i].x for i in ALL])
Y = np.array([[y[i, r].x for r in R] for i in ALL])
Z = np.array([[z[i, j].x for j in ALL] for i in ALL])
DELTA = np.array([[[delta[s, i, j].x for j in ALL] for i in ALL] for s in S])

sp2, X, RR = postbsp(T, Z,DELTA)

TrueObj = mp.objVal - 1 / len(S) * q.x + 1 / len(S) * sp2.objVal
print('TrueObj:', TrueObj)
ALPHA = np.array([alpha[i].x for i in ALL])
BETA = np.array([beta[i].x for i in ALL])


