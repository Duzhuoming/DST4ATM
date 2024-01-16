import gurobipy as gp
from gurobipy import GRB
from DST4ATM.optbase.aircraft_rso import Parameters, saveres, drawres
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')
modeltype = 'SAA'
timerange = 1000
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

mp = gp.Model("MP")
mp.setParam('LazyConstraints', 1)
mp.setParam('OutputFlag', 0)



t = mp.addVars(ALL, vtype=GRB.CONTINUOUS, name="t")
y = mp.addVars(ALL, R, vtype=GRB.CONTINUOUS, name="y")
z = mp.addVars(ALL, ALL, vtype=GRB.CONTINUOUS, name="z")

q=mp.addVar(vtype=GRB.CONTINUOUS, lb=0, name="q")
xmax = mp.addVars(ALL, vtype=GRB.CONTINUOUS, name="xmax")
alpha = mp.addVars(ALL, vtype=GRB.CONTINUOUS, lb=0, name="alpha")
beta = mp.addVars(ALL, vtype=GRB.CONTINUOUS, lb=0, name="beta")
Zmax = mp.addVar(vtype=GRB.CONTINUOUS, name="Zmax")
Zmin = mp.addVar(vtype=GRB.CONTINUOUS, name="Zmin")

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

mp.addConstrs((Zmax >= gp.quicksum(y[i, r] for i in ALL)) for r in R)
mp.addConstrs((Zmin <= gp.quicksum(y[i, r] for i in ALL)) for r in R)

mp.addConstrs(alpha[i] >= target[i] - t[i] for i in ALL)
mp.addConstrs(beta[i] >= t[i] - target[i] for i in ALL)
mp.addConstrs(alpha[i] <= target[i] - lb_t[i] for i in ALL)
mp.addConstrs(beta[i] <= ub_t[i] - target[i] for i in ALL)
mp.addConstrs(t[i] == target[i] - alpha[i] + beta[i] for i in ALL)
mp.addConstrs((xmax[i] == alpha[i] + beta[i]) for i in ALL)

# obj
obj1 = sum(xmax[i] for i in ALL)
obj2 = Zmax - Zmin
# obj3 = gp.quicksum(r[s, i] - x[s, i] for s in S for i in ALL)
mp.setObjective(obj2 + obj1 * weight + 1 / len(S) * q, GRB.MINIMIZE)

def makesp(sol_dict,relax=True):
    sp = gp.Model("sp")
    sp.setParam('PreCrush', 1)
    sp.setParam('InfUnbdInfo', 1)
    sp.setParam('OutputFlag', 0)

    x = sp.addVars(S, ALL, vtype=GRB.CONTINUOUS, name="x")
    r = sp.addVars(S, ALL, vtype=GRB.CONTINUOUS, name="r")
    if relax:
        delta = sp.addVars(S, ALL, ALL, vtype=GRB.CONTINUOUS, name="delta")
    else:
        delta = sp.addVars(S, ALL, ALL, vtype=GRB.BINARY, name="delta")

    sp.addConstrs(x[s, i] == sol_dict[t[i]] + ds[s][i] for s in S for i in ALL)
    sp.addConstrs(delta[s, j, i] + delta[s, i, j] == 1 for s in S for i in ALL for j in ALL if i > j)
    sp.addConstrs(r[s, i] - x[s, i] >= lb_u[i] for s in S for i in ALL)
    sp.addConstrs(x[s, i] - r[s, i] >= -ub_u[i] for s in S for i in ALL)

    sp.addConstrs(
        r[s, j] - r[s, i] + delta[s, j, i] * 10000 >= sep[i, j] * sol_dict[z[i, j]] for s in S for i in ALL for j in
        ALL if i != j)
    sp.update()
    sp.setObjective(gp.quicksum(r[s, i] - x[s, i] for s in S for i in ALL), GRB.MINIMIZE)

    sp.optimize()
    return sp
def solve_benders(mp, gap=10, relax=True):
    while True:
        mp.optimize()
        curr_sol = mp.getAttr('X')
        sol_dict = {var: curr_sol[i] for i, var in enumerate(mp.getVars())}
        sp = makesp(sol_dict, relax)

        rhs = [t[i] + ds[s][i] for s in S for i in ALL] + [1 for s in S for i in ALL for j in ALL if i > j] + \
              [lb_u[i] for s in S for i in ALL] + [-ub_u[i] for s in S for i in ALL] + \
              [sep[i, j] * z[i, j] for s in S for i in ALL for j in ALL if i != j]
        rhs = np.array(rhs)
        print(sp.status)
        if sp.status == GRB.OPTIMAL:
            sp_obj_val = sp.ObjVal
            q_value = mp.getVarByName("q").X
            if abs(sp_obj_val - q_value) > gap:
                print('Adding optimality cut')
                duals = [constr.Pi for constr in sp.getConstrs()]
                duals = np.array(duals)
                optcut = sum(duals * rhs)
                mp.addConstr(optcut <= q)
            else:
                print(mp.NumConstrs)
                break
        elif sp.status == GRB.INF_OR_UNBD:
            print('Adding infeasibility cut')
            farkas_duals = [constr.FarkasDual for constr in sp.getConstrs()]
            optcut = sum(farkas_duals * rhs)
            mp.addConstr(optcut <= 0)

# phase 1 : LP
solve_benders(mp)

for i in ALL:
    for j in R:
        y[i, j].setAttr(GRB.Attr.VType, GRB.BINARY)
for i in ALL:
    for j in ALL:
        z[i, j].setAttr(GRB.Attr.VType, GRB.BINARY)

# phase 2 : relax sp
solve_benders(mp)

# phase 3 :
mp.optimize()
curr_sol = mp.getAttr('X')
sol_dict = {var: curr_sol[i] for i, var in enumerate(mp.getVars())}
sp = makesp(sol_dict, relax=False)

def postbsp(t,z):
    sp = gp.Model("sp")
    # sp.setParam('PreCrush', 1)
    # sp.setParam('InfUnbdInfo', 1)
    sp.setParam('OutputFlag', 0)

    x = sp.addVars(S, ALL, vtype=GRB.CONTINUOUS, name="x")
    r = sp.addVars(S, ALL, vtype=GRB.CONTINUOUS, name="r")
    delta = sp.addVars(S, ALL, ALL, vtype=GRB.BINARY, name="delta")

    sp.addConstrs(x[s, i] == t[i] + ds[s][i] for s in S for i in ALL)
    sp.addConstrs(delta[s, j, i] + delta[s, i, j] == 1 for s in S for i in ALL for j in ALL if i > j)
    sp.addConstrs(r[s, i] - x[s, i] >= lb_u[i] for s in S for i in ALL)
    sp.addConstrs(x[s, i] - r[s, i] >= -ub_u[i] for s in S for i in ALL)

    sp.addConstrs(
        r[s, j] - r[s, i] + delta[s, j, i] * 10000 >= sep[i, j] * z[i, j] for s in S for i in ALL for j in
        ALL if i != j)
    sp.update()
    sp.setObjective(gp.quicksum(r[s, i] - x[s, i]for s in S for i in ALL), GRB.MINIMIZE)

    sp.optimize()
    X = np.array([[x[s, i].x for i in ALL] for s in S])
    RR = np.array([[r[s, i].x for i in ALL] for s in S])
    DELTA = np.array([[[delta[s, i, j].x for j in ALL] for i in ALL] for s in S])
    return sp,X,RR,DELTA

# 获取解池中的解的数量
solution_count = mp.SolCount
print(f"Number of solutions found: {solution_count}")
mp.setParam(GRB.Param.SolutionNumber,0)

# # 获取t,y,x,r,delta
T = np.array([t[i].Xn for i in ALL])
Y = np.array([[y[i, r].Xn for r in R] for i in ALL])
Z = np.array([[z[i, j].Xn for j in ALL] for i in ALL])
ALPHA = np.array([alpha[i].Xn for i in ALL])
BETA = np.array([beta[i].Xn for i in ALL])

sp2,X,RR,DELTA=postbsp(T,Z)

TrueObj=mp.objVal-1/len(S)*q.Xn+1/len(S)*sp2.objVal
print('s1:',mp.objVal-1/len(S)*q.Xn)
print('TrueObj:',TrueObj)

ac_list, df = saveres(D, A, ac_list, T, Y, X, RR, len(S), DELTA, ds, df, k)
drawres(D, A, ac_list, len(S), obte, obtl, ete, etl)