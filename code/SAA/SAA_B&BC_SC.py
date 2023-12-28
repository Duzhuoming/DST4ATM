import numpy as np
import gurobipy as gp
from gurobipy import GRB
from DST4ATM.optbase.aircraft_rso import Parameters, saveres, drawres
import numpy as np
import warnings
from docplex.mp.model import Model

warnings.filterwarnings('ignore')
modeltype = 'SAA'
timerange = 3000
S = 1
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
# 设置解池相关参数
# mp.setParam(GRB.Param.PoolSolutions, 10)  # 存储10个最优解
# mp.setParam(GRB.Param.PoolSearchMode, 1)  # 搜索更多解

t = mp.addVars(ALL, vtype=GRB.CONTINUOUS, name="t")
y = mp.addVars(ALL, R, vtype=GRB.BINARY, name="y")
z = mp.addVars(ALL, ALL, vtype=GRB.BINARY, name="z")
q=mp.addVar(vtype=GRB.CONTINUOUS, lb=0, name="q")
xmax = mp.addVars(ALL, vtype=GRB.CONTINUOUS, name="xmax")
xmax2 = mp.addVar(vtype=GRB.CONTINUOUS, name="xmax2")
xmax3 = mp.addVar(vtype=GRB.CONTINUOUS, name="xmax3")
xmax4 = mp.addVar(vtype=GRB.CONTINUOUS, name="xmax4")
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

def makesp(sol_dict={},relax=True):
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
    num_constrs1 = sp.NumConstrs
    sp.setObjective(gp.quicksum(r[s, i] - x[s, i] for s in S for i in ALL), GRB.MINIMIZE)

    sp.optimize()
    return sp


def mycallback(model, where):
    if where == GRB.Callback.MIPSOL:
        # 获取主问题的当前整数解
        curr_sol = model.cbGetSolution(model._vars)
        sol_dict = {var: curr_sol[i] for i, var in enumerate(model._vars)}
        # 构建并求解子问题
        sp=makesp(sol_dict)
        # bsp=makesp(sol_dict,relax=False)
        rhs = ([t[i] + ds[s][i] for s in S for i in ALL] + [1 for s in S for i in ALL for j in ALL if i > j] +
               [lb_u[i] for s in S for i in ALL] + [-ub_u[i] for s in S for i in ALL] +
               [sep[i, j] * z[i, j] for s in S for i in ALL for j in ALL if i != j])
        rhs=np.array(rhs)
        # 根据子问题的结果向主问题添加惰性约束（如适用）and bsp.status == GRB.OPTIMAL
        if sp.status == GRB.OPTIMAL :
            print('optimal cut')
            # extreme point,get dual var
            constraints = sp.getConstrs()
            print( model.NumConstrs)
            # num_constrs2 = len(constraints)
            duals = [constr.Pi for constr in constraints]
            duals=np.array(duals)
            # num_duals = len(duals)
            # 遍历约束和对偶变量
            optcut = sum(duals * rhs)
            # lazy cut
            model.cbLazy( optcut<=q)
            # in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
        elif sp.status == GRB.INFEASIBLE  or sp.status == GRB.INF_OR_UNBD :
            print('feasible cut,sp.status:',sp.status)

            farkas_duals = [constr.FarkasDual for constr in sp.getConstrs()]
            optcut = sum(farkas_duals * rhs)
            model.cbLazy( optcut<=0)
        # elif sp.status == GRB.OPTIMAL and bsp.status == GRB.INF_OR_UNBD:
        #     print('bsp cut')
        #     duals = [constr.Pi for constr in sp.getConstrs()]
        #     optcut = sum(duals * rhs)
        #     model.cbLazy( optcut<=0)



        else:
            print('error')
    # elif where == GRB.Callback.MIPSOL:

# 注册回调函数
mp._vars = list(t.values()) + list(y.values()) + list(z.values())
# mp.NumVars
# mp.NumConstrs
mp.update()
print(mp.NumConstrs)

mp.optimize(mycallback)

# bsp=makesp(relax=False)


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
    num_constrs1 = sp.NumConstrs
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
sp2,X,RR,DELTA=postbsp(T,Z)

TrueObj=mp.objVal-1/len(S)*q.Xn+1/len(S)*sp2.objVal
print('TrueObj:',TrueObj)
ALPHA = np.array([alpha[i].Xn for i in ALL])
BETA = np.array([beta[i].Xn for i in ALL])
# # zij
# # 获取obj
# OBJ = mp.objVal
# obj1_value = obj1.getValue()
# obj2_value = obj2.getValue()
# obj3_value = 1 / len(S) * obj3.getValue()
#
# gap = mp.getAttr('MIPGap')
# # mp.NumVars
# # mp.NumConstrs
#
#
# S = len(S)
# # ddf=pd.DataFrame([obj1_value,obj2_value,obj3_value,gap,t]).T
# # ddf.to_csv(fr'D:\nuaadzm\PycharmProjects\detour\rso_model3\saa\ans3.csv')
# dsd = get_random(randtype, S, D, p1d, p2d, seed=42)
# dsa = get_random(randtype, S, A, p1a, p2a, seed=42)
# ds = np.concatenate((dsd, dsa), axis=1)
# ac_list, df = saveres(D, A, ac_list, T, Y, X, RR, S, DELTA, ds, df, k)
# drawres(D, A, ac_list, S, obte, obtl, ete, etl)
#
# mtp(mp)
