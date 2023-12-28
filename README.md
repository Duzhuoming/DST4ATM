# DST4ATM
Optimization in Air Traffic Management
# 具体问题：
![image](https://github.com/Duzhuoming/DST4ATM/assets/65158088/f74b08e7-ec86-4a37-a1bf-dc4cd25c4a5a)

参见[pdf](https://github.com/Duzhuoming/DST4ATM/blob/c6c62bffd5eb14bc0592e5127c67f2698d938df4/Integrated%20runway%20scheduling%20under%20operational%20time%20uncertainty.pdf)

# 直接求解
目前
- [x] gurobi版本
- [x] CPLEX版本
# 基于LP relaxation：
由于benders 要求二阶段问题是linear program(LP), 而我们目前的模型二阶段有排序变量，故为使用benders，我们需要将二阶段问题进行线性松弛(LP relaxation)

正常情况下[^1], 一阶段完全求解结束后，可以固定一阶段决策，在二阶段重新引入整数约束，再次求解，使得二阶段相关变量恢复整数的性质。
然而，我们的问题比较特殊，并不是所有一阶段决策都可以正常求解出二阶段决策。 所以，存在一些情况使用LP relaxation后，固定一阶段决策，二阶段无解。
这种情况下,根据文献[^2]，应当添加一些约束重新求解，但是文献没有表述的很清楚具体是什么约束,所以比较尴尬。
或者想办法把这个一阶段解屏蔽掉，重新求解，但是屏蔽一阶段解好像也不是一件容易的事情。

或者直接把infeasible的场景去掉？

但是即使二阶段问题解决了，这个方法始终是一个非精确的方法，因为二阶段benders cut是基于LP relaxation的，使得原问题的潜在的最优整数解，在一阶段被提早prune掉了。

## 1. 传统benders dual
主问题求解完毕后，再去做子问题
![image](https://github.com/Duzhuoming/DST4ATM/assets/65158088/903515f1-4492-49ad-9176-dbe254a8fae3)
具体可以参考:https://blog.csdn.net/weixin_43509834/article/details/124617203

目前
- [ ] gurobi版本
- [ ] CPLEX版本
## 2. branch & benders cut
开始求解主问题，当发现MIP solution时，使用callback功能，求解子问题，使用求解器的lazy cut方法,添加benders cut
具体可以参考：

[1] [Two-Stage Stochastic Mixed-Integer Programming with Chance Constraints for Extended Aircraft Arrival Management](https://doi.org/10.1287/trsc.2020.0991)
![image](https://github.com/Duzhuoming/DST4ATM/assets/65158088/8e9a326a-aeb8-4dcc-b29a-003f06abd657)

[2] [Benders and its sub-problems, p.131][ref1]
![image](https://github.com/Duzhuoming/DST4ATM/assets/65158088/3b071d51-9db6-43e0-ac7d-b76640c74617)

目前
- [x] gurobi版本
- [ ] CPLEX版本
## 3. CPLEX annotated Benders
CPLEX支持直接在变量和约束后注释([benders_annotation][ref2])其所属是主问题还是子问题，从而自动进行分解[^3]。此外，利用benders_annotation，CPLEX还可以将一个子问题再次拆解为多个子问题进行求解.

CPLEX supports 4 values for this [parameter][ref3], from -1 to 3:

OFF (default value) will ignore Benders.
AUTO, USER, WORKERS, FULL will enable Benders.

根据，随机规划第二阶段所有场景的组合方式，benders cut 又有如下分类：

| cut                      | 说明               |
|:-------------------------|:-----------------|
| simple cut               | 所有场景归为一个子问题      |
| partially aggregated cut | 场景聚类， 一类场景为一个子问题 |
| multi cut                | 一个场景一个子问题        |



目前
- [x] AUTO, FULL
- [x] WORKERS
- [x] USER-simple
- [ ] USER-partial
- [ ] USER-multi

# IB&BC
文献[2][ref1]针对以上问题提出了一种新的求解方法，称为IB&BC，即integer benders & branch cut.通过启发式寻求上界，并close完整的搜索树，获得备选solution pool,可以获得全局最优解。
然而gurobi的callback不够完善，只有CPLEX支持自定义剪枝策略，所以目前着手实现CPLEX版本的IB&BC。

目前
- [ ] CPLEX版本



[ref1]:https://openresearch-repository.anu.edu.au/bitstream/1885/203507/1/thesis.pdf
[ref2]:https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/mp/jupyter/Benders_decomposition.ipynb
[ref3]:https://www.ibm.com/docs/zh/icos/22.1.1?topic=parameters-benders-strategy
[^1]: relatively complete recourse.
[^2]:[Benders and its sub-problems, Sec. 4.2][ref1] In case the solution violates previous Benders cuts, they add a constraint and restart; otherwise, the algorithm finishes.
[^3]:如果不加以注释，CPLEX将自动把整数变量放第一阶段，连续变量全放第二阶段。
