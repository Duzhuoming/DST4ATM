# DST4ATM
Optimization in Air Traffic Management
# 具体问题：
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
询问GPT后得知：

1) -1 (AUTO)： 这是一个自动模式，在此模式下，CPLEX将根据问题的结构和特点自动决定是否和如何应用Benders分解。在这种情况下，CPLEX优化器使用内部启发式来选择最适合当前问题的Benders策略。

2) 0 (OFF)： 这是默认值。在这个模式下，CPLEX不会自动应用Benders分解。然而，如果用户在模型中显式定义了Benders注解（通过分配变量和约束到不同的Benders阶段），CPLEX会根据这些注解应用Benders分解。

3) 1 (USER)： 在这个级别，CPLEX会使用用户指定的Benders注解来执行Benders分解。如果没有指定注解，则不会应用Benders分解。这允许用户有更细粒度的控制，但需要对Benders分解有深入的理解。

4) 2 (WORKERS)： CPLEX在这个级别会自动将变量和约束分配给不同的Benders阶段，但专注于并行环境中的工作效率。这意味着CPLEX会尝试通过适当地应用Benders分解来优化并行求解的效率。

5) 3 (FULL)： 这是最积极的Benders应用级别，CPLEX会尽可能广泛地应用Benders分解，哪怕这意味着采用更激进的策略。这可能导致更长的预处理时间，但在某些问题上可能显著提高求解效率。

根据，随机规划第二阶段所有场景的组合方式，benders cut 又有如下分类：

| cut                      | 说明               |
|:-------------------------|:-----------------|
| simple cut               | 所有场景归为一个子问题      |
| partially aggregated cut | 场景聚类， 一类场景为一个子问题 |
| multi cut                | 一个场景一个子问题        |



目前
- [x] FULL
- [ ] USER-simple
- [ ] USER-partial
- [ ] USER-multi

# IB&BC
文献[2][ref1]针对以上问题提出了一种新的求解方法，称为IB&BC，即integer benders & branch cut.通过启发式寻求上界，并close完整的搜索树，获得备选solution pool,可以获得全局最优解。
然而gurobi的callback不够完善，只有CPLEX支持自定义剪枝策略，所以目前着手实现CPLEX版本的IB&BC。

目前
- [ ] CPLEX版本



[ref1]:https://openresearch-repository.anu.edu.au/bitstream/1885/203507/1/thesis.pdf
[ref2]:https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/mp/jupyter/Benders_decomposition.ipynb
[^1]: relatively complete recourse.
[^2]:[Benders and its sub-problems, Sec. 4.2][ref1] In case the solution violates previous Benders cuts, they add a constraint and restart; otherwise, the algorithm finishes.
[^3]:如果不加以注释，CPLEX将自动把整数变量放第一阶段，连续变量全放第二阶段。