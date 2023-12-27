# DST4ATM
Optimization in Air Traffic Management
# 需要做的：
## 1. 传统benders 
主问题求解完毕后，再去做子问题
![image](https://github.com/Duzhuoming/DST4ATM/assets/65158088/903515f1-4492-49ad-9176-dbe254a8fae3)
具体可以参考:https://blog.csdn.net/weixin_43509834/article/details/124617203
## 2. branch & benders cut
开始求解主问题，当发现MIP solution时，使用callback功能，求解子问题，使用求解器的lazy cut方法,添加benders cut
具体可以参考：

[1] [Two-Stage Stochastic Mixed-Integer Programming with Chance Constraints for Extended Aircraft Arrival Management](https://doi.org/10.1287/trsc.2020.0991)
![image](https://github.com/Duzhuoming/DST4ATM/assets/65158088/8e9a326a-aeb8-4dcc-b29a-003f06abd657)


[2][Benders and its sub-problems, p.131](https://openresearch-repository.anu.edu.au/bitstream/1885/203507/1/thesis.pdf)
![image](https://github.com/Duzhuoming/DST4ATM/assets/65158088/3b071d51-9db6-43e0-ac7d-b76640c74617)

