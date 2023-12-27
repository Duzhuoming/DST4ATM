import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
# from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from joblib import dump, load
import gurobipy as gp
from gurobipy import GRB
import os
from pathlib import Path
current_directory = Path().cwd()

parent_directory = current_directory.parent.parent.parent

print(parent_directory)
feaset = pd.read_excel(fr'{parent_directory}\LocalData\allset1112.xlsx')
regressor = load(fr'{parent_directory}\LocalData\model_pretrain_py312.joblib')



def get_set(data, fix=None):
    data['date'] = pd.to_datetime(data['date'])

    X = data.iloc[:, 1:-2]
    y = data.iloc[:, -2]
    con1 = (data['date'] < '2019-12-03')  # | (data['date'] > '2019-12-14'))#&(data.entryfix ==fix)
    con2 = (data['date'] >= '2019-12-03') & (data['date'] <= '2019-12-12')  # & (data.entryfix ==fix)
    # con1 =  ((data['date'] < '2019-12-20') )&(data.entryfix == fix)
    # con2 = (data['date'] >= '2019-12-20') & (data['date'] <= '2019-12-31') & (data.entryfix == fix)
    X_train = data[con1].iloc[:, 1:-2]
    y_train = data[con1].iloc[:, -2]
    X_test = data[con2].iloc[:, 1:-2]
    y_test = data[con2].iloc[:, -2]
    return X_train, X_test, y_train, y_test, X, y


def print_test():
    # 1是ATAGA，2是IGONO， 3是P270，4是IDUMA，5是GYA，6是P71
    name = ['ATAGA', 'IGONO', 'P270', 'IDUMA', 'GYA', 'P71']
    for ind in range(6):
        ef = ind + 1

        condition = X_test['entryfix'] == ef

        condition2 = X['entryfix'] == ef
        realdif = y_pred[condition] - y_test[condition]

        # sns.displot(realdif)
        # plt.title(name[ind])
        # plt.tight_layout()
        # plt.show()
        print(name[ind], sum(condition2))
        # evaluation
        print('MSE:', metrics.mean_squared_error(y_test[condition], y_pred[condition]))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test[condition], y_pred[condition])))

        # mae
        print('MAE:', metrics.mean_absolute_error(y_test[condition], y_pred[condition]))
        # mape
        print('MAPE:', np.mean(np.abs((y_test[condition] - y_pred[condition]) / y_test[condition])) * 100)
        print('ME:', np.mean((y_test[condition] - y_pred[condition])))
        print(f'5%:', realdif.quantile(0.05), '95%', realdif.quantile(0.95))

        print(f'\n')

    # print(feature_imp)

    # Access the first tree in the forest
    tree = regressor.estimators_[0]

    # Plot the tree
    # plt.figure(figsize=(20,10))
    # plot_tree(tree, filled=True, feature_names=list(X.columns)[:-1], rounded=True, fontsize=10)
    # plt.show()
    # tree_structure = tree.tree_
    # print(compute_depth(tree_structure))

    # feature_imp = pd.Series(regressor.feature_importances_, index=X.columns).sort_values(ascending=False)
    # sns.barplot(x=feature_imp, y=feature_imp.index)
    # plt.xlabel('Feature Importance Score')
    # plt.ylabel('Features')
    # plt.title("Visualizing Important Features")
    # plt.legend()
    # plt.show()
    return realdif


def compute_depth(tree):
    left = tree.children_left
    right = tree.children_right

    def depth(node_id):
        if left[node_id] == right[node_id]:  # leaf node
            return 1
        left_depth = depth(left[node_id]) if left[node_id] != -1 else 0
        right_depth = depth(right[node_id]) if right[node_id] != -1 else 0
        return max(left_depth, right_depth) + 1

    return depth(0) - 1  # Subtract 1 to get depth starting from 0


def obtain_sample_insameleaf(pre_id):
    # X:测试集特征，y:训练集标签
    # regressor:训练好的模型
    # 获取测试集中每个样本所在的叶子节点的索引

    save = []
    saverow = []
    savelabel = []
    idl = leaf_indices[pre_id]  # list of leaf index in each tree
    # train leaf index
    trainleaf = leaf_indices[list(y_train.index)]
    for tree_idx, leaf_idx in enumerate(idl):
        # samples_in_leaf = X[leaf_indices[ :,tree_idx] == leaf_idx]
        labels_in_leaf = y_train[trainleaf[:, tree_idx] == leaf_idx]
        # rows = np.where(leaf_indices[:, tree_idx] == leaf_idx)[0]
        savelabel.append(labels_in_leaf)
        # save.append(samples_in_leaf)#save_feature
        # saverow.append(rows.tolist())
    # for i in range(1):
    #     unique_leaves = np.unique(leaf_indices[:, i])
    #     number_of_unique_leaves = len(unique_leaves)
    #     print(number_of_unique_leaves)

    return savelabel


def scenariodis(labelinsameleaf, pre_id, plot=False):
    #     统计每个标签出现的频数，统计标签分布
    index_counts = {}
    value_counts = {}
    values = []

    # 遍历每个Series对象
    for series in labelinsameleaf:
        # 遍历每个Series的索引和值
        for index, value in series.items():
            # 更新索引出现的次数
            if index in index_counts:
                index_counts[index] += 1
            else:
                index_counts[index] = 1

            # 收集值
            values.append(value)
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1

    # predict
    index_line_number_dict = {index: line_num for line_num, index in enumerate(y_test.index)}

    y_poipred = y_pred[index_line_number_dict[pre_id]]
    ytrue = y.iloc[pre_id]
    mean = np.mean(values)
    series_data = pd.Series(values)

    # 计算5%和95%的百分位数值
    quantile_5_percent = series_data.quantile(0.025)
    quantile_95_percent = series_data.quantile(0.975)

    # 选择5%到95%之间的数值
    selected_data = series_data[(series_data >= quantile_5_percent) & (series_data <= quantile_95_percent)]

    # 计算平均值
    average_pd = selected_data.mean()

    print('true:', ytrue, 'pred:', y_poipred, 'mean:', mean, '95mean:', average_pd)
    if plot:
        sns.kdeplot(values, label='Sample')
        sns.kdeplot(y_pred, label='Predictions')
        sns.kdeplot(y_test, label='true')
        plt.legend()
        plt.tight_layout()
        plt.show()
        sns.histplot(values, label='Sample')
        sns.histplot(selected_data, label='95sample', color='r', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return index_counts, values, value_counts


# realdif = print_test()
# 检测indexcount里是否存在测试集标签，不应该存在的.这只适用于同一个数据集里的训练测试。不同数据集索引重置了，所以方法无效


def ambiguous_set(values):
    lb = min(values)
    ub = max(values)
    mean = np.mean(values)
    sigma = np.std(values)

    return lb, ub, mean, sigma,


def sgm(value_counts,values,S):
    # S=3
    m = gp.Model("sync")
    m.setParam('OutputFlag', 0)

    # Sn= range(S)
    So = [k for k, v in value_counts.items()]  # range(len(w))
    gamma = m.addVars(So, vtype=GRB.BINARY, name="gamma")
    lamda = m.addVars(So, So, vtype=GRB.BINARY, name="lamda")
    d = m.addVars(So, So, vtype=GRB.CONTINUOUS, name='d')
    p = m.addVars(So, vtype=GRB.CONTINUOUS, name='p')

    m.addConstr((gp.quicksum(gamma[n] for n in So) == S), "sumgamma")
    m.addConstrs((gp.quicksum(lamda[o, n] for n in So) == 1 for o in So), "sumlamda")
    m.addConstrs((lamda[o, n] <= gamma[n] for o in So for n in So), "lamdasmallgamma")
    m.addConstrs((d[o, n] == abs(o - n) for o in So for n in So), "d")
    m.addConstrs((p[n] == gp.quicksum(lamda[o, n] * value_counts[o] for o in So) for n in So), "p")
    m.setObjective(gp.quicksum(d[o, n] * lamda[o, n] * value_counts[o] for o in So for n in So), GRB.MINIMIZE)
    m.optimize()

    gamma_values = m.getAttr('X', gamma)
    lamda_values = m.getAttr('X', lamda)
    p_values = m.getAttr('X', p)
    selectvalue = [(o, p_values[o], p_values[o] / len(values)) for o in So if gamma_values[o] > 0.5]
    # sort
    selectvalue.sort(key=lambda x: x[2], reverse=True)
    if len(values) != sum([i[1] for i in selectvalue]):
        print('error weight')
        return None
    else:
        return selectvalue

def getfset(sample_id, S=3):
    savelabel = obtain_sample_insameleaf(sample_id)
    index_counts, values, value_counts = scenariodis(savelabel, sample_id, 0)
    selectvalue = sgm(value_counts,values,S)

    if sum(index_counts.values()) != len(values):
        print('Error in weight of scenario')
        return None
    found_keys = [value in index_counts for value in list(y_test.index)]
    if sum(found_keys) > 0:
        print('Error:test keys in training set')
        # print(sum(found_keys))
        return None

    return ambiguous_set(values), selectvalue

X_train, X_test, y_train, y_test, X, y = get_set(feaset)
#

y_pred = regressor.predict(X_test)
leaf_indices = regressor.apply(X)
#
# fltid = 21720
# (lb, ub, mean, sigma), selectvalue = getfset(fltid)


# aaa=sum([i[0]*i[2] for i in selectvalue])
