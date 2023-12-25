import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
# data=pd.read_excel(r"D:\nuaadzm\PycharmProjects\detour\rso_model3\db\ref.xlsx")
# data = pd.read_csv(r'D:\nuaadzm\PycharmProjects\detour\ML\allset1112.csv')
data=pd.read_excel(r'D:\nuaadzm\PycharmProjects\detour\ML\allset1112.xlsx')

# data=pd.read_excel(r'D:\nuaadzm\PycharmProjects\detour\rso_model3\db\ref.xlsx')

# def get_set(data):
#     selected_columns = list(range(73)) + [74]
#     X = data.iloc[:, selected_columns]
#     y = data.iloc[:, 73]
#     trainX = list(range(16795)) + [i for i in range(16795 + 9596, len(X))]
#     X_train = X.iloc[trainX, :]
#     y_train = y.iloc[trainX]
#     testX = [i for i in range(16795, 16795 + 9596)]
#     X_test = X.iloc[testX, :]
#     y_test = y.iloc[testX]
#     return X_train, X_test, y_train, y_test, X, y

def get_set(data,fix=None):
    # 01 1 02 2 19 3 20 4
    data['date'] = pd.to_datetime(data['date'])

    X = data.iloc[:, 1:-2]
    y = data.iloc[:, -2]
    con1 = (data['date'] < '2019-12-03')# | (data['date'] > '2019-12-14'))#&(data.entryfix ==fix)
    con2 = (data['date'] >= '2019-12-03') & (data['date'] <= '2019-12-12') #& (data.entryfix ==fix)
    # con1 =  ((data['date'] < '2019-12-20') )&(data.entryfix == fix)
    # con2 = (data['date'] >= '2019-12-20') & (data['date'] <= '2019-12-31') & (data.entryfix == fix)
    X_train = data[con1].iloc[:, 1:-2]
    y_train = data[con1].iloc[:, -2]
    X_test = data[con2].iloc[:, 1:-2]
    y_test = data[con2].iloc[:, -2]
    return X_train, X_test, y_train, y_test, X, y


X_train, X_test, y_train, y_test, X, y = get_set(data)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# # 假设 `model` 是你的训练好的模型
#
# # find parameter
# param_grid = {
#     'n_estimators': [ 200, 300],  # 更多的树可以提高准确性，但也会增加计算成本
#     'max_features': [1.0, 'sqrt', 'log2'],  # 特征选择的策略
#     'max_depth': [6,10,15, 20, ],  # 树的最大深度，None表示树会在所有叶子都纯净或者达到最小样本数时才停止生长
#     'min_samples_split': [2, 5, 10],  # 分割内部节点所需的最小样本数
#     # 'min_samples_leaf': [1, 2, 4],  # 叶节点上所需的最小样本数
#     # 'bootstrap': [True, False]  # 是否采用bootstrap样本
#     'criterion':['mse','friedman_mse','mae']
#
# }
# from sklearn.model_selection import KFold
# cv = KFold(n_splits=5, shuffle=True, random_state=42)
#
# gsearch1 = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
#                         param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv,n_jobs=-1)
#
# gsearch1.fit(X_train.iloc[:, :-1], y_train)
# # 使用cross validation
# print(gsearch1.best_params_, gsearch1.best_score_)
# regressor=gsearch1.best_estimator_
regressor = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1, max_depth=20, max_features=0.75,
                                  criterion='friedman_mse')


def m():
    sns.kdeplot(y_train)
    sns.kdeplot(y_test)
    plt.show()
    print(regressor.score(X_train, y_train))
    # 训练集maemape

    y_pred = regressor.predict(X_train)
    print('train')
    print('MSE:', metrics.mean_squared_error(y_train, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    # mae
    print('MAE:', metrics.mean_absolute_error(y_train, y_pred))
    # mape
    print('MAPE:', np.mean(np.abs((y_train - y_pred) / y_train)) * 100)
    # 打印测试集的准确率
    # maemape
    y_pred = regressor.predict(X_test)
    print('tq')
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # mae
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    # mape
    print('MAPE:', np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    # 打印测试集的准确率
    print(regressor.score(X_test, y_test))


regressor.fit(X_train, y_train)
# 打印训练集的准确率
m()
# regressor.fit(X, y)
# m()
# dump(regressor, r'D:\nuaadzm\PycharmProjects\detour\ML\model_pretrain.joblib')
dump(regressor, r'D:\nuaadzm\PycharmProjects\detour\ML\model_pretrain_py312.joblib')

# from joblib import load
#
# regressor = load(r'D:\nuaadzm\PycharmProjects\detour\ML\model_pretrain.joblib')
regressor = load(r'D:\nuaadzm\PycharmProjects\detour\ML\model_pretrain_py312.joblib')
# # split data
# selected_columns = list(range(73)) + [74]
# X = data.iloc[:,selected_columns ]
# y = data.iloc[:, 73]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# regressor = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,max_depth=10,criterion='friedman_mse')
# regressor.fit(X_train.iloc[:, :-1], y_train)

feature_imp = pd.Series(regressor.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()