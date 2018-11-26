# 导入需要使用的库
import pandas as pd
import re
import warnings
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
# 忽略警告
warnings.filterwarnings("ignore")
# 导入并处理数据
bank_data = pd.read_csv("./data/germancredit.csv")

# 数据标准化
bank_data['checkingstatus1'] = bank_data['checkingstatus1'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['history'] = bank_data['history'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['purpose'] = bank_data['purpose'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['savings'] = bank_data['savings'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['employ'] = bank_data['employ'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['status'] = bank_data['status'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['others'] = bank_data['others'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['property'] = bank_data['property'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['otherplans'] = bank_data['otherplans'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['housing'] = bank_data['housing'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['job'] = bank_data['job'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['tele'] = bank_data['tele'].map(lambda x: int(re.sub("\D", "", x)))
bank_data['foreign'] = bank_data['foreign'].map(lambda x: int(re.sub("\D", "", x)))

#查看描述统计数据
# print(bank_data.describe())

#提取建模用数据
train_data = bank_data[:900]
# 提取需要进行预测的数据
predict_data = bank_data[900:]

# 去除无关变量
train_data = train_data.drop(['tele'], axis=1)
predict_data = predict_data.drop(['tele'], axis=1)

# 定义一个函数对因变量进行重新编码，编程成数值型，即0和1
def coding(col, code_dict):
    colCoded = pd.Series(col, copy=True)  
    for key, value in code_dict.items():
        colCoded.replace(key, value, inplace=True) 
    return colCoded 

# 是=1, 否=0:
train_data["Default"] = coding(train_data['Default'], {'否':0,'是':1})
# 将自变量与因变量分开
X,y = train_data.drop(['Default'],axis=1),train_data[['Default']]

# 随机抽取训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 10)
# 开始构建一个逻辑回归模型
model = LogisticRegression()
# model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False)
# 模型以X_train,y_train为输入数据进行训练
model.fit(X_train,y_train)
# 打印针对测试集而言的准确率
print("预测测试集的准确率:", accuracy_score(y_test,model.predict(X_test)))
# 使用训练得到模型对这些新申请贷款的人的违约风险进行预测
print("测试集中贷款申请人违约风险预测情况:", model.predict(predict_data.drop(['Default'],axis=1)))