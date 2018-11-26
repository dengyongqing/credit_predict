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

#违约情况统计
# print("违约情况统计:", bank_data['Default'].value_counts(normalize=True))
# Default = bank_data['Default'].value_counts(normalize=True)
# Default.plot.bar(title='Default')
# # sns.countplot(x='Default', data=bank_data, palette = 'Set1')
# plt.show()

#工作情况统计
# print("工作情况统计:", bank_data['job'].value_counts(normalize=True))
# Default = bank_data['job'].value_counts(normalize=True)
# Default.plot.bar(title='job')
# # sns.countplot(x='Default', data=bank_data, palette = 'Set1')
# plt.show()

#婚姻情况统计
print("婚姻情况统计:", bank_data['status'].value_counts(normalize=True))
Default = bank_data['status'].value_counts(normalize=True)
Default.plot.bar(title='status')
# sns.countplot(x='Default', data=bank_data, palette = 'Set1')
plt.show()