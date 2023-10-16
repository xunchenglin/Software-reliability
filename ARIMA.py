from __future__ import print_function
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import pmdarima as pm
import numpy as np

data=[0, 9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 98, 104, 105, 116, 149, 156, 247, 249, 250, 337]

plt.plot(data, 'o-', label='初始值')
plt.legend()

model = pm.auto_arima(data, start_p=1, start_q=1,
                           max_p=8, max_q=8, m=1,
                           start_P=0, seasonal=False,
                           max_d=3, trace=True,
                           information_criterion='aic',
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=False)
forecast = model.predict(7)
print("后7次预测值：")
for i in range(len(forecast)):
    print(forecast[i])


PredicValue = data
for i in range(len(forecast)):
    PredicValue.append(forecast[i])
PredicValue=pd.Series(PredicValue)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(PredicValue, '*-', label='预测值')
plt.legend()
plt.show()