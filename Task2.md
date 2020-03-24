
# Task2 EDA-数据探索性分析
## 2.1EDA目标
* EDA的价值主要在于熟悉数据集，了解数据集，对数据集进行验证来确定所获得数据集可以用于接下来的机器学习或者深度学习使用。

* 当了解了数据集之后我们下一步就是要去了解变量间的相互关系以及变量与预测值之间的存在关系。

* 引导数据科学从业者进行数据处理以及特征工程的步骤,使数据集的结构和特征集让接下来的预测问题更加可靠。

* 完成对于数据的探索性分析，并对于数据进行一些图表或者文字总结并打卡。
## 2.2  内容介绍
1.  载入各种数据科学以及可视化库:
    - 数据科学库 pandas、numpy、scipy；
    - 可视化库 matplotlib、seabon；
    - 其他；
2.  载入数据：
    - 载入训练集和测试集；
    - 简略观察数据(head()+shape)；
3.  数据总览:
    - 通过describe()来熟悉数据的相关统计量
    - 通过info()来熟悉数据类型
4.  判断数据缺失和异常
    - 查看每列的存在nan情况
    - 异常值检测
5.  了解预测值的分布
    - 总体分布概况（无界约翰逊分布等）
    - 查看skewness and kurtosis
    - 查看预测值的具体频数
6.  特征分为类别特征和数字特征，并对类别特征查看unique分布
7.  数字特征分析
    - 相关性分析
    - 查看几个特征得 偏度和峰值
    - 每个数字特征得分布可视化
    - 数字特征相互之间的关系可视化
    - 多变量互相回归关系可视化
8.  类型特征分析
    - unique分布
    - 类别特征箱形图可视化
    - 类别特征的小提琴图可视化
    - 类别特征的柱形图可视化类别
    - 特征的每个类别频数可视化(count_plot)
9. 用pandas_profiling生成数据报告
## 2.3 代码示例
```python
#coding:utf-8
#导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
```
```python
## 1) 载入训练集和测试集；
Train_data = pd.read_csv('train.csv', sep=' ')
Test_data = pd.read_csv('testA.csv', sep=' ')
```
```python
## 2) 简略观察数据(head()+shape)
Train_data.head().append(Train_data.tail())
```

```python
# 可视化看下缺省值
msno.matrix(Train_data.sample(250))
```
![](https://img-blog.csdnimg.cn/20200321224455424.png)

```python
msno.bar(Train_data.sample(1000))
```

![](https://img-blog.csdnimg.cn/20200321224537812.png)



```python
# 可视化看下缺省值
msno.matrix(Test_data.sample(250))
```


![](https://img-blog.csdnimg.cn/20200321224619231.png)


    


```python
f , ax = plt.subplots(figsize = (7, 7))

plt.title('Correlation of Numeric Features with Price',y=1,size=16)

sns.heatmap(correlation,square = True,  vmax=0.8)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x111b6ed5b70>




![61-9](https://img-blog.csdnimg.cn/20200321225237140.png)



```python
del price_numeric['price']
```


```python
## 3) 每个数字特征得分布可视化
f = pd.melt(Train_data, value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
```


![72-0](https://img-blog.csdnimg.cn/20200321225310318.png)



```python
## 4) 数字特征相互之间的关系可视化
sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()
```


![74-0](https://img-blog.csdnimg.cn/20200321225347270.png)


```python
## 5) 多变量互相回归关系可视化
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(24, 20))
# ['v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
v_12_scatter_plot = pd.concat([Y_train,Train_data['v_12']],axis = 1)
sns.regplot(x='v_12',y = 'price', data = v_12_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

v_8_scatter_plot = pd.concat([Y_train,Train_data['v_8']],axis = 1)
sns.regplot(x='v_8',y = 'price',data = v_8_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

v_0_scatter_plot = pd.concat([Y_train,Train_data['v_0']],axis = 1)
sns.regplot(x='v_0',y = 'price',data = v_0_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

power_scatter_plot = pd.concat([Y_train,Train_data['power']],axis = 1)
sns.regplot(x='power',y = 'price',data = power_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

v_5_scatter_plot = pd.concat([Y_train,Train_data['v_5']],axis = 1)
sns.regplot(x='v_5',y = 'price',data = v_5_scatter_plot,scatter= True, fit_reg=True, ax=ax5)

v_2_scatter_plot = pd.concat([Y_train,Train_data['v_2']],axis = 1)
sns.regplot(x='v_2',y = 'price',data = v_2_scatter_plot,scatter= True, fit_reg=True, ax=ax6)

v_6_scatter_plot = pd.concat([Y_train,Train_data['v_6']],axis = 1)
sns.regplot(x='v_6',y = 'price',data = v_6_scatter_plot,scatter= True, fit_reg=True, ax=ax7)

v_1_scatter_plot = pd.concat([Y_train,Train_data['v_1']],axis = 1)
sns.regplot(x='v_1',y = 'price',data = v_1_scatter_plot,scatter= True, fit_reg=True, ax=ax8)

v_14_scatter_plot = pd.concat([Y_train,Train_data['v_14']],axis = 1)
sns.regplot(x='v_14',y = 'price',data = v_14_scatter_plot,scatter= True, fit_reg=True, ax=ax9)

v_13_scatter_plot = pd.concat([Y_train,Train_data['v_13']],axis = 1)
sns.regplot(x='v_13',y = 'price',data = v_13_scatter_plot,scatter= True, fit_reg=True, ax=ax10)

```
    <matplotlib.axes._subplots.AxesSubplot at 0x111b47b2b38>


![78-1](https://img-blog.csdnimg.cn/20200321225421868.png)


```python
## 4) 类别特征的柱形图可视化
def bar_plot(x, y, **kwargs):
    sns.barplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(bar_plot, "value", "price")
```


![86](https://img-blog.csdnimg.cn/2020032123000585.png)



```python
##  5) 类别特征的每个类别频数可视化(count_plot)
def count_plot(x,  **kwargs):
    sns.countplot(x=x)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data,  value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(count_plot, "value")

```



### 用pandas_profiling生成数据报告
用pandas_profiling生成一个较为全面的可视化和数据报告(较为简单、方便) 最终打开html文件即可


```python
import pandas_profiling
```


```python
pfr = pandas_profiling.ProfileReport(Train_data)
pfr.to_file("./example.html")
```
  

## 2.4 经验总结

在大数据分析工具出现之前，参与决策指导的数据一般都是人工分析得出的。科学的决策需要科学的数据，人工分析数据并不能保证数据的绝对真实和客观。这意味着在大数据分析工具的使用中，数据必须确保真实与可靠。

　　国内有些数据分析工具在性能上已经能比肩国外同类技术。国云数据的新锐产品 大数据魔镜，作为国内领先的数据分析工具，能为用户提供完整的数据分析。随着数据市场和云BI等功能的开放，大数据魔镜有望成为新的数据分析平台。

　　大数据价值体现在服务人类，大数据和大数据分析工具都是为人服务的，这在大数据魔镜的功能中被体现地淋漓尽致——人性化、智能化服务于用户。数据分析工具的作用取决于人们的需要，而不是数据本身。

　　在大数据的帮助下，我们将会越来越清晰地看到这个世界的本来面目，也会越来越清晰地认识人类自身。而大数据分析工具，就是探索大数据与现实世界之间联系的放大镜和启明灯!
所给出的EDA步骤为广为普遍的步骤，在实际的不管是工程还是比赛过程中，这只是最开始的一步，也是最基本的一步。

接下来一般要结合模型的效果以及特征工程等来分析数据的实际建模情况，根据自己的一些理解，查阅文献，对实际问题做出判断和深入的理解。

最后不断进行EDA与数据处理和挖掘，来到达更好的数据结构和分布以及较为强势相关的特征



**Task2 EDA数据分析 END.**


