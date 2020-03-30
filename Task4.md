# Task4 建模调参 



##  学习目标

* 了解常用的机器学习模型，并掌握机器学习模型的建模与调参流程

## 内容介绍

1. 线性回归模型：
    - 线性回归对于特征的要求；
    - 处理长尾分布；
    - 理解线性回归模型；
2. 模型性能验证：
    - 评价函数与目标函数；
    - 交叉验证方法；
    - 留一验证方法；
    - 针对时间序列问题的验证；
    - 绘制学习率曲线；
    - 绘制验证曲线；
3. 嵌入式特征选择：
    - Lasso回归；
    - Ridge回归；
    - 决策树；
4. 模型对比：
    - 常用线性模型；
    - 常用非线性模型；
5. 模型调参：
    - 贪心调参方法；
    - 网格调参方法；
    - 贝叶斯调参方法；

## 本文推荐了一些博客与教材供初学者们进行学习。

###  线性回归模型

https://zhuanlan.zhihu.com/p/49480391

###  决策树模型

https://zhuanlan.zhihu.com/p/65304798

###  GBDT模型

https://zhuanlan.zhihu.com/p/45145899

###  XGBoost模型

https://zhuanlan.zhihu.com/p/86816771

###  LightGBM模型

https://zhuanlan.zhihu.com/p/89360721

###  推荐教材：

   - 《机器学习》 https://book.douban.com/subject/26708119/
   - 《统计学习方法》 https://book.douban.com/subject/10590856/
   - 《Python大战机器学习》 https://book.douban.com/subject/26987890/
   - 《面向机器学习的特征工程》 https://book.douban.com/subject/26826639/
   - 《数据科学家访谈录》 https://book.douban.com/subject/30129410/


## 代码示例

###  读取数据


```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```

reduce_mem_usage 函数通过调整数据类型，帮助我们减少数据在内存中占用的空间


```python
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
```


```python
sample_feature = reduce_mem_usage(pd.read_csv('data_for_tree.csv'))
```

    Memory usage of dataframe is 60507328.00 MB
    Memory usage after optimization is: 15724107.00 MB
    Decreased by 74.0%
    


```python
continuous_feature_names = [x for x in sample_feature.columns if x not in ['price','brand','model','brand']]
```

###  线性回归 & 五折交叉验证 & 模拟真实业务情况


```python
sample_feature = sample_feature.dropna().replace('-', 0).reset_index(drop=True)
sample_feature['notRepairedDamage'] = sample_feature['notRepairedDamage'].astype(np.float32)
train = sample_feature[continuous_feature_names + ['price']]

train_X = train[continuous_feature_names]
train_y = train['price']
```

####  简单建模


```python
from sklearn.linear_model import LinearRegression
```


```python
model = LinearRegression(normalize=True)
```


```python
model = model.fit(train_X, train_y)
```

查看训练的线性回归模型的截距（intercept）与权重(coef)


```python
'intercept:'+ str(model.intercept_)

sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)
```




    [('v_6', 3342612.384537345),
     ('v_8', 684205.534533214),
     ('v_9', 178967.94192530424),
     ('v_7', 35223.07319016895),
     ('v_5', 21917.550249749802),
     ('v_3', 12782.03250792227),
     ('v_12', 11654.925634146672),
     ('v_13', 9884.194615297649),
     ('v_11', 5519.182176035517),
     ('v_10', 3765.6101415594258),
     ('gearbox', 900.3205339198406),
     ('fuelType', 353.5206495542567),
     ('bodyType', 186.51797317460046),
     ('city', 45.17354204168846),
     ('power', 31.163045441455335),
     ('brand_price_median', 0.535967111869784),
     ('brand_price_std', 0.4346788365040235),
     ('brand_amount', 0.15308295553300566),
     ('brand_price_max', 0.003891831020467389),
     ('seller', -1.2684613466262817e-06),
     ('offerType', -4.759058356285095e-06),
     ('brand_price_sum', -2.2430642281682917e-05),
     ('name', -0.00042591632723759166),
     ('used_time', -0.012574429533889028),
     ('brand_price_average', -0.414105722833381),
     ('brand_price_min', -2.3163823428971835),
     ('train', -5.392535065078232),
     ('power_bin', -59.24591853031839),
     ('v_14', -233.1604256172217),
     ('kilometer', -372.96600915402496),
     ('notRepairedDamage', -449.29703564695365),
     ('v_0', -1490.6790578168238),
     ('v_4', -14219.648899108111),
     ('v_2', -16528.55239086934),
     ('v_1', -42869.43976200439)]




```python
from matplotlib import pyplot as plt
```


```python
subsample_index = np.random.randint(low=0, high=len(train_y), size=50)
```

绘制特征v_9的值与标签的散点图，图片发现模型的预测结果（蓝色点）与真实标签（黑色点）的分布差异较大，且部分预测值出现了小于0的情况，说明我们的模型存在一些问题


```python
plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], model.predict(train_X.loc[subsample_index]), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price is obvious different from true price')
plt.show()
```

    The predicted price is obvious different from true price
    


![output_22_1](https://img-blog.csdnimg.cn/20200321231804889.png)



通过作图我们发现数据的标签（price）呈现长尾分布，不利于我们的建模预测。原因是很多模型都假设数据误差项符合正态分布，而长尾分布的数据违背了这一假设。参考博客：https://blog.csdn.net/Noob_daniel/article/details/76087829


```python
import seaborn as sns
print('It is clear to see the price shows a typical exponential distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y)
plt.subplot(1,2,2)
sns.distplot(train_y[train_y < np.quantile(train_y, 0.9)])
```

    It is clear to see the price shows a typical exponential distribution
    




    <matplotlib.axes._subplots.AxesSubplot at 0x1b33efb2f98>




![output_24_2](https://img-blog.csdnimg.cn/20200321231820197.png)


在这里我们对标签进行了 $log(x+1)$ 变换，使标签贴近于正态分布


```python
train_y_ln = np.log(train_y + 1)
```


```python
import seaborn as sns
print('The transformed price seems like normal distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y_ln)
plt.subplot(1,2,2)
sns.distplot(train_y_ln[train_y_ln < np.quantile(train_y_ln, 0.9)])
```

    The transformed price seems like normal distribution
    




    <matplotlib.axes._subplots.AxesSubplot at 0x1b33f077160>




![output_27_2](https://img-blog.csdnimg.cn/20200321231840673.png)



```python
model = model.fit(train_X, train_y_ln)

print('intercept:'+ str(model.intercept_))
sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)
```

    intercept:23.515920686637713
    




    [('v_9', 6.043993029165403),
     ('v_12', 2.0357439855551394),
     ('v_11', 1.3607608712255672),
     ('v_1', 1.3079816298861897),
     ('v_13', 1.0788833838535354),
     ('v_3', 0.9895814429387444),
     ('gearbox', 0.009170812023421397),
     ('fuelType', 0.006447089787635784),
     ('bodyType', 0.004815242907679581),
     ('power_bin', 0.003151801949447194),
     ('power', 0.0012550361843629999),
     ('train', 0.0001429273782925814),
     ('brand_price_min', 2.0721302299502698e-05),
     ('brand_price_average', 5.308179717783439e-06),
     ('brand_amount', 2.8308531339942507e-06),
     ('brand_price_max', 6.764442596115763e-07),
     ('offerType', 1.6765966392995324e-10),
     ('seller', 9.308109838457312e-12),
     ('brand_price_sum', -1.3473184925468486e-10),
     ('name', -7.11403461065247e-08),
     ('brand_price_median', -1.7608143661053008e-06),
     ('brand_price_std', -2.7899058266986454e-06),
     ('used_time', -5.6142735899344175e-06),
     ('city', -0.0024992974087053223),
     ('v_14', -0.012754139659375262),
     ('kilometer', -0.013999175312751872),
     ('v_0', -0.04553774829634237),
     ('notRepairedDamage', -0.273686961116076),
     ('v_7', -0.7455902679730504),
     ('v_4', -0.9281349233755761),
     ('v_2', -1.2781892166433606),
     ('v_5', -1.5458846136756323),
     ('v_10', -1.8059217242413748),
     ('v_8', -42.611729973490604),
     ('v_6', -241.30992120503035)]



再次进行可视化，发现预测结果与真实值较为接近，且未出现异常状况


```python
plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], np.exp(model.predict(train_X.loc[subsample_index])), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price seems normal after np.log transforming')
plt.show()
```

    The predicted price seems normal after np.log transforming
    


![output_30_1](https://img-blog.csdnimg.cn/20200321231902283.png)


#### 模拟真实业务情况

但在事实上，由于我们并不具有预知未来的能力，五折交叉验证在某些与时间相关的数据集上反而反映了不真实的情况。通过2018年的二手车价格预测2017年的二手车价格，这显然是不合理的，因此我们还可以采用时间顺序对数据集进行分隔。在本例中，我们选用靠前时间的4/5样本当作训练集，靠后时间的1/5当作验证集，最终结果与五折交叉验证差距不大


```python
import datetime
```


```python
sample_feature = sample_feature.reset_index(drop=True)
```


```python
split_point = len(sample_feature) // 5 * 4
```


```python
train = sample_feature.loc[:split_point].dropna()
val = sample_feature.loc[split_point:].dropna()

train_X = train[continuous_feature_names]
train_y_ln = np.log(train['price'] + 1)
val_X = val[continuous_feature_names]
val_y_ln = np.log(val['price'] + 1)
```


```python
model = model.fit(train_X, train_y_ln)
```


```python
mean_absolute_error(val_y_ln, model.predict(val_X))
```




    0.19443858353490887





除此之外，决策树通过信息熵或GINI指数选择分裂节点时，优先选择的分裂特征也更加重要，这同样是一种特征选择的方法。XGBoost与LightGBM模型中的model_importance指标正是基于此计算的


####  模型调参

在此我们介绍了三种常用的调参方法如下：

  - 贪心算法 https://www.jianshu.com/p/ab89df9759c8
  - 网格调参 https://blog.csdn.net/weixin_43172660/article/details/83032029
  - 贝叶斯调参 https://blog.csdn.net/linxid/article/details/81189154


```python
## LGB的参数集合：

objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']

num_leaves = [3,5,10,15,20,40, 55]
max_depth = [3,5,10,15,20,40, 55]
bagging_fraction = []
feature_fraction = []
drop_rate = []
```

####  贪心调参


```python
best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score
    
best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0], num_leaves=leaves)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score
    
best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x:x[1])[0],
                          max_depth=depth)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score
```


```python
sns.lineplot(x=['0_initial','1_turning_obj','2_turning_leaves','3_turning_depth'], y=[0.143 ,min(best_obj.values()), min(best_leaves.values()), min(best_depth.values())])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1fea93f6080>




![83-1](https://img-blog.csdnimg.cn/20200321232159934.png)


####  Grid Search 调参


```python
from sklearn.model_selection import GridSearchCV
```


```python
parameters = {'objective': objective , 'num_leaves': num_leaves, 'max_depth': max_depth}
model = LGBMRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(train_X, train_y)
```


```python
clf.best_params_
```




    {'max_depth': 15, 'num_leaves': 55, 'objective': 'regression'}




```python
model = LGBMRegressor(objective='regression',
                          num_leaves=55,
                          max_depth=15)
```


```python
np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
```




    0.13626164479243302




## 总结

在本章中，我们完成了建模与调参的工作，并对我们的模型进行了验证。此外，我们还采用了一些基本方法来提高预测的精度，提升如下图所示。


```python
plt.figure(figsize=(13,5))
sns.lineplot(x=['0_origin','1_log_transfer','2_L1_&_L2','3_change_model','4_parameter_turning'], y=[1.36 ,0.19, 0.19, 0.14, 0.13])
```





![98-1](https://img-blog.csdnimg.cn/20200321232216795.png)


**Task5 建模调参 END.**

