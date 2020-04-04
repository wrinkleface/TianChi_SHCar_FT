$$
\overbrace{\left(\begin{array}{c}{\vdots} \\ {P_{1}} \\ {\vdots}\end{array} \begin{array}{c}{\vdots} \\ {P_{2}} \\ {\vdots}\end{array} \right)}^{\text {Train_2 }}  
and 
\overbrace{\left(\begin{array}{c}{\vdots} \\ {T_{1}} \\ {\vdots}\end{array} \begin{array}{c}{\vdots} \\ {T_{2}} \\ {\vdots}\end{array} \right)}^{\text {Test_2 }}
$$

再用 次级模型 Model2 以真实训练集标签为标签训练,以train2为特征进行训练，预测test2,得到最终的测试集预测的标签列 $Y_{Pre}$。

$$
\overbrace{\left(\begin{array}{c}{\vdots} \\ {P_{1}} \\ {\vdots}\end{array} \begin{array}{c}{\vdots} \\ {P_{2}} \\ {\vdots}\end{array} \right)}^{\text {Train_2 }} \overbrace{\Longrightarrow}^{\text {Model2 Train} }\left(\begin{array}{c}{\vdots} \\ {Y}_{True} \\ {\vdots}\end{array}\right)
$$

$$
\overbrace{\left(\begin{array}{c}{\vdots} \\ {T_{1}} \\ {\vdots}\end{array} \begin{array}{c}{\vdots} \\ {T_{2}} \\ {\vdots}\end{array} \right)}^{\text {Test_2 }} \overbrace{\Longrightarrow}^{\text {Model1_2 Predict} }\left(\begin{array}{c}{\vdots} \\ {Y}_{Pre} \\ {\vdots}\end{array}\right)
$$

这就是我们两层堆叠的一种基本的原始思路想法。在不同模型预测的结果基础上再加一层模型，进行再训练，从而得到模型最终的预测。

Stacking本质上就是这么直接的思路，但是直接这样有时对于如果训练集和测试集分布不那么一致的情况下是有一点问题的，其问题在于用初始模型训练的标签再利用真实标签进行再训练，毫无疑问会导致一定的模型过拟合训练集，这样或许模型在测试集上的泛化能力或者说效果会有一定的下降，因此现在的问题变成了如何降低再训练的过拟合性，这里我们一般有两种方法。
* 1. 次级模型尽量选择简单的线性模型
* 2. 利用K折交叉验证

K-折交叉验证：
训练：

![](http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/2326541042/1584448819632_YvJOXMk02P.jpg)
