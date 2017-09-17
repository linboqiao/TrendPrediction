# Virtual\_Stock_Forecasting
## **虚拟股票预测**

比赛网址：[https://challenger.ai/competition/trendsense](https://challenger.ai/competition/trendsense)

主要任务为通过挖掘虚拟股票大规模历史数据的内在规律，实现对虚拟股票未来趋势的预测。

### 数据说明:

**训练数据**，是一个以逗号分隔的文本文件(csv)，格式示例：

|id|feature_0|...|feature_n|weight|label|group|era|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0|0.254232|...|0.473321|9.0|1.0|1.0|1.0|
|1|0.763212|...|0.309311|3.0|0.0|7.0|1.0|

其中id列为数据唯一标识编码，feature列为原始数据经过变换之后得到的特征，weight列为样本重要性，label列为待预测二分类标签，group列为样本所属分组编号，era列为时间区间编号(取值1-20为时间顺序)。

**测试数据**，是一个以逗号分隔的文本文件(csv)，格式示例：

|id|feature_0|...|feature_n|group|
|:---:|:---:|:---:|:---:|:---:|
|600001|0.427248|...|0.754322|3.0|
|600002|0.253232|...|0.543121|5.0|


## 解决方案:
**方案一：**LR（Logistic Regression）  
**方案二：**XGBOOST  
**方案三：**DNN（Deep Neural Networks）
