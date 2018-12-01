# Machine To Learn
<!--
* 罗列1
	- 罗列二
[回到底部](#resume) 跳转到某个标题，标题都是小写，且用-连接
> 引用的文字
**文字** 字体加粗
`文字` 背景加灰
![](path/to/*.jpg) 引用图片
[百度](https://baidu.com) 跳转到百度
## 二级目录，两行作分隔符
### 三级目录，一行作分隔符
-->


## SQL

### Joins

* Inner join: return records that have matched values in both tables
* Left join: return all records from the left table, if matched in the right table 1 else 0
* Right join: oppsite to left join
* Outer join: return all records when there is a match in either left or right table

### Optimize

[todo]


### FAQ

[todo]


## Statistics and ML

### Project Workflow

Given a data science problem, what steps should we flollow? Or how to design a ML system?
or How to design a recommend engine? Heres are some major steps:

* Specify business objective
* Define the problem: The most important step. You should know how to model your problem and figure out which type it is, supervised or unsupervised, classification or regression.
* Create a baseline: No need to use ML or DL, even random select or rule based.
* Review ML literatures
* ML design
	- Do exploratory data analysis
	- Patition data
	- Preprocess
	- Engineer features
	- Choose proper alogorithm, ML or DL
	- Train, test and validation
	- Ensemble
* Deploy model
* Monitor model
* Iterate model

### Cross Validation

Cross-validation is a technique to enhance model stability and performace by partitioning the original data into training data to train the model, validation data to evaluate it. Usually, a k-fold cross validation divides the data into k folds, trains on each k-1 folds and the left 1 fold to evalatue. This results k models, which can be averaged to get an overall model performance.

### Feature Importance

* In linear models, feature importance can be calculated by the scale of the coefficients!!!
* In tree-based mothods, important features are likely to appear closer to the root of the tree. It can be computed by the average of the depth across all trees.

### Mean Squared Error vs. Mean Absolute Error

namely MSE vs. MAE.

* Similarity:






























