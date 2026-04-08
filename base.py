import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

params = {
    'booster': 'gbtree',          # 使用基於樹的模型 (預設)
    'objective': 'binary:logistic',# 指定任務為「二分類」並輸出機率
    'eval_metric': 'auc',         # 評價指標使用 AUC (ROC 曲線下面積)
    'max_depth': 5,               # 樹的最大深度，深度越高越容易過擬合
    'lambda': 10,                 # L2 正則化權重，數值越大模型越保守
    'subsample': 0.75,            # 訓練每棵樹時隨機採樣 75% 的數據，防止過擬合
    'colsample_bytree': 0.75,     # 訓練每棵樹時隨機採樣 75% 的特徵
    'min_child_weight': 2,        # 葉子節點所需的最小樣本權重和
    'eta': 0.025,                 # 學習率 (Learning Rate)，與下方 learning_rate 重複，通常選其一
    'seed': 0,                    # 隨機種子
    'nthread': 8,                 # 使用 8 個 CPU 執行緒進行運算
    'gamma': 0.15,                # 節點分裂所需的最小損失下降值
    'learning_rate': 0.01         # 學習率，步長越小，訓練越慢但可能更精確
}

df = pd.read_csv("./src/diabetes.csv")
data = df.iloc[:, :8]   # 選取前 8 欄作為特徵 (X)
target = df.iloc[:, -1] # 選取最後一欄作為標籤 (y)

train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=7)

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x)
watchlist = [(dtrain, 'train')]

bst = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist)
ypred = bst.predict(dtest) # 預測結果為 0~1 之間的機率值

# 设置阈值、评价指标
y_pred = (ypred >= 0.5)*1
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print ('Accuracy: %.4f' % metrics.accuracy_score(test_y,y_pred))
print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))

ypred = bst.predict(dtest)
print("Scroes of every test set's smaple\n",ypred)
ypred_leaf = bst.predict(dtest, pred_leaf=True)
print("The number of nodes of every test set's tree\n",ypred_leaf)
ypred_contribs = bst.predict(dtest, pred_contribs=True)
print("Importance of features\n",ypred_contribs )

xgb.plot_importance(bst,height=0.8,title='key features', ylabel='features')
plt.rc('font', family='Arial Unicode MS', size=14)
plt.show()