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

bst = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)

ypred = bst.predict(dtest) # 預測結果為 0~1 之間的機率值
y_pred = (ypred >= 0.5) * 1 # 將機率轉化為類別：大於等於 0.5 為 1 (患病)，否則為 0

# 1. 每個樣本的得分
ypred = bst.predict(dtest) 
# 輸出的是每個測試樣本被預測為「1」的機率

# 2. 每個樣本在每棵樹所屬的節點
ypred_leaf = bst.predict(dtest, pred_leaf=True)
# 輸出一個矩陣 (樣本數, 樹的數量)，顯示樣本最後掉落在每棵樹的哪個葉子節點索引。這可用於特徵轉化。

# 3. 特徵貢獻度 (SHAP 值的概念)
ypred_contribs = bst.predict(dtest, pred_contribs=True)
# 輸出每個特徵對該預測值的貢獻度（偏離平均值的程度）。最後一列是 Bias (基準值)。

xgb.plot_importance(bst, height=0.8, title='Key features', ylabel='features')
plt.rc('font', family='Arial Unicode MS', size=14) # 設定中文顯示字型
plt.show()