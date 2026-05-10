# 測試與操作說明

本文件說明如何在本專案中重新產生特徵、訓練 XGBoost 死亡數預測模型、評估模型表現，並檢查輸出產物。

## 1. 開發環境與依賴管理

建議在專案根目錄建立並使用專屬虛擬環境：

```bash
python3 -m venv venv
```

啟動環境：

```bash
source venv/bin/activate
```

Windows 使用：

```powershell
.\venv\Scripts\activate
```

安裝專案依賴：

```bash
python -m pip install -r requirements.txt
```

更新依賴清單：

```bash
python -m pip freeze > requirements.txt
```

本專案的 `.gitignore` 已忽略 `venv/`, `.venv/`, `.env`, `.env.*`，避免虛擬環境與本機密鑰設定進入版本控制。

目前程式不需要 `.env` 才能執行，因為資料、模型與報告路徑都使用專案內相對路徑，`trainer.py` / `evaluator.py` 也可用 CLI 參數覆寫路徑。若未來加入資料庫連線、API key、雲端儲存路徑、不同環境的資料根目錄，再新增 `.env` 與 `.env.example` 會比較合適。

## 2. 環境檢查

先確認主要套件可用：

```bash
python -c "import pandas, sklearn, xgboost, optuna, shap, matplotlib; print('environment ok')"
```

若 `optuna` 不存在，`src/trainer.py` 會自動改用 `ParameterSampler` fallback。若 `shap` 不存在，`src/evaluator.py` 會自動改用 XGBoost 原生 `pred_contribs=True` fallback。

## 3. 重新產生特徵資料

當 `src/feature_builder.py` 或原始資料更新後，執行：

```bash
python3 -m src.feature_builder
```

預期產物：

```text
data/processed/merged_daily_features.csv
data/processed/train_set.csv
data/processed/test_set.csv
```

檢查重點：

```bash
python3 - <<'PY'
import pandas as pd

train = pd.read_csv("data/processed/train_set.csv")
test = pd.read_csv("data/processed/test_set.csv")
print("train shape:", train.shape)
print("test shape:", test.shape)
print("train labeled:", train["death_count"].notna().sum())
print("test labeled:", test["death_count"].notna().sum())
PY
```

## 4. 訓練模型

執行完整訓練、時間序列交叉驗證、Optuna 調參、評估與圖表輸出：

```bash
python3 -m src.trainer
```

可用參數：

```bash
python3 -m src.trainer --n-trials 50 --n-splits 4 --early-stopping-rounds 30
```

快速 smoke test 可降低 trial 數：

```bash
python3 -m src.trainer --n-trials 2 --n-splits 2 --early-stopping-rounds 5
```

預期模型產物：

```text
models/final_best_model.json
models/feature_columns.json
```

預期報告產物：

```text
reports/metrics.json
reports/best_params.json
reports/cv_results.csv
reports/test_predictions.csv
reports/feature_importance.csv
reports/shap_feature_contributions.csv
reports/shap_thresholds.csv
reports/figures/feature_importance.png
reports/figures/learning_curve.png
reports/figures/prediction_vs_actual.png
reports/figures/residuals.png
reports/figures/shap_summary.png
reports/figures/top_feature_shap_dependence.png
```

## 5. 單獨評估既有模型

若模型已存在，只想重新產生測試集指標與圖表：

```bash
python3 -m src.evaluator
```

也可以指定路徑：

```bash
python3 -m src.evaluator \
  --model-path models/final_best_model.json \
  --test-path data/processed/test_set.csv \
  --feature-columns-path models/feature_columns.json
```

## 6. 基本測試指令

語法與 import 檢查：

```bash
python3 -m compileall src main.py
```

主流程檢查：

```bash
python3 main.py
```

完整端到端檢查：

```bash
python3 -m src.feature_builder
python3 -m src.trainer --n-trials 2 --n-splits 2 --early-stopping-rounds 5
python3 -m src.evaluator
```

## 7. 評估解讀

主要指標位於 `reports/metrics.json`：

*   `MAE`：平均預測死亡數誤差，越低越好。
*   `RMSE`：對大誤差懲罰較高，適合觀察大量死亡事件是否被錯估。
*   `R-Squared`：特徵對死亡數變異的解釋能力。若有標籤資料很少，此值可能不穩定，需搭配 MAE、RMSE 與殘差圖判讀。

SHAP 解釋性輸出：

*   `reports/shap_feature_contributions.csv`：特徵平均絕對 SHAP 貢獻度排名。
*   `reports/figures/shap_summary.png`：前幾名特徵的視覺化排名。
*   `reports/figures/top_feature_shap_dependence.png`：最重要特徵與風險貢獻之間的關係。
*   `reports/shap_thresholds.csv`：目前資料下，最重要特徵開始產生正向風險貢獻的近似值。

## 8. 常見狀況

*   `main.py` 只負責資料準備與提示訓練入口；真正訓練請執行 `python3 -m src.trainer`。
*   若更新了特徵工程，請先重新跑 `python3 -m src.feature_builder`，再訓練模型。
*   若圖表無法輸出，先確認 `reports/figures/` 可寫入，並執行 `python3 -m compileall src main.py` 檢查語法。
*   若 Optuna trial 很慢，可先用 `--n-trials 2` 做 smoke test，再增加 trial 數。
