---

# 專案報告：基於 IoT 的魚類疾病與死亡預測系統

## 專案概覽 (Project Overview)

本專案旨在透過分析高頻率的 **IoT 感測器數據** 與 **每日養殖日誌**，預測水產養殖場中的魚類死亡率。最終目標是建立一個基於 **XGBoost** 的預測模型，能夠在潛在的疾病爆發或環境壓力發生時，即時提醒養殖場管理人員。本系統的處理流程（Pipeline）負責將原始的感測器數據轉換為結構化的時序特徵，以進行監督式學習。

---

## 1. `main.py` - 專案編排器 (The Project Orchestrator)

**用途：** 此檔案是整個數據處理與機器學習流程的核心入口點與編排器。它定義了運算的執行順序、管理檔案路徑，並包含了提升開發效率的邏輯。

### 核心區段與邏輯：

*   **路徑定義 (`SENSOR_DATA_PATH`, `POND_DAILY_LOGS_PATH` 等)：**
    *   **目的：** 這些路徑定義了原始輸入資料（感測器讀數與每日日誌）的位置，以及處理後輸出檔案（如 `train_set.csv`, `test_set.csv`）的儲存位置。使用變數管理路徑可以提高程式碼的可維護性，並在檔案位置變更時輕鬆調整。
*   **`main()` 函式：**
    *   **目的：** 此函式封裝了整個工作流程。當 `main.py` 被直接執行時，系統會觸發此函式。
    *   **條件式重複處理邏輯 (`if os.path.exists(...)`)：**
        *   **目的：** 這是提升效率的關鍵功能。資料預處理與特徵工程通常涉及大量運算且耗時。此區塊會檢查最終的訓練集與測試集檔案是否已經存在。
        *   **使用的函式：**
            *   `os.path.exists()`: 檢查給定路徑的檔案或目錄是否存在。
            *   `pd.read_csv()`: 若檔案已存在，使用 Pandas 快速載入為 DataFrame。
            *   `pd.to_datetime()`: 將「日期」欄位轉回 datetime 物件，因為 CSV 儲存時會將其轉為字串，這對於時序運算至關重要。
        *   **為何使用：** 避免冗餘運算。如果您已經處理過一次數據，除非原始數據或處理邏輯有變動，否則無需重新執行。這能顯著加快開發與測試週期。
*   **呼叫 `build_features_and_split()`：**
    *   **目的：** 若處理後的檔案不存在，此行將觸發整個特徵工程與數據切分流程。
    *   **使用的函式：** 匯入自 `src.feature_builder.py` 的 `build_features_and_split()`。這是將原始數據轉換為最終訓練與測試資料集的主要函式。
*   **階段 3：機器學習與 XGBoost (佔位符)：**
    *   **目的：** 此部分清楚標示了專案的後續步驟（模型訓練、評估與可解釋性）。目前包含 TODO 註釋以引導未來開發。
    *   **為何使用：** 明確定義 Pipeline 的各個階段，並根據技術文件規劃的結構，為加入機器學習組件提供清晰的框架。
*   **`if __name__ == '__main__':` 區塊：**
    *   **目的：** 這是標準的 Python 慣用語，確保 `main()` 函式僅在 `main.py` 被直接執行時運行。若該檔案被作為模組匯入其他腳本，則不會自動觸發。
    *   **為何使用：** 使檔案兼具「可執行腳本」與「可匯入模組」的特性，促進程式碼重用且避免意外副作用。

---

## 2. `src/data_loader.py` - 數據基礎與初步清理

**用途：** 此檔案負責數據管線的基礎步驟：載入感測器與每日日誌原始資料、執行初步的時間對齊、轉換時間戳，並執行關鍵的數據完整性檢查與初步清理。這對應了預處理階段中的「數據對齊與預處理」。

### 核心函式：

*   **`align_data(sensor_data_path, pond_daily_logs_path, min_records_per_day=10)`：**
    *   **目的：** 這是處理的第一步，接收原始 CSV 路徑，並回傳對齊且初步過濾後的 DataFrame。
    *   **使用的函式與原因：**
        *   `pd.read_csv()`: 從 CSV 檔案載入數據，這是讀取表格數據的標準方式。
        *   `pd.to_datetime()`: 將 `created_at`（感測器）與 `log_date`（日誌）欄位轉為 datetime 物件。這對於時序分析至關重要，能支援時區轉換、日期提取與排序。
        *   **時區處理 (`tz_localize('UTC')` 與 `tz_convert('Asia/Taipei')`)：**
            *   **目的：** IoT 設備通常以 UTC 或無時區資訊記錄。此步驟將其轉換為本地時區（台北），確保所有時間戳一致。
            *   **為何使用：** 精確的時區處理可避免在聚合每日數據時出現「差一天」的錯誤，這對於將感測器讀數正確關聯到每日日誌事件至關重要。
        *   **日期提取：** 從 `created_at` 中提取日期部分並轉回 datetime，以便將高頻率感測器數據與每日日誌進行合併（Merge）。
        *   **狀態過濾 (`status == 'complete'`)：** 僅保留狀態為「完成」的記錄，確保數據品質，剔除不完整或錯誤的讀數。
        *   **數據稀疏性檢查 (`groupby` & `size`)：** 計算每個魚塘每天的記錄數。若某天記錄過少，該數據可能不具代表性。
        *   **內連接過濾 (`inner merge`)：** 僅保留滿足「每日最低記錄數」門檻的日期與魚塘，防止後續特徵工程產生偏差。

*   **`impute_missing_sensor_values(df)`：**
    *   **目的：** 使用「前向填補法」（Forward-fill）填補感測器欄位的缺失值。
    *   **使用的函式與原因：**
        *   `sort_values()`: 依據魚塘 ID 與時間排序。對於時序數據，前向填補（使用前一個值）僅在時間序正確時才有意義。
        *   `ffill()`: 在各個魚塘組別內，將最後一個有效觀測值向下傳遞到下一個缺失值。
        *   **為何使用：** IoT 數據通常是連續的。若感測器暫時斷線，環境條件在短時間內通常不會劇烈變化，前向填補提供了一個合理的估計值。
        *   `bfill()`: （選擇性）用於填補序列開頭的缺失值，確保數據盡可能完整，以便進行數值計算。

*   **`handle_outliers(df)`：**
    *   **目的：** 識別並修正物理上不合理（Physically Implausible）的讀數，特別是針對 pH 值與深層水溫。
    *   **使用的函式與原因：**
        *   **數值替換 (`df.loc`)：** 將超出物理常規範圍（如 pH 0-14 之外）的數值替換為 `None`（即 NaN）。
        *   **為何使用：** 感測器故障可能產生極端且不可能的讀數，將其轉為 NaN 後，可使用更穩健的方法重新估計，而非直接刪除整筆記錄。
        *   **線性插值 (`interpolate(method='linear')`)：** 使用線性插值填補因異常值處理產生的空缺。
        *   **為何使用：** 線性插值假設有效點之間是逐漸變化的，這非常符合環境參數（如溫度、pH）的實際物理特性。

---

## 3. `src/cleaner.py` - 感測資料清理

**用途：** 此檔案承接 `data_loader.py` 已完成時間對齊與品質過濾後的資料，專注處理缺失值與物理異常值，避免後續時間窗口特徵被錯誤讀數放大。

### 核心函式：

*   **`impute_missing_sensor_values(df)`：**
    *   依 `pond_id` 與 `created_at` 排序後，對 `ph`, `turbidity`, `mq135`, `mq137`, `temp_deep`, `temp_shallow` 進行前向填補與必要的後向填補。
    *   這個策略適合短暫斷線情境：環境數值通常連續變化，前一筆有效讀數比全域平均值更能代表短時間內的真實狀態。
*   **`handle_outliers(df)`：**
    *   將 pH 超出 0-14、深層水溫超出 10-40 的值視為不合理讀數，先轉為缺失值，再於每個魚塘內做線性插值。
    *   此設計保留時間序列連續性，同時避免明顯故障值污染每日平均、最大值、最小值與 rolling 特徵。

---

## 4. `src/feature_builder.py` - 特徵工程與時間窗口

**用途：** 此檔案是本專案最重要的特徵工程模組，負責把高頻 IoT 感測資料轉換成每日監督式學習資料表，並建立死亡預測所需的滯後與累積壓力特徵。

### 每日聚合 (`daily_aggregation`)

*   以 `pond_id` 與 `date` 分組，對感測欄位建立 `mean`, `std`, `max`, `min`。
*   聚合欄位包含水質、氣體、水溫、RGB、水溫垂直差、亮度，以及本次補強後的 `mq137_6hr_change`。
*   額外建立 `ph_range = ph_max - ph_min`，用來表示每日 pH 波動幅度。

### 領域特徵 (`create_domain_specific_features`)

*   **`vertical_temp_diff`：** `temp_shallow - temp_deep`，反映上下層水溫差，可能代表水體分層或循環不足。
*   **`rgb_brightness`：** `rgb_r + rgb_g + rgb_b`，作為水色與透明度的簡化代理特徵。
*   **`mq137_6hr_change`：** 使用 Pandas 的時間窗口 `rolling(window='6h')`，在每個魚塘內計算目前讀數與最近 6 小時窗口中最早讀數的差值。
    *   舊做法使用 `diff(periods=360)`，隱含「每分鐘固定一筆」的假設。
    *   新做法直接依 `created_at` 的實際時間戳計算，因此可處理 IoT 斷線、延遲上傳或採樣間隔不固定的狀況。
    *   `min_periods=2` 可避免單一讀數被誤解為 6 小時變化量；資料不足時保留 NaN，交由後續模型或處理策略面對。

### Lag 與 Rolling 特徵 (`create_lag_rolling_features`)

此函式本身就是時間窗口邏輯的一種實作。為避免 IoT 或日誌缺日造成「上一筆資料」不等於「昨天」的問題，目前改成真正基於日曆時間的 Pandas 計算。

*   **`death_count_lag1` / `feeding_amount_lag1`：**
    *   透過日期位移與合併取得「同一魚塘、前一個日曆日」的值。
    *   若昨天沒有資料，lag 會是 NaN，而不是拿更早之前的上一筆資料補上。這能避免把 3 天前或 5 天前的管理狀態誤當作昨天。
*   **`mq137_mean_roll3` / `ph_range_roll3`：**
    *   使用 `rolling(window='3D', min_periods=1)`，代表實際經過時間中的最近 3 天，而不是固定最近 3 筆資料。
    *   若中間缺日，窗口只會納入時間上仍落在最近 3 天內的觀測值，符合 IoT 資料不規則到達的情境。

### 輸出資料

*   `data/processed/merged_daily_features.csv`：合併每日感測特徵與日誌標籤後的完整資料。
*   `data/processed/train_set.csv`：依時間排序後的前 80% 資料。
*   `data/processed/test_set.csv`：依時間排序後的後 20% 資料。

---

## 5. `src/trainer.py` 與 `src/evaluator.py` - 模型訓練與效能評估

**用途：** 這兩個模組負責建立預測魚類死亡數 `death_count` 的迴歸模型，並產出可供報告使用的效能指標、模型檔案與圖表。模型選用 XGBoost，原因是它能處理非線性關係、特徵交互作用與缺失值，適合 IoT 感測資料常見的不完整與不規則特性。

### 任務 A：特徵與標籤準備

*   **標籤：** `death_count`。
*   **特徵：** 每日感測器統計量、`mq137_6hr_change`、Lag 與 Rolling 時間窗口特徵。
*   **排除欄位：** `date`, `pond_id`, `death_count` 不作為模型輸入，避免模型記住日期或魚塘 ID，而是學習環境與管理狀態對死亡數的影響。
*   **缺失標籤處理：** 因為每日誌不是每天都有記錄，訓練與測試時會只保留 `death_count` 非空的列；特徵中的 NaN 則交由 XGBoost 處理。

### 任務 B：時間序列交叉驗證

*   使用 `TimeSeriesSplit` 取代隨機 K-Fold。
*   每一折都只用較早時間的資料訓練，並在較晚時間的資料驗證，模擬真實部署時「用過去預測未來」的情境。
*   驗證分數以 RMSE 為主要最佳化目標，同時保存 MAE 與 R-Squared。

### 任務 C：超參數搜尋

*   若環境已安裝 Optuna，`trainer.py` 會使用 TPE sampler 進行貝氏優化。
*   若尚未安裝 Optuna，會自動改用 `ParameterSampler` fallback，仍然搭配 `TimeSeriesSplit` 搜尋 `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`。
*   每次折內訓練都支援 `early_stopping_rounds`，用驗證集監控 RMSE，降低過擬合風險。

### 任務 D：多維度指標評估

*   **MAE：** 平均預測死亡數誤差，最直觀地反映實務上平均差幾隻魚。
*   **RMSE：** 對大誤差給更高懲罰，適合觀察模型是否錯失大量死亡事件。
*   **R-Squared：** 衡量目前感測與管理特徵對死亡數變異的解釋能力。
*   目前執行結果會寫入 `reports/metrics.json`，並輸出 `reports/test_predictions.csv` 供殘差分析。

### 任務 E：SHAP 風格模型可解釋性

*   `evaluator.py` 優先使用 `shap.TreeExplainer` 取得 SHAP contribution；若未安裝 `shap` 或版本不相容，才改用 XGBoost Booster 的 `pred_contribs=True` fallback。
*   `reports/shap_feature_contributions.csv` 會保存平均絕對貢獻度與平均貢獻方向。
*   `reports/figures/shap_summary.png` 顯示死亡數預測最重要的特徵，`reports/figures/top_feature_shap_dependence.png` 顯示最重要特徵與死亡風險貢獻之間的關係。

### 模型產物

*   `models/final_best_model.json`：最佳參數訓練後的 XGBoost 模型。
*   `models/feature_columns.json`：訓練時使用的特徵欄位順序，確保未來載入模型時欄位一致。
*   `reports/cv_results.csv`：每組超參數的時間序列交叉驗證結果。
*   `reports/best_params.json`：最終模型使用的超參數。
*   `reports/figures/feature_importance.png`：XGBoost 特徵重要度。
*   `reports/figures/prediction_vs_actual.png`：預測值與實際死亡數對照圖。
*   `reports/figures/residuals.png`：殘差分析圖。
*   `reports/figures/learning_curve.png`：訓練與驗證 RMSE 曲線。

### 本次執行摘要

*   目前環境已安裝 Optuna 與 shap，本次使用 Optuna 完成 10 組超參數搜尋，並以 `shap.TreeExplainer` 產生解釋性輸出。
*   最佳時間序列交叉驗證 RMSE：2.5106。
*   訓練集：MAE 0.4455、RMSE 0.8548、R-Squared 0.9419，共 28 筆有標籤資料。
*   測試集：MAE 1.6002、RMSE 2.0616、R-Squared 0.0000，共 11 筆有標籤資料。
*   SHAP contribution 排名前段顯示 `mq137_6hr_change_max`、`temp_deep_min`、`ph_max` 等特徵對死亡數預測最有影響；目前估計 `mq137_6hr_change_max` 約在 160 以上開始出現正向風險貢獻。此閾值只代表目前資料與模型下的解釋性觀察，仍需更多標籤資料驗證。

---

## 6. 目前限制與注意事項

*   `main.py` 若偵測到既有 `train_set.csv` 與 `test_set.csv`，會直接載入快取檔案。當特徵工程邏輯更新後，應重新執行 `python3 -m src.feature_builder` 產生新的 processed CSV。
*   `pond_daily_logs` 可能不是每天都有紀錄，因此 `death_count`, `feeding_amount`, `medication_given` 允許在合併後出現 NaN。這是資料缺口，而不是時間窗口計算錯誤。
*   若環境未安裝 `optuna`，系統會使用 fallback 搜尋，仍可完成訓練與評估。
*   目前資料中有標籤的列數偏少，測試集 R-Squared 可能不穩定；MAE、RMSE 與殘差圖應一起解讀。
