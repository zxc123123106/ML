# IoT-Based Fish Disease & Mortality Prediction System

This project implements a machine learning pipeline to predict fish mortality rates (`death_count`) in aquaculture ponds by analyzing high-frequency IoT sensor data and daily farm logs.

## 1. Project Overview
The primary objective is to build a predictive model that alerts pond managers to potential disease outbreaks or environmental stress by forecasting the number of fish deaths. We transition from raw, high-frequency sensor streams to a structured time-series supervised learning problem using **XGBoost**.

## 2. Dataset Description
The system utilizes two primary data sources:

*   **`sensor_data` (High Frequency):**
    *   Environmental metrics: `ph`, `temp_deep`, `temp_shallow`, `turbidity`, `hardness`.
    *   Gas/Air quality: `mq135` (Air quality), `mq137` (Ammonia).
    *   Water Color: `rgb_r`, `rgb_g`, `rgb_b`.
    *   Temporal Info: `created_at` (Timestamped).
*   **`pond_daily_logs` (Daily Labels):**
    *   Target Variable: `death_count`.
    *   Management Data: `feeding_amount`, `medication_given`, `log_date`.

---

## 3. Implementation Workflow

### Phase 1: Data Alignment & Preprocessing
*   **Temporal Alignment:** Convert `sensor_data` timestamps to local timezones and extract the date to join with `pond_daily_logs`.
*   **Data Integrity:** Filter records where `status == 'complete'`.
*   **Outlier Handling:** Implement physical threshold clipping (e.g., pH 0-14) and use linear interpolation for missing sensor packets.

### Phase 2: Time-Window Feature Engineering
Since fish mortality is often a cumulative result of environmental stress, we apply a **Sliding Window** approach:
*   **Daily Aggregation:** For every sensor metric, we calculate 24-hour statistics:
    *   `Mean`: General environment level.
    *   `Std`: Environmental stability/fluctuation.
    *   `Max/Min`: Exposure to extreme lethal conditions.
*   **Domain-Specific Features:**
    *   `Vertical Temp Diff`: `temp_shallow` - `temp_deep`.
    *   `Ammonia Surge`: Rate of change in `mq137` over the last 6 hours.
*   **Lag & Rolling Features:**
    *   **Lag-1:** Previous day's `death_count` and `feeding_amount`.
    *   **Rolling-3:** 3-day moving average of Ammonia and pH volatility to capture chronic stress.

### Phase 3: Machine Learning with XGBoost
We bypass traditional models and utilize **XGBoost (Extreme Gradient Boosting)** for its ability to handle non-linear relationships and missing values internally.

*   **Model Selection:** `XGBRegressor` with `objective='reg:squarederror'`.
*   **Validation Strategy:** **Time-Series Split**. Data is sorted chronologically. We train on the first 80% of the timeline and test on the final 20% to prevent "looking into the future."
*   **Hyperparameter Tuning:** Utilizing **Optuna** for Bayesian Optimization of `max_depth`, `learning_rate`, `subsample`, and `gamma`.

---

## 4. Key Execution Tasks

| Task ID | Component | Execution Detail |
| :--- | :--- | :--- |
| **T1** | **Data Sync** | Aggregate `sensor_data` by `Date` and `Pond_ID` using `groupby`. |
| **T2** | **Feature Ops** | Create 3-day rolling mean for `mq137` and `temp_deep`. |
| **T3** | **Baseline** | Train default XGBoost on daily aggregated stats. |
| **T4** | **Optimization** | Execute 100 trials of Optuna to minimize RMSE. |
| **T5** | **Evaluation** | Compare MAE (Mean Absolute Error) vs. a naive persistence model. |
| **T6** | **Explainability**| Generate **SHAP Summary Plots** to identify the top 5 mortality drivers. |

---

## 5. Model Evaluation Metrics
To ensure the system is reliable for aquaculture management, we evaluate:
1.  **MAE (Mean Absolute Error):** Average deviation in predicted fish deaths.
2.  **RMSE (Root Mean Square Error):** To penalize large errors during mass mortality events.
3.  **R-Squared:** The proportion of variance explained by environmental sensor inputs.

## 6. Project Explainability (SHAP)
Using the SHAP (SHapley Additive exPlanations) library, the system provides "Why" behind a prediction:
*   *Example:* If the model predicts 10 deaths, SHAP might show that a spike in `mq137_max` contributed +7 to that prediction, while `medication_given=True` reduced it by -2.

---

## 7. Future Enhancements
*   Integration of an **Anomaly Detection** layer to detect sensor drift.
*   Deployment via a **Flask API** to provide real-time alerts to fish farmers.

---
