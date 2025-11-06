# BMW-Sales-Classification
End-to-end ML project predicting BMW car sales (High/Low). Includes EDA, preprocessing, SMOTE for imbalance, ensemble modeling (RandomForest + XGBoost + Voting/Stacking), hyperparameter tuning, and feature importance visualization.

#  BMW Sales Prediction (2010–2024) — Machine Learning Project

###  Objective  
Predict whether a BMW car listing will have **High** or **Low sales classification** based on features like:
- Engine Size  
- Mileage  
- Price  
- Model, Color, Region, Fuel Type, Transmission  

This is a real-world **classification problem** using tabular data.

---

## 📊 Project Workflow (End-to-End ML Pipeline)

###  1. Exploratory Data Analysis (EDA)
- Distribution plots (histograms)
- Boxplots for outlier detection
- Heatmaps (correlation check — detected leakage)
- Pairplots, violin plots, hexbin plots
- Category-wise comparison (Model, Region, Color, Fuel Type)

###  2. Data Preprocessing
- Removed **data leakage** (Sales Volume was directly driving the target)
- Handled missing values (not present)
- Encoding:
  - `Label Encoding` (target)
  - `OneHot Encoding` (categorical features)
- Stored cleaned dataset as `BMW_Cleaned_Preprocessed.csv`

###  3. Handling Imbalanced Dataset
- Target classes were skewed (`Low` ≈ 70%)
- Applied **SMOTE (Synthetic Minority Oversampling Technique)**
- Balanced classes: `Low = High`

###  4. Machine Learning Models Used
| Model | Method | Result |
|--------|--------|--------|
| Random Forest | Bagging Ensemble | ~74% |
| XGBoost | Boosting Ensemble | ~73% |
| Voting Classifier | RF + XGBoost | ~72% |
| **Tuned XGBoost (Best)** | Boosting + Hyperparameter tuning | **~75% Accuracy** ✅ |

###  5. Hyperparameter Tuning
- Used **RandomizedSearchCV**
- Best parameters:
  ```python
  {'subsample': 1.0, 'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.01, 'colsample_bytree': 0.6}
