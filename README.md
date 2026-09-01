# Housing Price Prediction – Kaggle Top 179

**Status:** Ongoing Kaggle competition | Continuously updated with improvements

## Overview
Regression model for housing price prediction using systematic data cleaning, domain-specific feature engineering, and automated model comparison across tree-based architectures.

## Performance
- **Best Model:** XGBoost (`XGBRegressor`)
- **Validation MAE:** $13,960.60
- **Test Submission Score:** $13,777.71
- **Kaggle Ranking:** Top 179 (out of 5,000+ participants)

## Key Findings

### Model Comparison Results
| Model | Validation MAE | Improvement vs. RF |
|-------|----------------|-------------------|
| **XGBoost** | $13,960.60 | Baseline (winner) |
| Random Forest | $16,374.02 | -17% |
| 50/50 Blend | $14,842.64 | -6% |

**Insight:** XGBoost's sequential gradient boosting and depth control (`max_depth=4`) significantly outperform bagging-based ensembles (`RandomForestRegressor`). An equal-weight 50/50 blend underperformed the single XGBoost model because the weaker Random Forest predictions dragged down overall accuracy.

### Model Robustness
- Validation MAE ($13,960.60) vs. Test MAE ($13,777.71) = **1.3% difference**
- **Conclusion:** Model generalizes exceptionally well to completely unseen test data with minimal overfitting.

## Methodology

1. **Outlier Removal:** Filtered two extreme low-sale-price outlier properties (`GrLivArea > 4000` sq ft sold under $300,000) prior to split and training.
2. **Data Cleaning & Imputation:**
   - **Ordinal Quality Mapping:** Encoded ordinal categorical features (`Ex`, `Gd`, `TA`, `Fa`, `Po`) to numeric integers (`1-5`).
   - **Neighborhood-Based Median Imputation:** Grouped `LotFrontage` missing values by `Neighborhood` medians.
   - **Categorical Missing Values:** Imputed categorical/text columns (`Alley`, `MasVnrType`, `GarageType`, etc.) with `'None'` and `Electrical` with its statistical mode.
3. **Feature Engineering:** Constructed 5 aggregate domain features:
   - `TotalSF`: Sum of `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`
   - `TotalBath`: Combined total of full and half baths (`FullBath` + `0.5*HalfBath` + `BsmtFullBath` + `0.5*BsmtHalfBath`)
   - `HouseAge`: Elapsed years between `YrSold` and `YearBuilt`
   - `RemodelAge`: Elapsed years between `YrSold` and `YearRemodAdd`
   - `TotalPorchSF`: Combined total porch and deck square footage
4. **Encoding & Alignment:** Applied One-Hot Encoding (`pd.get_dummies`) with left-alignment across training and test splits to guarantee structural feature symmetry.
5. **Target Transformation:** Applied `np.log1p()` transformation on target variable `SalePrice` to compress variance from high-value properties, evaluating predictions back in true dollar terms via `np.expm1()`.
6. **Automated Model Comparison:** Evaluated `RandomForestRegressor` vs. `XGBRegressor` vs. an averaged blend on a 20% validation split, dynamically selecting the best performer to fit 100% of the training data for final submission.

## Files
- `housing_model.ipynb` – Model development & side-by-side comparison pipeline
- `submission.csv` – Final output predictions formatted for Kaggle

## How to Run
```bash
git clone [github.com/S0leja/housing-prediction](https://github.com/S0leja/housing-prediction)
cd housing-prediction
pip install -r requirements.txt
jupyter notebook housing_model.ipynb
