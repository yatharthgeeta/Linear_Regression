# Delhi House Price Prediction — Linear Regression

Predicting house prices in Delhi using Linear Regression, with a full preprocessing pipeline covering missing value imputation, log transformation, and categorical encoding.

## Problem Statement

Given property listing data from Delhi (area, BHK, bathrooms, furnishing, locality, etc.), predict the sale price of a house. This is a regression problem — the target (`Price`) is continuous.

## Dataset

- **Source:** Delhi housing listings dataset (`Delhi_house_data.csv`)
- **Size:** 1,259 rows × 11 columns
- **Features:**

| Feature | Type | Description |
|---|---|---|
| Area | Numeric | Property area (sq. ft.) |
| BHK | Numeric | Number of bedrooms |
| Bathroom | Numeric | Number of bathrooms |
| Furnishing | Categorical | Furnished / Semi-Furnished / Unfurnished |
| Locality | Categorical | Locality/neighborhood name |
| Parking | Numeric | Parking spots available |
| Status | Categorical | Ready to move / Almost ready |
| Transaction | Categorical | New Property / Resale |
| Type | Categorical | Apartment / Builder Floor |
| Per_Sqft | Numeric | Price per sq. ft. |
| **Price** | Numeric | **Target variable** |

Missing values were present in `Bathroom`, `Furnishing`, `Parking`, `Type`, and `Per_Sqft`.

## Approach

### 1. Exploratory Data Analysis
- Distribution plots (histograms, KDE) for all numeric features
- Boxplots to visually inspect outliers
- IQR-based outlier counting per feature
- Skewness calculation to flag features needing transformation
- Bar plots for categorical variable frequency

### 2. Feature Engineering
- **Missing value imputation:**
  - Numeric (`Bathroom`, `Per_Sqft`, `Parking`) → mean imputation
  - Categorical (`Furnishing`, `Type`) → mode imputation
- **Transformation:** Log transform (`log1p`) applied to `Area`, `Price`, and `Per_Sqft` to correct right-skewed distributions
- **Encoding:**
  - Label Encoding for `Furnishing`, `Status`, `Transaction`, `Type`
  - Target Encoding for `Locality` (high-cardinality categorical, encoded against `Price`)

### 3. Model Training
- 80/20 train-test split (`random_state=42`)
- Training set: 1,007 rows | Test set: 252 rows
- Model: `sklearn.linear_model.LinearRegression`

## Why Linear Regression

Linear Regression was chosen as a baseline/interpretable model to establish how well a linear combination of property features explains price variation, and to examine feature-level coefficients before moving to more complex models (e.g., Random Forest, XGBoost) for comparison.

## Results

| Metric | Value |
|---|---|
| MAE | 0.33 |
| MSE | 0.18 |
| RMSE | 0.43 |
| R² Score | 0.8351 |

*(Metrics computed on the log-transformed target; an R² of 0.835 means the model explains ~83.5% of the variance in log(Price) on unseen test data.)*

### Feature Coefficients

| Feature | Coefficient |
|---|---|
| Locality | 1.178 |
| Area | 0.354 |
| Bathroom | 0.263 |
| Per_Sqft | 0.226 |
| BHK | 0.158 |
| Type | -0.126 |
| Status | -0.106 |
| Furnishing | -0.029 |
| Transaction | -0.024 |
| Parking | 0.002 |

`Locality` (target-encoded) and `Area` have the strongest positive influence on price — intuitive, since location and size are the biggest price drivers in real estate.

## Repository Structure

```
House-Price-Prediction-Linear-Regression/
├── data/
│   ├── Delhi_house_data.csv              # Raw dataset
│   ├── Cleaned_Delhi_House_Data.csv      # After missing value imputation
│   └── Preprocessed_Delhi_House_Data.csv # After transformation + encoding
├── House_Price_Prediction.ipynb          # Full EDA -> preprocessing -> model notebook
├── House_Price_Pipeline.ipynb            # Reusable preprocessing/prediction pipeline
├── House_price_prediction_pipeline.pkl   # Saved trained pipeline
└── README.md
```

## Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `category_encoders`

## What I'd Improve Next

- Compare against Ridge/Lasso to check if regularization improves generalization on this feature set
- Try Random Forest / XGBoost on the same preprocessed data for a direct model comparison
- Engineer a `price_per_bhk` or locality-tier feature to reduce reliance on target encoding alone
- Cross-validation instead of a single train-test split for more robust metric estimates
