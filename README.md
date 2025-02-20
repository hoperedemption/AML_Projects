
# Advanced Machine Learning Projects
These 3 projects, were part of the AS 2024 **Advanced Machine Learning** course, developed by my friend Sebastian and myself.

## Project 1 - Age Regression Based on MRI Brain Scans

### Overview
The goal is to predict age from MRI brain scans using various ML techniques. The focus was on data preprocessing, feature selection, and model ensembling to achieve optimal regression performance.

### Key Steps
### Data Preprocessing
- **Missing Values:** KNN imputation provided the best R² score.
- **Scaling Methods:** 
  - **Quantile Transformer** improved performance by normalizing data distribution.
  - **RobustScaler** enhanced results for SVR and Tree Regressors by reducing outlier impact.

### Outlier Detection
- Used **Elliptic Envelope (GMM with one component)** after PCA transformation to detect and remove outliers.

### Feature Selection
- **Correlation-based filtering:** Removed features with correlation < 0.2.
- **Multicollinearity reduction:** Eliminated highly correlated features.

## Models Used
- **Gaussian Process Regression (Best Model)**
  - Optimized kernel: **RationalQuadratic * Matern * DotProduct**.
  - Implemented with `gpytorch`.
- **Support Vector Regression (SVR)** optimized with RobustScaler.
- **Tree-Based Models:** LightGBM, ExtraTreeRegressor, XGBoost.

## Final Model and Submission
- **Ensemble Model** combining **Gaussian Process Regressor, SVR, and LightGBM** to reduce overfitting and improve performance.

![Outlier Removal](Project1/Outliers.png)
