from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

transformed_data = pd.read_csv('features.csv')
X = transformed_data.drop(columns=['y'])
y = transformed_data['y']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_val = std_scaler.transform(X_val)

lgbm = LGBMClassifier(n_estimators=1000)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_val)


# gives 0.7977386934673367
print(f1_score(y_val, y_pred, average='micro'))
print(classification_report(y_val, y_pred))