import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS

from sklearn.tree import (DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR, plot_tree, export_text)
from sklearn.metrics import (accuracy_score, log_loss)
from sklearn.ensemble import (RandomForestRegressor as RF, GradientBoostingRegressor as GBR)
from ISLP.bart import BART

# regression tree

#importing the files
df_OSA = pd.read_excel("OSA_DB_UPM_Clean_men.xlsx")
df_OSA['BMI'] = df_OSA.Weight / (df_OSA.Height/100)**2

#df_OSA = pd.read_excel("OSA_extreme_male.xlsx")

print(df_OSA.head())

df = df_OSA[['Weight','Height','Age','Cervical','BMI']]

model = MS(df, intercept=False)
D = model.fit_transform(df)
feature_names = list(D.columns)
print("Feature_names:", feature_names)
X = np.asarray(D)

(X_train, 
 X_test, y_train, 
 y_test) = skm.train_test_split(X, 
                                df_OSA['IAH'], 
                                test_size=0.3, 
                                random_state=0)

reg = DTR(max_depth=3)
reg.fit(X_train, y_train)
ax = subplots(figsize=(12, 12))[1]
plot_tree(reg, feature_names=feature_names,ax=ax);

ccp_path = reg.cost_complexity_pruning_path(X_train, y_train)
kfold = skm.KFold(5, shuffle=True, random_state=10)
grid = skm.GridSearchCV(reg, {'ccp_alpha': ccp_path.ccp_alphas}, 
                        refit=True, 
                        cv=kfold, 
                        scoring='neg_mean_squared_error')
G = grid.fit(X_train, y_train)

best_ = grid.best_estimator_
mean = np.mean((y_test - best_.predict(X_test))**2)
print("MSE mean pruning:", mean)
print("avg prediction error pruning:", np.sqrt(mean))

ax = subplots(figsize=(12, 12))[1]
plot_tree(G.best_estimator_, feature_names=feature_names, ax=ax);

print("\n")

#bagging
print("--- Bagging ---")
bag_OSA = RF(max_features=X_train.shape[1],
                 n_estimators=500, random_state=0).fit(X_train, y_train)

ax = subplots(figsize=(8 ,8))[1]
y_hat_bag = bag_OSA.predict(X_test)
ax.scatter(y_hat_bag , y_test)
mean = np.mean((y_test - y_hat_bag)**2)

print("mean bagging:", mean)
print("avg prediction error bagging", np.sqrt(mean))
print("\n")


# Random forest
print("--- RF ---")
RF_OSA = RF(max_features=3,
                 n_estimators=250, random_state=0).fit(X, df_OSA["IAH"])

y_hat_RF = RF_OSA.predict(X)

mean = np.mean((df_OSA["IAH"] - y_hat_RF)**2)
print("mean forest before cv: ", mean)

kfold = skm.KFold(10, shuffle=True, random_state=10)
grid = skm.GridSearchCV(RF_OSA, {'ccp_alpha': ccp_path.ccp_alphas}, 
                        refit=True, 
                        cv=kfold, 
                        scoring='neg_mean_squared_error')
G = grid.fit(X_train, y_train)

best_ = grid.best_estimator_
mean = np.mean((df_OSA["IAH"] - best_.predict(X))**2)

print("mean forest after cv: ", mean)
print("avg prediction error forest:", np.sqrt(mean))
print("\n")

feature_imp = pd.DataFrame({'importance':RF_OSA.feature_importances_}, index=feature_names)
print("feature importance:", feature_imp.sort_values(by='importance', ascending=False))

fig, ax = plt.subplots()
feature_imp.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")

#plt.show()