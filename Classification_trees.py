import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
import sklearn.model_selection as skm
from sklearn.metrics import f1_score
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS

from sklearn.tree import (DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR, plot_tree, export_text)
from sklearn.metrics import (accuracy_score, log_loss)
from sklearn.ensemble import (RandomForestClassifier as RF, GradientBoostingRegressor as GBR)
from ISLP.bart import BART

# for classification
df_OSA = pd.read_excel("OSA_extreme_male.xlsx")

columns = df_OSA.columns.tolist()
columns = [c for c in columns if c not in ["IAH", "Patient", "OSA", "Gender"]]

#Defining OSA = severe as 1 and OSA = healthy as 0
df_OSA["Class"] = np.where((df_OSA["OSA"] == "Severe"), 1, 0)

model = MS(df_OSA, intercept=False)
D = model.fit_transform(df_OSA)
feature_names = list(D.columns)
print("Feature_names:", feature_names)

X = np.array(df_OSA[columns])
y = np.array(df_OSA['Class'])

(X_train, 
 X_test, y_train, 
 y_test) = skm.train_test_split(X, 
                                y, 
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
y_pred = best_.predict(X_test)
mean = np.mean((y_test - y_pred)**2)
#f_one = f1_score(y_test, y_pred)

print("MSE mean pruning:", mean)
#print("f1 score:", f_one)
print("avg prediction error pruning:", np.sqrt(mean))

ax = subplots(figsize=(12, 12))[1]
plot_tree(G.best_estimator_, feature_names=feature_names, ax=ax);

print("\n")

#bagging
print("--- Bagging ---")
bag_OSA = RF(max_features=X_train.shape[1],
                 n_estimators=500, random_state=0).fit(X_train, y_train)

ax = subplots(figsize=(8 ,8))[1]
y_pred = bag_OSA.predict(X_test)
ax.scatter(y_pred , y_test)
mean = np.mean((y_test - y_pred)**2)

f_one = f1_score(y_test, y_pred)


print("mean bagging:", mean)
print("avg prediction error bagging", np.sqrt(mean))
print("f1 score:", f_one)
print("\n")


# Random forest
print("--- RF ---")
RF_OSA = RF(max_features=3,
                 n_estimators=250, random_state=0).fit(X, y)

y_hat_RF = RF_OSA.predict(X)
f_one = f1_score(y, y_hat_RF)

mean = np.mean((y - y_hat_RF)**2)
print("mean forest before cv: ", mean)
print("f1 before cv", f_one)

kfold = skm.KFold(10, shuffle=True, random_state=10)
grid = skm.GridSearchCV(RF_OSA, {'ccp_alpha': ccp_path.ccp_alphas}, 
                        refit=True, 
                        cv=kfold, 
                        scoring='neg_mean_squared_error')
G = grid.fit(X_train, y_train)

best_ = grid.best_estimator_
y_pred = best_.predict(X)
mean = np.mean((y - y_pred)**2)

f_one = f1_score(y, y_pred)



print("mean forest after cv: ", mean)
print("avg prediction error forest:", np.sqrt(mean))
print("f1 score:", f_one)
print("\n")



#plt.show()