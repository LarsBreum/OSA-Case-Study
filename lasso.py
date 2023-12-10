import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from statsmodels.api import OLS
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import sklearn.model_selection as skm
import sklearn.linear_model as skl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn. cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

df_OSA = pd.read_excel("OSA_Regression.xlsx")
df = df_OSA[['Weight','Height','Age','Cervical','BMI']]

### Picking predictor columns
columns = df_OSA.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["IAH"]]
# Store the variable we'll be predicting on.
target = "IAH"
print('Predictors: ',columns)

X = np.array(df_OSA[columns])
y = np.array(df_OSA['IAH'])

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error

# Assuming you have your data X and y ready
# X is your feature matrix
# y is your target variable

# Split the data into training and test sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alpha_value = 0.01

lasso_model = Lasso(alpha=alpha_value, max_iter=10000)

k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=1)

# Perform cross-validation using cross_val_score
cv_scores = cross_val_score(lasso_model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_scores = -cv_scores

# Print the cross-validation scores
print(f'Cross-Validation MSE Scores for training: {cv_scores}')
print(f'Mean MSE Score for training: {np.mean(cv_scores)}')

# Fit the Lasso model on the entire training data
lasso_model.fit(X_train_scaled, y_train)

# Print the coefficient weights
coefficients = lasso_model.coef_
print('Feature Coefficients:')
for feature, coef in zip(columns, coefficients):
    print(f'{feature}: {coef}')

# Make predictions on the test data
y_pred = lasso_model.predict(X_test_scaled)

# Evaluate the model performance on the test data
mse_test = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse_test}')


lambdas = 10** np.linspace (8, -2, 100) / y.std()
# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.title('Actual vs. Predicted Values for Lasso Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.title('Residual Plot for Lasso Regression')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.show()




n_alphas = 100
alphas = np.logspace(1, -1, n_alphas)
coefs = []
for a in alphas:
    lasso = linear_model.Lasso(alpha=a, fit_intercept=False)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()

ax.plot(alphas, coefs, label=columns)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Lasso coefficients as a function of the regularization")
plt.axis("tight")
plt.show()
