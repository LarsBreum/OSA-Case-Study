import numpy as np , pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (summarize, poly, ModelSpec as MS)
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

from pygam import (s as s_gam, l as l_gam, f as f_gam, LinearGAM, LogisticGAM)
from ISLP.transforms import (BSpline, NaturalSpline)
from ISLP.models import bs, ns
from ISLP.pygam import (approx_lam, degrees_of_freedom, plot as plot_gam, anova as anova_gam)

#importing the files
#file = 'OSA_extreme_male.xlsx'
file = 'OSA_DB_UPM_Clean_men.xlsx'
xl = pd.ExcelFile(file)
df_OSA = pd.read_excel("OSA_extreme_male.xlsx")

#df_OSA = pd.read_excel("OSA_DB_UPM_Clean_men.xlsx")
#df_OSA['BMI'] = df_OSA.Weight / (df_OSA.Height/100)**2

print(df_OSA)

df = df_OSA[['Weight','Height','Age','Cervical','BMI']]
print("Number of degrees you want to fit")
degrees = int(input())



# fitting a degree=4 to the data
y = df_OSA['IAH']
bmi = df_OSA['BMI']
x = MS(['Age', 'Cervical', 'BMI']).fit_transform(df_OSA)

poly_BMI = MS([ poly('BMI', degree=degrees)]).fit(df_OSA)
M = sm.OLS(y, poly_BMI.transform(df_OSA)).fit()
print(summarize(M))
print("R_squared: " + str(M.rsquared))

# create grid of values for bmi
bmi_grid = np.linspace(bmi.min(), bmi.max(), 100)
bmi_df = pd.DataFrame ({'BMI': bmi_grid})

# plot
def plot_bmi_fit(bmi_df, basis, title):
    X = basis.transform(df_OSA)
    Xnew = basis.transform(bmi_df)
    M = sm.OLS(y, X).fit()
    preds = M.get_prediction(Xnew)
    bands = preds.conf_int(alpha=0.05)
    fig , ax = subplots(figsize =(8 ,8))
    ax.scatter(bmi, y, facecolor='gray', alpha=0.5)

    for val, ls in zip([preds.predicted_mean, bands[:,0],bands[:,1]],['b','r--','r--']):
                       ax.plot(bmi_df.values, val, ls, linewidth=3)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel('BMI', fontsize =20)
    ax.set_ylabel('AHI ', fontsize =20);
    return ax


plot_bmi_fit(bmi_df, poly_BMI, 'Degree ' + str(degrees) + ' Polynomial ');

# ANOVA (analysis of variance test)
print("--- ANOVA ---")

models = [MS([poly('BMI', degree=d)])
    for d in range(1, 6)]

Xs = [model.fit_transform(df_OSA) for model in models]
anova_lm(*[sm.OLS(y, X_).fit()
    for X_ in Xs])

print(summarize(M))
print("R_squared: " + str(M.rsquared))

# cross validation
print("--- CROSS VALIDATION ---")

X = poly_BMI.transform(df_OSA)
severe = df_OSA['IAH'] = y > 30 # shorthand
glm = sm.GLM(y > 30,
             X,
             family=sm.families.Binomial ())

B = glm.fit()
print(summarize(B))

newX = poly_BMI.transform(bmi_df)
preds = B.get_prediction(newX)
bands = preds.conf_int(alpha=0.05)

fig, ax = subplots(figsize=(8 ,8))
rng = np.random.default_rng(0)

ax.scatter(bmi + 0.2 * rng.uniform(size=y.shape [0]),
            np.where(severe, 0.198 , 0.002),
            fc='gray', marker='|')

for val, ls in zip([ preds.predicted_mean, bands [:,0], bands [:,1]], ['b','r--','r--']):
    ax.plot(bmi_df.values , val , ls , linewidth=3)

ax.set_title('Degree ' + str(degrees) + ' Polynomial', fontsize =20)
ax.set_xlabel('BMI', fontsize =20)
ax.set_ylim ([0 ,1])
ax.set_ylabel('P(AHI > 30)', fontsize=20);

cut_bmi = pd.qcut(bmi, 4)
print(summarize(sm.OLS(y, pd.get_dummies(cut_bmi)).fit()))

# Polinomial logistic regression

print("--- POLINOMIAL LOGISTIC REGRESSION ---")

X = poly_BMI.transform(df_OSA)
severe = df_OSA['severe '] = y > 30 # shorthand
glm = sm.GLM(y > 30, X, family=sm.families.Binomial ())
B = glm.fit()
print(summarize(B))

newX = poly_BMI.transform(bmi_df)
preds = B.get_prediction(newX)
bands = preds.conf_int(alpha=0.05)

fix, ax = subplots(figsize=(8,8))
rng = np.random.default_rng(0)
ax.scatter(bmi + 
           0.2 + rng.uniform(size=y.shape[0]),
           np.where(severe, 0.198, 0.002),
           fc='gray',
           marker='|')
for val, ls in zip([preds.predicted_mean,
                    bands[:,0],
                    bands[:,1]],
                    ['b', 'r--', 'r--']):
    ax.plot(bmi_df.values, val, ls, linewidth=3)

ax.set_title('Degree ' + str(degrees) + ' Polynomial', fontsize =20)
ax.set_xlabel('BMI', fontsize =20)
ax.set_ylim ([0 ,1])
ax.set_ylabel('P(AHI > 30) ', fontsize =20);

# quartiles
print("--- QUARTILES ---")
cut_bmi = pd.cut(bmi, [20, 25, 30, 35, 52])
print(summarize(sm.OLS(y, pd.get_dummies(cut_bmi)).fit()))

""" The mean AHI for people in the "normal weight" category is 10,8
    19 for "overweight", 30 for "obese" and 44 for "extreme obesity" """


# Additive Models with several terms
print("--- ADDITIVE MODELS WITH SEVERAL TERMS ---")

ns_bmi = NaturalSpline(df=4).fit(bmi)
ns_age = NaturalSpline(df=5).fit(df_OSA['Age'])
ns_cervical = NaturalSpline(df=5).fit(df_OSA['Cervical'])

# BMI on AHI

Xs = [ns_bmi.transform(bmi),
       ns_age.transform(df_OSA['Age']),
       ns_cervical.transform(df_OSA['Cervical'])]

X_bh = np.hstack(Xs)
gam_bh = sm.OLS(y, X_bh).fit()

bmi_grid = np.linspace(bmi.min(),
                       bmi.max(),
                       100)

X_bmi_bh = X_bh.copy()[:100]
X_bmi_bh [:] = X_bh [:].mean (0)[None ,:]
X_bmi_bh [:,:4] = ns_bmi.transform(bmi_grid)
preds = gam_bh.get_prediction(X_bmi_bh)
bounds_bmi = preds.conf_int(alpha=0.05)
partial_bmi = preds.predicted_mean
center = partial_bmi.mean()
partial_bmi -= center
bounds_bmi-= center

fig , ax = subplots(figsize=(8 ,8))
ax.plot(bmi_grid , partial_bmi , 'b', linewidth =3)
ax.plot(bmi_grid , bounds_bmi [:,0], 'r--', linewidth =3)
ax.plot(bmi_grid , bounds_bmi [:,1], 'r--', linewidth =3)
ax.set_xlabel('BMI')
ax.set_ylabel('Effect on AHI')
ax.set_title('Partial dependence of BMI on AHI', fontsize=20);

#Cervical on AHI
cervical_grid = np.linspace (2003 , 2009, 100)
cervical_grid = np.linspace(df_OSA['Cervical'].min(),
                        df_OSA['Cervical'].max(),
                        100)
X_cervical_bh = X_bh.copy () [:100]
X_cervical_bh [:] = X_bh [:]. mean (0)[None ,:]
X_cervical_bh [:,4:9] = ns_cervical.transform(cervical_grid)
preds = gam_bh.get_prediction(X_cervical_bh)
bounds_cervical = preds.conf_int(alpha =0.05)
partial_cervical = preds.predicted_mean
center = partial_cervical.mean ()
partial_cervical -= center
bounds_cervical -= center
fig, ax = subplots(figsize =(8 ,8))
ax.plot(cervical_grid, partial_cervical , 'b', linewidth =3)
ax.plot(cervical_grid, bounds_cervical [:,0], 'r--', linewidth =3)
ax.plot(cervical_grid, bounds_cervical [:,1], 'r--', linewidth =3)
ax.set_xlabel('Cervical')
ax.set_ylabel('Effect on AHI')
ax.set_title('Partial dependence of Cervical on AHI', fontsize=20);

# GAM



plt.show()