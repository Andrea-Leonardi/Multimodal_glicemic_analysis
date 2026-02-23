import pandas as pd
import numpy as np
import config as cfg
from pathlib import Path
import runpy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression


# =========================
# Reload cleaning and preprocessing
# =========================

def run_script(path: Path) -> None:
    runpy.run_path(str(path), run_name="__lag__")

if False:
    scripts_dir = (Path(__file__).resolve().parents[0])
    run_script(scripts_dir / "Processing_and_descriptive.py")

# =========================
# Load data
# =========================
df = pd.read_parquet(cfg.DATA_PROCESSED).dropna()

target_col = "bg_auc_above_limit_rate"
y = df[target_col]

#tolgo variabili che causano la risposta

X = df.drop(columns=[target_col,"bg_mean","bg_sd","bg_cv","bg_min","bg_median","bg_max","bg_tir_%","bg_tar_%","ins_bolo_tot","ins_tot","ins_basal_tot"])

# =========================
# Defining variables
# =========================

cv = KFold(n_splits=5, shuffle= False)







# =========================
# 1) Lasso
# =========================

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=200000, tol=1e-4))
    ])
param_grid = {
    "lasso__alpha": np.logspace(-3, 3, 80)}

grid = GridSearchCV(pipe, param_grid, cv=cv,scoring="r2",n_jobs=-1)
grid.fit(X,y)

print("Lasso Best R2:", grid.best_score_)
print("Lasso Best alpha:", grid.best_params_["lasso__alpha"])


# =========================
# 2) PCR RidgeCV
# =========================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("ridge", Ridge())
])

param_grid = {
    "pca__n_components": range(2, min(15, X.shape[1])),
    "ridge__alpha": np.logspace(0,5 , 50)
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,
    scoring="r2",
    n_jobs=-1 #uso tutti i core della cpu
)

grid.fit(X, y)

print("\nPCR RidgeCV Best R2:", grid.best_score_)
print("PCR RidgeCV Best n_components:", grid.best_params_["pca__n_components"])
print("PCR RidgeCV Best alpha:", grid.best_params_["ridge__alpha"])




# =========================
# 3) PLS
# =========================

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pls", PLSRegression())
    
    ])
param_grid = {
    "pls__n_components": range(1,min(15, X.shape[1]))
    }

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,
    scoring= "r2",
    n_jobs=-1
    )

grid.fit(X,y)
print("\nPLS Best R2:", grid.best_score_)
print("PLS Best n_components:", grid.best_params_["pls__n_components"])



# =========================
# MODIFY THE TARGET TO BINARY
# =========================
# =========================
# 4) logistic regression
# =========================



logreg_df= pd.DataFrame(columns=["q", "AUC", "C"])


for q in range(25,76, 1):
    q = q/100
    y_bin = (y >= y.quantile(q)).astype(int)
    #è 1 quando la glicemia va male e 0 quando va bene

    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=10000))
        ])
    
    param_grid = {
        "logit__C": np.logspace(-3, 3, 50)  # C = 1/lambda
    }
    
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="roc_auc",   # meglio di accuracy
        n_jobs=-1
    )
    
    grid.fit(X, y_bin)
    
    logreg_df.loc[len(logreg_df)] =[q,grid.best_score_,grid.best_params_["logit__C"]]

logregbest = logreg_df[logreg_df["AUC"]==logreg_df["AUC"].max()]

print(f"\nBest logistic model -> | quantile = {logregbest.iloc[0,0]} | AUC = {logregbest.iloc[0,1]} | C = {logregbest.iloc[0,2]}")

# =========================
# 5) Naive Bayes
# =========================
# =========================
# 6) LDA
# =========================
# =========================
# 7) SVM
# =========================












