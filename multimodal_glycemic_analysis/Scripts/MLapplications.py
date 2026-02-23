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
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel



# =========================
# Reload cleaning and preprocessing
# =========================

def run_script(path: Path) -> None:
    """
    Esegue uno script .py come se fosse lanciato da terminale.
    Questo fa scattare il blocco:
        if __name__ == "__main__": main()
    """
    runpy.run_path(str(path), run_name="__lag__")

if True:
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

cv = TimeSeriesSplit(n_splits=5)








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


for q in [35,50,65]:
    q = q/100
    y_bin = (y >= y.quantile(q)).astype(int)
    #è 1 quando la glicemia va male e 0 quando va bene

    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=10000, class_weight="balanced"))
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

NB_df= pd.DataFrame(columns=["q", "AUC"])


for q in [35,50,65]:
    q = q/100
    y_bin = (y >= y.quantile(q)).astype(int)
    #è 1 quando la glicemia va male e 0 quando va bene

    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("nb", GaussianNB())
        ])

    
    grid = GridSearchCV(
        pipe,
        param_grid ={},
        cv=cv,
        scoring="roc_auc",   # meglio di accuracy
        n_jobs=-1
    )
    
    grid.fit(X, y_bin)
    
    NB_df.loc[len(NB_df)] = [q,grid.best_score_]

NB_df = NB_df[NB_df["AUC"]==NB_df["AUC"].max()]

print(f"\nBest NaiveBayes -> | quantile = {NB_df.iloc[0,0]} | AUC = {NB_df.iloc[0,1]}")

# =========================
# 6) LDA
# =========================

LDA_df= pd.DataFrame(columns=["q", "AUC"])


for q in [35,50,65]:
    q = q/100
    y_bin = (y >= y.quantile(q)).astype(int)
    #è 1 quando la glicemia va male e 0 quando va bene

    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis())
        ])

    
    grid = GridSearchCV(
        pipe,
        param_grid ={},
        cv=cv,
        scoring="roc_auc",   # meglio di accuracy
        n_jobs=-1
    )
    
    grid.fit(X, y_bin)
    
    LDA_df.loc[len(LDA_df)] = [q,grid.best_score_]

LDA_df = LDA_df[LDA_df["AUC"]==LDA_df["AUC"].max()]

print(f"\nBest LDA -> | quantile = {LDA_df.iloc[0,0]} | AUC = {LDA_df.iloc[0,1]}")

# =========================
# 7) SVM 
# =========================
SVM_df= pd.DataFrame(columns=["q", "AUC","kernel"])


for q in [35,50,65]:
    q = q/100
    y_bin = (y >= y.quantile(q)).astype(int)
    #è 1 quando la glicemia va male e 0 quando va bene
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(class_weight="balanced"))  # decision_function disponibile
    ])
    
    param_grid = [
        # LINEARE
        {
            "svm__kernel": ["linear"],
            "svm__C": np.logspace(-3, 3, 25)
        },
        # RBF
        {
            "svm__kernel": ["rbf"],
            "svm__C": np.logspace(-1, 2, 15),
            "svm__gamma": np.logspace(-4, -1, 15)  # gamma = 1/(2*sigma^2)
        }
    ]
    
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    grid.fit(X, y_bin)
    SVM_df.loc[len(SVM_df)] = [q, grid.best_score_, grid.best_params_["svm__kernel"]]

best = SVM_df.loc[SVM_df["AUC"].idxmax()]
print(f"Best SVM -> q = {best['q']} | AUC = {best['AUC']} | kernel = {best['kernel']}")

# =========================
# 7) SVM PCA
# =========================
SVM_df= pd.DataFrame(columns=["q", "AUC","kernel"])


for q in [35,50,65]:
    q = q/100
    y_bin = (y >= y.quantile(q)).astype(int)
    #è 1 quando la glicemia va male e 0 quando va bene
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10)),
        ("svm", SVC(class_weight="balanced"))  # decision_function disponibile
    ])
    
    param_grid = [
        # LINEARE
        {
            "svm__kernel": ["linear"],
            "svm__C": np.logspace(-3, 3, 25)
        },
        # RBF
        {
            "svm__kernel": ["rbf"],
            "svm__C": np.logspace(-1, 2, 15),
            "svm__gamma": np.logspace(-4, -1, 15)  # gamma = 1/(2*sigma^2)
        }
    ]
    
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    grid.fit(X, y_bin)
    SVM_df.loc[len(SVM_df)] = [q, grid.best_score_, grid.best_params_["svm__kernel"]]

best = SVM_df.loc[SVM_df["AUC"].idxmax()]
print(f"Best SVM (PCA) -> q = {best['q']} | AUC = {best['AUC']} | kernel = {best['kernel']}")


# =========================
# 7) SVM (log selector)
# =========================
SVM_df= pd.DataFrame(columns=["q", "AUC","kernel"])


for q in [35,50,65]:
    q = q/100
    y_bin = (y >= y.quantile(q)).astype(int)
    #è 1 quando la glicemia va male e 0 quando va bene
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectFromModel(LogisticRegression(l1_ratio=1,solver="liblinear"))),
        ("svm", SVC(class_weight="balanced"))  # decision_function disponibile
    ])
    
    param_grid = [
        # LINEARE
        {
            "svm__kernel": ["linear"],
            "svm__C": np.logspace(-3, 3, 25)
        },
        # RBF
        {
            "svm__kernel": ["rbf"],
            "svm__C": np.logspace(-1, 2, 15),
            "svm__gamma": np.logspace(-4, -1, 15)  # gamma = 1/(2*sigma^2)
        }
    ]
    
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    grid.fit(X, y_bin)
    SVM_df.loc[len(SVM_df)] = [q, grid.best_score_, grid.best_params_["svm__kernel"]]

best = SVM_df.loc[SVM_df["AUC"].idxmax()]
print(f"Best SVM (log selector) -> q = {best['q']} | AUC = {best['AUC']} | kernel = {best['kernel']}")


