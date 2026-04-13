import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve, train_test_split, StratifiedKFold
from sklearn.metrics import log_loss

# Load Data from pkl file
with open("best_trial_payload.pkl", "rb") as f:
    payload = pickle.load(f)

df = payload["df"].dropna(subset=['target']).reset_index(drop=True)
X = df[payload["selected_features"]]
y = df['target'].astype(int)
params = payload["best_params"]

# Use the same XGBoost hyperparameters
xgb_config = {
    "objective": "multi:softprob",
    "num_class": 5,
    "eval_metric": "mlogloss",
    "learning_rate": payload["best_params"]["learning_rate"],
    "max_depth": int(payload["best_params"]["max_depth"]),
    "n_estimators": int(payload["best_params"]["n_estimators"]),
    "reg_lambda": payload["best_params"]["reg_lambda"],
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": 1
}

# Log Loss vs Estimators
# We split the data once to watch the loss drop over "time" (iterations)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = XGBClassifier(**xgb_config)
# eval_set allows XGBoost to track the loss at every single step
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

results = model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Training Loss', color='#1B6CA8')
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Validation Loss', color='#C0392B')
plt.title('Multiclass Logarithmic Loss vs Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Multiclass Logarithmic Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Loss_vs_Estimators.png", dpi=500)
plt.show()




# Stratified K-fold for Log Loss vs Estimators
# Use 5 or 10 folds to get a representative curve. In this case, 10 is used.  
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_fold_train_loss = []
all_fold_val_loss = []

for train_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**xgb_config)
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
    
    results = model.evals_result()
    all_fold_train_loss.append(results['validation_0']['mlogloss'])
    all_fold_val_loss.append(results['validation_1']['mlogloss'])

# Average the loss across all folds for each estimator
mean_train_loss = np.mean(all_fold_train_loss, axis=0)
mean_val_loss = np.mean(all_fold_val_loss, axis=0)


plt.figure(figsize=(10, 6))
plt.plot(x_axis, mean_train_loss, label='Training Loss', color='#1B6CA8')
plt.plot(x_axis, mean_val_loss, label='Validation Loss', color='#C0392B')
plt.title('Multiclass Logarithmic Loss vs Number of Estimators using a Stratified 10-Fold Cross Validation')
plt.xlabel('Number of Estimators')
plt.ylabel('Multiclass Logarithmic Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Loss_vs_Estimators_10_Fold_Cross_Validation.png", dpi=500)
plt.show()




# Learning Curve (Accuracy vs. Data Size)
print("Calculating Learning Curve...")
train_sizes, train_scores, val_scores = learning_curve(
    XGBClassifier(**xgb_config), X, y, 
    cv=10, # 10-fold cross-validation is used to evaluate the model's performance at different training set sizes
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1_macro', 
    n_jobs=-1 # Uses all cores to compute the learning curve
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='#1B6CA8', label='Training Macro-F1')
plt.plot(train_sizes, val_mean, 's-', color='#C0392B', label='Cross-Validation Macro-F1')
plt.title('Learning Curve: Macro-F1 Score vs. Training Set Size')
plt.xlabel('Number of Proteins for Training')
plt.ylabel('Macro-F1 Score')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig("Learning_Curve.png", dpi=500)
plt.show()

print("\nAll plots saved!")
