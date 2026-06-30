import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.SeqUtils.ProtParam import ProteinAnalysis as IP

if __name__ == "__main__":
    with open("best_trial_payload.pkl", "rb") as f:
        payload = pickle.load(f)

    df_final = payload["df"].dropna(subset=['target']).reset_index(drop=True)
    selected_features = payload["selected_features"]
    best_radius = payload["best_params"]["radius"]
    X, y = df_final[selected_features], df_final['target'].astype(int)

    best_params = {
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

    print("Training final production model on all 868 development proteins...")
    final_weights = compute_sample_weight(class_weight='balanced', y=y)
    final_model = XGBClassifier(**best_params)
    final_model.fit(X, y, sample_weight=final_weights, verbose=False)

    # Save the master model
    with open("final_xgboost_model.pkl", "wb") as f:
        pickle.dump(final_model, f)

