import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.utils.class_weight import compute_sample_weight
from matplotlib.ticker import FuncFormatter, MaxNLocator

# --- 1. DATA CONFIGURATION ---
train_file = "features_ML_POSITIVE_1_NEGATIVE_1_improved_07_04.csv"

HEME_pdbs = ["1YZP", "1CRI", "4NVA", "4D3T", "3QM8", "2J18", "1VXA", "1PHA", "7RKR", 
             "1DVE", "1DS4", "1SOG", "3P6N", "3WEC", "1D3S", "3FKG", "7PQ1", "3P6U", "8EWQ", "4NVN"]
FAD_pdbs  = ["8A1H", "1REO", "1DOB", "8Z45", "3NLC", "3FG2", "3AXB", "2AQJ", "8Z3G", "1PXC", "8JDY", 
             "8Z44", "6RTM", "1OGI", "7OR2", "5GRT", "1E1N", "1E1L", "1OWP", "2YYM"]
ZN_2_pdbs = ["7G3B", "2H2I", "1FT7", "7Y2E", "4CXO", "3T87", "3PB9", "2WEG", "7UHI", 
             "2E0P", "7HUX", "7XJ4", "6KM5", "1YSO", "6UGR", "5NS5", "1HFC", "1H3N", "2AW1", "3ORJ"]
CU_1_pdbs = ["2CAL", "1DZ0", "3QQX", "3F7L", "3DSO", "2FT7", "4MAI", "1OF0", "4DPC", "5SSZ", "5NQM", 
             "3NT0", "1JZG", "4BTE", "2XV2", "1BXA", "1A8Z", "5MSZ", "2CAK", "7OG7"]
cofactorless_pdbs = [
    "2BQQ", "1FHG", "4TTP", "4G2E", "3SYJ", "2O6X", "1YU7", "1UIA", 
    "7IE4", "1HE9", "1GVL", "1WKA", "3R63", "3ZM2", "1FQI", "3JSN", 
    "7ID0", "3RLE", "7W8U", "4WJ1", "1BFG", "2WLW", "7IEG", "5R2I", 
    "4WEI", "2TMY", "3PTW", "5P7D", "1YRV", "1W7B", "6K1Y", "1XAA", 
    "5ZO0", "5RCS", "4OEF", "1KAB", "8C9L", "2FGT", "6J6E", "1SNQ", 
    "5DZ9", "6AO9", "3H2G", "1QG5", "1KWB", "3V75", "5C10", "1SZT", 
    "3ZPJ", "1YHW", "6K5M", "4WFU", "7Y5J", "6CB6", "2Y7N", "6F4M", 
    "5YDN", "3NPO", "4P9L", "1QTO", "3ASD", "4G14", "2YLH", "132L", 
    "6JUI", "4QRZ", "3RFY", "5P1Q", "1MJZ", "3VSR", "1HEP", "5NPT", 
    "6VV4", "4PMD", "1PEN", "3O7K", "5NJM", "3OUV", "9L1R", "6RU3"
]

def assign_label(pdb_code):
    if pdb_code in HEME_pdbs: return 0
    elif pdb_code in FAD_pdbs: return 1
    elif pdb_code in ZN_2_pdbs: return 2
    elif pdb_code in CU_1_pdbs: return 3
    elif pdb_code in cofactorless_pdbs: return 4
    else: return None

# --- 2. DATA LOADING & FEATURES ---
selected_features = [
    "Ranking_Score", "Avg_MW", "NCD", "RCO", "Norm_ALA", "Norm_ARG", "Norm_ASN", 
    "Norm_ASP", "Norm_CYS", "Norm_GLN", "Norm_GLU", "Norm_GLY", "Norm_HIS", "Norm_ILE", 
    "Norm_LEU", "Norm_LYS", "Norm_MET", "Norm_PHE", "Norm_PRO", "Norm_SER", "Norm_THR", 
    "Norm_TRP", "Norm_TYR", "Norm_VAL", "Weight_ALA", "Weight_ARG", "Weight_ASN", "Weight_ASP", 
    "Weight_CYS", "Weight_GLN", "Weight_GLU", "Weight_GLY", "Weight_HIS", "Weight_ILE", "Weight_LEU", 
    "Weight_LYS", "Weight_MET", "Weight_PHE","Weight_PRO","Weight_SER","Weight_THR","Weight_TRP","Weight_TYR","Weight_VAL"
]

df = pd.read_csv(train_file)
df['PDB_Code'] = df['PDB_Code'].str.upper()
df['target'] = df['PDB_Code'].apply(assign_label)
full_df = df.dropna(subset=['target']).drop_duplicates(subset=['PDB_Code']).reset_index(drop=True)

X = full_df[selected_features]
y = full_df['target'].astype(int)
class_names = ["HEME", "FAD", "Zn2+", "Cu+", "Cofactorless"]

# --- 3. LOOCV LOOP ---
loo = LeaveOneOut()
y_true = []
y_pred_loocv = []
results_list = [] # For CSV export

print(f"Starting LOOCV on {len(full_df)} samples...")
print(f"{'PDB':<6} | {'Actual':<12} | {'Predicted':<12} | {'Confidence Breakdown'}")
print("-" * 80)

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pdb_id = full_df.iloc[test_index[0]]['PDB_Code']
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    xgb_params = dict(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        max_depth=5,
        learning_rate=0.01,
        reg_lambda=3,
        random_state=42,
        tree_method="hist",
        device="cuda"
    )

    model = XGBClassifier(**xgb_params, n_estimators=100)
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    # 1. Store Predictions
    actual_idx = y_test.values[0]
    pred_idx = model.predict(X_test)[0]
    probs = model.predict_proba(X_test)[0]

    y_true.append(actual_idx)
    y_pred_loocv.append(pred_idx)

    # 2. Print console table row
    conf_str = " | ".join([f"{class_names[j]}: {probs[j]*100:.1f}%" for j in range(5)])
    mistake = " [X]" if actual_idx != pred_idx else ""
    print(f"{pdb_id:<6} | {class_names[actual_idx]:<12} | {class_names[pred_idx]:<12}{mistake} | {conf_str}")

    # 3. Save for CSV
    results_list.append({
        "PDB_Code": pdb_id,
        "Actual": class_names[actual_idx],
        "Predicted": class_names[pred_idx],
        "Correct": 1 if actual_idx == pred_idx else 0, # This column can be used to easily calculate accuracy in Excel or other tools.
        "Prob_HEME": probs[0], "Prob_FAD": probs[1], "Prob_Zn2": probs[2], "Prob_Cu1": probs[3], "Prob_Cofactorless": probs[4]
    })

# EXPORT & FINAL METRICS
pd.DataFrame(results_list).to_csv("LOOCV_Detailed_Results.csv", index=False)

loocv_acc = accuracy_score(y_true, y_pred_loocv)
print(f"\nFinal LOOCV Accuracy: {loocv_acc * 100:.2f}%")

# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred_loocv)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title(f"LOOCV Confusion Matrix (Accuracy: {loocv_acc*100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("LOOCV_confusion_matrix_with_control.png", dpi=500)
plt.close()

# --- 6. FEATURE IMPORTANCE ---
final_weights = compute_sample_weight(class_weight='balanced', y=y)
final_model = XGBClassifier(**xgb_params, n_estimators=100)
final_model.fit(X, y, sample_weight=final_weights, verbose=False)

plt.figure(figsize=(10, 8))
importances = pd.Series(final_model.feature_importances_, index=selected_features)
importances.nlargest(10).plot(kind='barh', color='teal')
plt.title("Top 10 Features (5-Class Model)")
plt.xlabel("Feature Importance (Gain)") 
plt.gca().invert_yaxis()
plt.savefig("LOOCV_feature_importance_with_control.png", dpi=500)
plt.close()