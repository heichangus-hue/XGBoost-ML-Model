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
train_file = "features_ML_POSITIVE_NEGATIVE_1_2_centre_of_mass_plddt_pae_new_15.csv"

HEME_pdbs = ["1YZP", "1CRI", "4NVA", "4D3T", "3QM8", "2J18", "1VXA", "1PHA", "7RKR", "1DVE",
             "1DS4", "1SOG", "3P6N", "3WEC", "1D3S", "3FKG", "7PQ1", "3P6U", "8EWQ", "4NVN",
             "5HLQ", "1JIN", "1BEK", "6HQM", "2J19", "2CYM", "2ACP", "1MKR", "6GEQ", "1IRC",
             "5U5U", "6HQK", "8FDJ", "4NVG", "1GWU", "4ZDY", "3M8M", "1CCC", "1C53", "1H5H",
             "1YYD", "2BLI", "4G8U", "5CMV", "1BES", "4RM4", "1UX8", "6BD7", "5KD1", "5XXI"]

FAD_pdbs = ["8A1H", "1REO", "1DOB", "8Z45", "3NLC", "3FG2", "3AXB", "2AQJ", "8Z3G", "1PXC",
            "8JDY", "8Z44", "6RTM", "1OGI", "7OR2", "5GRT", "1E1N", "1E1L", "1OWP", "2YYM",
            "8CCL", "1TJ1", "1DOC", "9GN5", "3QFS", "3GYJ", "3DJJ", "2BAB", "9DTK", "1QNF",
            "8K41", "9F1Y", "6YRZ", "1OWN", "7RT0", "5KOW", "1E63", "1E39", "1PHH", "3COX",
            "3EF6", "6ICI", "7VJ0", "1DOE", "7C4N", "2XRY", "8Z26", "8JDG", "8X38", "5JCK"]

ZN_2_pdbs = ["7G3B", "2H2I", "1FT7", "7Y2E", "4CXO", "3T87", "3PB9", "2WEG", "7UHI", "2E0P",
             "7HUX", "7XJ4", "6KM5", "1YSO", "6UGR", "5NS5", "1HFC", "1H3N", "2AW1", "3ORJ",
             "7G3R", "2H6S", "1FUA", "7Z0N", "4DEF", "3TGE", "3PN5", "2WHZ", "7VFR", "2E88",
             "7HVJ", "7YHD", "6LD4", "1Z1N", "6V4V", "5NYA", "1HKK", "1H71", "2B5W", "3OY0",
             "3RTT", "6CLD", "6YHE", "1G4J", "6PJV", "3KZZ", "7O2S", "7G68", "7JOB", "6LD1"]

CU_1_pdbs = ["2CAL", "1DZ0", "3QQX", "3F7L", "3DSO", "2FT7", "4MAI", "1OF0", "4DPC", "5SSZ",
             "5NQM", "3NT0", "1JZG", "4BTE", "2XV2", "1BXA", "1A8Z", "5MSZ", "2CAK", "7OG7",
             "5NQN", "2BZC", "1HAW", "2XMW", "2QDW", "2JCW", "2CCW", "1SF3", "4TM7", "1UUY",
             "5KBK", "4DP8", "1RJU", "6QVH", "5KBM", "2IDU", "5SSY", "5ARN", "6L9S", "5NQO",
             "3I9Z", "3F7K", "4DPA", "4F2F", "4P5S", "6R01", "4N3T", "2XV0", "5SSX", "1A3Z"]

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
    "6VV4", "4PMD", "1PEN", "3O7K", "5NJM", "3OUV", "9L1R", "6RU3",
    "5P18", "1MN4", "1CNU", "6HYN", "2NWD", "2F9F", "2CG7", "1QMR",
    "6G9P", "1LNS", "5R3B", "6HHN", "7X3H", "4U64", "1KF7", "5GLX",
    "3P2J", "1DST", "1DG3", "1L18", "2BMJ", "2DX1", "4KIA", "5I4W",
    "1CUC", "4YWF", "1YG2", "6AR0", "5P3O", "5Y30", "4U3V", "3OKQ",
    "2BTZ", "3UMF", "5G0Z", "2OBT", "7BAJ", "7Q6B", "1ANF", "6L2A",
    "7A8B", "1T7N", "5XCN", "3P7K", "3B5O", "2OBR", "1SLL", "2B2H",
    "8X8S", "6LYT", "3AA5", "1LMQ", "1L0B", "3GW3", "1L40", "3DQI",
    "7IHZ", "3BVS", "5IJM", "2ICC", "7AHW", "1EY0", "6EHX", "3ZDE",
    "4R1B", "1OEM", "9DD2", "8HEK", "3GS9", "1JAM", "4X1O", "2PW5",
    "7ICB", "5OYR", "5LZ1", "7VKP", "7NMO", "3E0E", "5DDV", "1XEI",
    "5YNL", "1I9Y", "1F1S", "5P6A", "2D27", "6P1F", "2PII", "9JLA",
    "1JI6", "7L6W", "2DYI", "7P25", "1LIB", "3GYW", "2OBS", "3VYC",
    "5P0D", "7IDX", "3EMI", "1TK1", "3FFM", "3DGJ", "1ZXQ", "5P8C",
    "2IQT", "5Y5D", "8P33", "5RDZ", "5P39", "1IIW", "5K2N", "5P07",
    "1UEK", "4QSC", "6EFR", "2F8H", "1TP0", "4ACO", "3GVQ", "2JDW"
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
    "Ranking_Score", "Global_pLDDT", "Global_PAE", "Avg_MW", "NCD", "RCO", "Norm_ALA", "Norm_ARG", "Norm_ASN", 
    "Norm_ASP", "Norm_CYS", "Norm_GLN", "Norm_GLU", "Norm_GLY", "Norm_HIS", "Norm_ILE", 
    "Norm_LEU", "Norm_LYS", "Norm_MET", "Norm_PHE", "Norm_PRO", "Norm_SER", "Norm_THR", 
    "Norm_TRP", "Norm_TYR", "Norm_VAL", "Weight_ALA", "Weight_ARG", "Weight_ASN", "Weight_ASP", 
    "Weight_CYS", "Weight_GLN", "Weight_GLU", "Weight_GLY", "Weight_HIS", "Weight_ILE", "Weight_LEU", 
    "Weight_LYS", "Weight_MET", "Weight_PHE","Weight_PRO","Weight_SER","Weight_THR","Weight_TRP","Weight_TYR","Weight_VAL"
]

#selected_features = [
    #"Ranking_Score", "NCD", "RCO", "Norm_ALA", "Norm_ARG", "Norm_ASN", "Norm_GLN", "Norm_LEU",
    #"Norm_TRP", "Norm_VAL", "Weight_ARG", "Weight_CYS", "Weight_GLN", "Weight_GLY", "Weight_HIS",
    #"Weight_ILE", "Weight_MET", "Weight_PHE", "Weight_SER", "Weight_THR", "Weight_TRP", "Weight_TYR", "Weight_VAL"
#]


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
        "Correct": 1 if actual_idx == pred_idx else 0,
        "Prob_HEME": probs[0], "Prob_FAD": probs[1], "Prob_Zn2": probs[2], "Prob_Cu1": probs[3], "Prob_Cofactorless": probs[4]
    })

# EXPORT & FINAL METRICS
pd.DataFrame(results_list).to_csv("LOOCV_Detailed_Results.csv", index=False)

loocv_acc = accuracy_score(y_true, y_pred_loocv)
print(f"\nFinal LOOCV Accuracy: {loocv_acc * 100:.2f}%")

# --- 5. CONFUSION MATRIX ---
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
