# XGBoost Model for Identifying and Predicting Missing Cofactors in Predicted Protein Structures
A comprehensive pipeline developed for my MChem dissertation, which features the scripts for training data generation and analysis. This repository also includes scripts for feature engineering, XGBoost model construct and hyperparameter optimisation. 

**Author:** Angus Chan

**Affiliations:** Manchester Institute of Biotechnology & Department of Chemistry at The University of Manchester

Project Structure: 
1. data_generation - Generation of training dataset and holo-proteins. Training dataset is fed to the XGBoost models for learning. The apo- and holo-proteins are used for data analysis.
2. data_analysis - Involves the generation of contact and PAE maps for 160 proteins. Used ColabAlign to align apo- and holo-proteins to compute the RMSD and all atom RMSD plots. pLDDT plots were produced for 160 proteins. 80 of the 160 proteins were used to plot the plddt vs RMSD graphs.
3. frustratometeR_analysis - Data retrieved from running the frustratometeR for the entire training dataset
4. ML_models_testing_stage - Invovles testing how the guided vs unguided aproach, spherical pocket size and feature additions would affect the model's performance
5. Final_XGBoost_Model - Contains the training and optimisation of 50 XGBoost models. The best model was reported in the report, and the confusion matrix, feature importance chart, feature-feature correlation matrix, confidence calibration histogram, stratified 10-fold cross-validation loss curve and learning curve were obtained. 

Final_XGBoost_Model - 


    * Best_Features_Table.csv - 
    * Best_Hyperparameters_Summary.csv -
    * Boxplot_Uncertainty.png - 
  Confidence Histogram.png
  Feature_Feature_Correlation_Matrix.png - 
  LOOCV_Detailed_Results.csv - 
  LOOCV_confusion_matrix.png - 
  LOOCV_feature_importance_with_control.png - 
  Learning_Curve.png - 
  Loss_vs_Estimators_10_Fold_Cross_Validation.png - 
  Selected_Features_List.txt - 
  bayesian_FINAL.py - 
  bayesian_results.out - 
  bayesian_search_log.csv - 
  best_features.json - 
  best_trial_payload.pkl - 
  loss_and_learning_curves.py - 
  loss_and_learning_results.out - 
  macro_XG_boost_FINAL.py - 
  ml_results.out - 

<details>
<summary><b>Click here to see the full list of 160 files and descriptions</b></summary>

* `Best_Features_Table.csv` - Summary of feature importance.
* `Best_Hyperparameters_Summary.csv` - Final tuning parameters.
* (And so on...)

</details>
