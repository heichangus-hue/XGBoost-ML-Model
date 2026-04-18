# XGBoost Model for Identifying and Predicting Missing Cofactors in Predicted Protein Structures
A comprehensive pipeline developed for my MChem dissertation, which features the scripts for training data generation and analysis. This repository also includes scripts for feature engineering, XGBoost model construct and hyperparameter optimisation. 

**Author:** Angus Chan

**Affiliations:** Manchester Institute of Biotechnology & Department of Chemistry at The University of Manchester

Project Structure: 
1. data_generation/
* PDB code extraction and obtaining their corresponding sequence
* Generate the training datasets and holo-protein structures.
  
2. data_analysis/
* PAE & Contact Maps: Generated for 200 proteins (apo and holo).
* Structural alignment between apo- and holo-proteins by ColabAlign. Obtain RMSD and All-atom RMSD plots
* Obtain pLDDT vs index (i.e. all atom) plots from AlphaFold 3 data
* A filtered subset of the 80 proteins was used for plotting pLDDT vs RMSD correlation graphs


3. frustratometeR_analysis/
* Contains energetic frustration data retrieved via the frustratometeR package for the entire training dataset

4. ML_models_testing_stage/
* Guided vs unguided Approach
* Changes in spherical pocket size
* Impact of specific feature additions on the model's performance

5. Final_XGBoost_Model/
* Contains the training and optimisation logs for 50 distinct XGBoost models.
* The best model (i.e. model with the highest macro-F1 score) was reported.
* Confusion matrix, feature importance charts, feature-feature correlation matrix, confidence calibration histogram, stratified 10-fold cross-validation loss and learning curves were plotted


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
<summary><b>Click here to see the description of the key files</b></summary>

Final_XGBoost_Model:
* `bayesian_FINAL.py` - Python script for the Bayesian optimisation
* `bayesian_search_log.csv` - Contains a log file of running the optimisations. It reports the macro-F1 (value) and accuracy scores. Tree-structured Parzen Estimator is used to binarily (yes/no) classify all input features in each instance of model training and optimisation. 
* `bayesian_results.out` - Contains the details of all 50 instances of model training and optimisation. Also identifies the top 3 best and worst features for each model. 
* `Best_Features_Table.csv` - Key features used after optimisation in the final model
* `Selected_Features_List.txt` - Key list of features used in the final model
* `Best_Hyperparameters_Summary.csv` - Final tuning parameters in the final model
* `macro_XG_boost_FINAL.py` - Python script for retraining of the best model. Contains the script for the plots excluding loss and learning curves 
* `ml_results.out` - Outputs the final classification report and the top 5 high logarithmic loss for each class (scroll to the bottom)
* `loss_and_learning_cruves.py` - Python script for plotting loss and learning curves

</details>
