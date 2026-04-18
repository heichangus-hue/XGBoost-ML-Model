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

<details>
<summary><b>Click here to see the description of the key files</b></summary>

data_generation/:
1.  `pdb_extraction/filtered_pdb_list.txt` - Stores all PDB codes satisfying the filtered parameters
2.  `pdb_extraction/pdb_extraction.py` - Python script for extracting the PDB codes from the filtered_pdb_list.txt
3.  `sequence_extraction/sequence_extraction.py` - Python script for extracting the sequences of the chosen PDB codes.
4.  `sequence_extraction/sequences_{cofactor_type}_{total_number_of_pdb_codes_sampled}.txt` - Sequence of all the sampled PDB codes
5.  `alphafold/input/input_list.txt` - Example of how the input is written for generating structure predictions for apo-proteins
6.  `alphafold/input/input_list_cofactor.txt` - Example of how the input is written for generating structure predictions for holo-proteins. Includes the SMILES representation
7.  `alphafold/output` - Contains all the output of the training dataset and the holo-proteins. Each PDB code only contains the sample 0 output (i.e. the best structure prediction from AlphaFold 3)

Final_XGBoost_Model/:
1. `bayesian_FINAL.py` - Python script for the Bayesian optimisation
2. `bayesian_search_log.csv` - Contains a log file of running the optimisations. It reports the macro-F1 (value) and accuracy scores. Tree-structured Parzen Estimator is used to binarily (yes/no) classify all input features in each instance of model training and optimisation.
3. `bayesian_results.out` - Contains the details of all 50 instances of model training and optimisation. Also identifies the top 3 best and worst features for each model.
4. `Best_Features_Table.csv` - Key features used after optimisation in the final model
5. `Selected_Features_List.txt` - Key list of features used in the final model
6. `Best_Hyperparameters_Summary.csv` - Final tuning parameters in the final model
7. `best_trial_payload.pkl` - Stores all the data (features, hyperparameters, etc.) in binary serialisation format. No rounding in data. csv files will round data (up to a 15 digit limit)
8. `macro_XG_boost_FINAL.py` - Python script for retraining of the best model. Contains the script for the plots excluding loss and learning curves
9. `ml_results.out` - Outputs the final classification report and the top 5 high logarithmic loss for each class (scroll to the bottom)
10. `loss_and_learning_cruves.py` - Python script for plotting loss and learning curves



</details>
