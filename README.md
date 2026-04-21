# An XGBoost Model for Identifying and Predicting Missing Cofactors in Predicted Protein Structures
A comprehensive framework developed for my MChem dissertation, which features the scripts for training data generation and analysis. This repository also includes scripts for feature engineering, XGBoost model construction and hyperparameter optimisation. 

**Author:** Angus Chan

**Affiliations:** Manchester Institute of Biotechnology & Department of Chemistry at The University of Manchester

Project Structure Summary: 
1. data_generation/
* PDB code extraction and obtaining their corresponding sequence
* Generate the training datasets and holo-protein structures.
  
2. data_analysis/
* PAE & Contact Maps: Generated for 400 proteins (200 apo and 200 holo).
* Structural alignment between apo- and holo-proteins by ColabAlign. Obtain RMSD and All-atom RMSD plots
* Obtain pLDDT vs index (i.e. all atom) plots from AlphaFold 3 data
* A filtered subset of the 80 proteins was used for plotting pLDDT vs RMSD correlation graphs

3. frustratometeR_analysis/
* Contains energetic frustration data retrieved via the frustratometeR package for the entire training dataset

4. ML_models_testing_stage/
* Guided vs unguided approach
* Changes in spherical pocket size
* Impact of specific feature additions on the model's performance

5. Final_XGBoost_Model/
* Contains the training and optimisation logs for 50 distinct XGBoost models.
* The best model (i.e. model with the highest macro-F1 score) was reported.
* Confusion matrix, feature importance charts, feature-feature correlation matrix, confidence calibration histogram, stratified 10-fold cross-validation loss and learning curves were plotted

<details>
<summary><b>Click here to see the detailed description of the key files</b></summary>

data_generation/:
1.  `pdb_extraction/filtered_pdb_list.txt` - Stores all PDB codes satisfying the filtered parameters
2.  `pdb_extraction/pdb_extraction.py` - Python script for extracting the PDB codes from the filtered_pdb_list.txt
3.  `sequence_extraction/sequence_extraction.py` - Python script for extracting the sequences of the chosen PDB codes.
4.  `sequence_extraction/sequences_{cofactor_type}_{total_number_of_pdb_codes_sampled}.txt` - Sequence of all the sampled PDB codes
5.  `alphafold/input/input_list.txt` - Example of how the input is written for generating structure predictions for apo-proteins
6.  `alphafold/input/input_list_cofactor.txt` - Example of how the input is written for generating structure predictions for holo-proteins. Includes the SMILES representation
7.  `alphafold/output` - Contains all the output of the training dataset and the holo-proteins. Each PDB code only contains the sample 0 output (i.e. the best structure prediction from AlphaFold 3)

data_analysis/:
1.  `colabalign/colabalign_scripts` - Scripts for running structural alignments
2.  `colabalign/structural_alignment_data/{pdb_code}` - Contains the reference and aligned coordinates, written in CSV files. Also has the all atom rmsd for the corresponding PDB code
3.  `colabalign/structural_alignment_data/alignment_summary.csv` - Summary of the PDB codes that have run a structural alignment (80 in total)
4.  `colabalign/structural_alignment_data/average_rmsd_summary.csv` - Average RMSD for each PDB code
5.  `colabalign/structural_alignment_data/maximum_rmsd_summary.csv` - Maximum RMSD for each code
6.  `structural_alignment_plots/python_scripts_for_processing_and_plotting_rmsd` - Python scripts for plotting the all atom RMSDs
7.  `structural_alignment_plots/plots_rmsd_per_index` - Plots for the all atom RMSD
8.  `structural_alignment_plots/plots_rmsd_per_index_cofactor_distance_included` - Plots for the all atom RMSD (inclusion of the cofactor coordinates in the plots)
9.  `plddt_vs_index_plots` - Plotted for 400 proteins (200 apo, 200 holo)
10.  `plddt_vs_rmsd` - Plotted for 80 proteins based on (3)

frustratometeR_analysis/:
1. `frustratometeR_analysis_for_af3_structures.R` - R script for running frustratometeR on the AlphaFold-obtained structures
2. `{pdb_code}/{pdb_code}_configurational.csv` - Raw energetic frustration data of each PDB code

ML_models_testing_stage/:
1. `guided_vs_unguided_approach`,`spherical_pocket_size_testing` and `feature_addition`: Each sub-folder contains independent training sets, execution scripts, and evaluation metrics (feature importance charts, confusion matrices and LOOCV results).

Final_XGBoost_Model/:
1. `bayesian_FINAL.py` - Python script for the Bayesian optimisation
2. `bayesian_search_log.csv` - Contains a log file of running the optimisations. It reports the macro-F1 (value) and accuracy scores. Tree-structured Parzen Estimator is used to binarily (yes/no) classify all input features in each instance of model training and optimisation
3. `bayesian_results.out` - Contains the details of all 50 instances of model training and optimisation. Also identifies the top 3 best and worst features for each model
4. `Best_Features_Table.csv` - Feature matrix used for final model
5. `Selected_Features_List.txt` - Key list of features for final model
6. `Best_Hyperparameters_Summary.csv` - List of hyperparameters used for final model
7. `best_trial_payload.pkl` - Stores all the data (features, hyperparameters, etc.) in binary serialisation format. No rounding in data. CSV files will round data (up to a 15 digit limit)
8. `macro_XG_boost_FINAL.py` - Python script for retraining of the best model. Contains the script for the plots excluding loss and learning curves
9. `ml_results.out` - Outputs the final classification report and the log loss analysis (Lists top 5 proteins with a high logarithmic loss for each class)
10. `loss_and_learning_curves.py` - Python script for plotting loss and learning curves


</details>
