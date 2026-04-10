#!/bin/sh
#
#SBATCH -p gpuL         ## Options: gpuA, gpuL
#SBATCH -G 1            ## Alphafold3 can only use 1 GPU
#SBATCH -n 1            ## One task
#SBATCH -c 12           ## 12 cpu cores
#SBATCH -t 4-00:00:00   ## wallclock limit Days-HH:MM:SS
#SBATCH --array=1-10
#SBATCH -J af3_batch_txt

module purge

module load apps/singularity/alphafold3/3.0.1


## run-specific input and output locations (scratch used as example, modify paths as needed)
INPUT=/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/input_batch_dir_2/input_list.txt
OUTPUT=/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2
TMP_JSON_DIR=${OUTPUT}/json_files

## Model weights - obtained individually and saved in your own user space 
WEIGHTS=/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job

## Databases we have pre-downloaded - you should not need to change this
DATABASES=/mnt/data-sets/alphafold3/3.0.1

mkdir -p "$TMP_JSON_DIR"
mkdir -p "$OUTPUT"

LINE_NUM=$SLURM_ARRAY_TASK_ID
TARGET_DATA=$(sed -n "${LINE_NUM}p" "$INPUT")

if [ -z "$TARGET_DATA" ]; then
	echo "No data found for task $SLURM_ARRAY_TASK_ID in $INPUT_LIST"
	exit 1
fi

# Parse the data: PDB_CODE, SEQUENCE, SMILES are separated by whitespace (TAB or SPACE)
PDB_CODE=$(echo "$TARGET_DATA" | awk '{print $1}')
SEQUENCE=$(echo "$TARGET_DATA" | awk '{print $2}')

# Define the output JSON path
JSON_PATH="${TMP_JSON_DIR}/${PDB_CODE}.json"

# --- Dynamically Generate the JSON File ---
echo "--- Generating JSON for ${PDB_CODE} ---"
# FILES=( "${INPUT}"/*.json )
# JSON="${FILES[$SLURM_ARRAY_TASK_ID-1]}"

cat <<EOF > "$JSON_PATH"
{
  "name": "${PDB_CODE}",
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "${SEQUENCE}"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
EOF

# --- Run AlphaFold3 ---
echo "Task $SLURM_ARRAY_TASK_ID"
echo "PDB Code: $PDB_CODE"
echo "JSON File: $JSON_PATH"
echo "Output Dir: $OUTPUT"

## execute alphafold3
run_alphafold \
        --json_path="${JSON_PATH}" \
        --model_dir=${WEIGHTS} \
        --db_dir=${DATABASES} \
        --output_dir=${OUTPUT} \
        --run_inference=true \
        --run_data_pipeline=true

echo "--- AlphaFold3 job finished for ${PDB_CODE} ---"
