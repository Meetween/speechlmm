module load Miniconda3/23.3.1-0
module load CUDA/12.1.1

eval "$(conda shell.bash hook)"

# if you want to send a message to a slurm-bot when the job starts [SLURM-BOT: https://github.com/St3p99/slurm-bot]
# conda activate slurm-bot [conda env with slurm-bot dependencies]
# if [ -z "$SBATCH_JOB_NAME" ]; then
#     SLURM_BOT_TOKEN=<YOUR_SLURM_BOT_TOKEN> SLURM_BOT_CHAT_ID=<YOUR_CHAT_ID_W_BOT> python $SCRATCH/slurm-bot/job_started.py --job-id $SLURM_JOB_ID
# else
#     SLURM_BOT_TOKEN=<YOUR_SLURM_BOT_TOKEN> SLURM_BOT_CHAT_ID=<YOUR_CHAT_ID_W_BOT> python $SCRATCH/slurm-bot/job_started.py --job-id $SLURM_JOB_ID --job-name $SBATCH_JOB_NAME
# fi

conda deactivate

conda activate speechlmm

# YOUR SCRIPT HERE > out.stdout 2> err.stderr

# If your script fails, the SLURM Job will be still running, so you can check the error and fix it
/bin/bash -l
