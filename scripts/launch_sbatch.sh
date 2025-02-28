#bin/bash -l

if [ ! -d ".git" ]; then
    echo "Error: Must be run from the root speechlmm directory of the Git repository!"
    exit 1
fi

job_name=""
sbatch_script=""
wdir=""
warmup=false
rebuild_dataset_cache=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j | --job_name)
      job_name="$2"
      shift 2
      ;;
    -s | --sbatch_script)
      sbatch_script="$2"
      shift 2
      ;;
    -w | --wdir)
      wdir="$2"
      shift 2
      ;;
    --warmup)
      warmup=true
      shift 1
      ;;
    --rebuild_dataset_cache)
      rebuild_dataset_cache=true
      shift 1
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$job_name" ] || [ -z "$sbatch_script" ] || [ -z "$wdir" ]; then
  echo "Usage: $0 [-j|--job_name] <job_name> [-s|--sbatch_script] <sbatch_script> [-w|--wdir] <wdir>"
  exit 1
fi

# Create remote working directory
if [ ! -d "$SCRATCH/wdirs" ]; then
    mkdir -p $SCRATCH/wdirs
fi

date_time=$(date '+%Y_%m_%d_%H_%M_%S')
wdir_copy="$SCRATCH/wdirs/$date_time-$job_name"

if [ -d "$wdir_copy" ]; then
    i=1
    while [ -d "$wdir_copy"_"$i" ]; do
        i=$((i + 1))
    done
    wdir_copy=$wdir_copy"_"$i
fi

mkdir $wdir_copy

# Define remote stderr and stdout files
stderr_file="$wdir_copy/$job_name.stderr"
stdout_file="$wdir_copy/$job_name.stdout"

cp -r $wdir/* $wdir_copy
rm -rf $wdir_copy/.git




if [ "$warmup" = true ]; then
  second_path=$(echo $sbatch_script | cut -d ' ' -f 2)
  config_path=$(grep -- '--data_config_path' "$second_path" | awk '{print $2}')
  job_name_warmup="warmup-$config_path"

  # we need to check if the warmup for that config is already running:
  # if it is, we don't submit a new one

  USER=$(whoami)
  RUNNING_JOB_ID=$(squeue -u $USER -n $job_name_warmup -h -o %A)
  if [ "$RUNNING_JOB_ID" ]; then
    echo "WARNING: the warmup job for the dataset $config_path is already running. The job will be submitted with that dependency."
    sbatch --job-name=$job_name \
      --output=$stdout_file \
      --error=$stderr_file \
      --chdir=$wdir_copy \
      --kill-on-invalid-dep=yes \
      --dependency=afterok:$RUNNING_JOB_ID \
      $sbatch_script

  else

    stderr_file_warmup="$wdir_copy/$job_name.warmup.stderr"
    stdout_file_warmup="$wdir_copy/$job_name.warmup.stdout"
    warmup_script="speechlmm/dataset/warmup_dataset.sbatch"


    export_=ALL,config_path="$config_path"
    if [ "$rebuild_dataset_cache" = true ]; then
      export_=$export_",rebuild_dataset_cache"
    fi
    job_id_warmup=$(sbatch --job-name=$job_name_warmup \
      --output=$stdout_file_warmup \
      --error=$stderr_file_warmup \
      --chdir=$wdir_copy \
      --export=$export_ \
      $warmup_script | awk '{print $4}')

    # Submit the second job with dependency on the first job
    sbatch --job-name=$job_name \
      --output=$stdout_file \
      --error=$stderr_file \
      --chdir=$wdir_copy \
      --kill-on-invalid-dep=yes \
      --dependency=afterok:$job_id_warmup \
      $sbatch_script
  fi
else
  if [ "$rebuild_dataset_cache" = true ]; then
    echo "Error: the --rebuild_dataset_cache flag can only be used with the --warmup flag."
    exit 1
  fi

  echo "WARNING: the job will be submitted without dataset warmup. It is recommended to use the --warmup flag to enable warmup if the dataset is not cached."
  sbatch --job-name=$job_name \
    --output=$stdout_file \
    --error=$stderr_file \
    --chdir=$wdir_copy \
    $sbatch_script
fi

exit 0
