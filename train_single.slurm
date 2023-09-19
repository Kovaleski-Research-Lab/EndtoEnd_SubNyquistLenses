#! /bin/bash
#SBATCH --job-name=TEST_TRAIN
#SBATCH --nodes=1
#SBATCH --partition=gpu32,gpu16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=10G
#SBATCH --output=/cgi/home/lindsaymb/workspace/documents/code/end-to-end/logs/%x-%j.out

wavelengths=($(seq 193.75e-9 77.5e-9 2906.25e-9))

srun enroot start --rw \
    -m /cgi/data/erdc_xai:/develop/data \
    -m /cgi/data/erdc_xai/results:/develop/results \
    -m /cgi/home/lindsaymb/workspace/documents/code:/develop/code \
    -m /cgi/data/erdc_xai/pretrained_models:/develop/pretrained_models \
    -m /cgi/data/erdc_xai/models:/develop/models \
    lrn_env \
    python3 /develop/code/end-to-end/experiment.py -config /develop/code/end-to-end/config.yaml -job_id $SLURM_JOB_ID -wavelength ${wavelengths[-1]}



