#!/bin/bash
#SBATCH -p dgx2q
#SBATCH  --job-name=HK-ENViSEC
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --time 03-00:00:00    # time (D-HH:MM:SS)
#SBATCH -o ./output/%j.out # STDOUT
# SBATCH -e ./output/%j.err # STDERR

ulimit -s 10240
echo "Job started at:" `date +"%Y-%m-%d %H:%M:%S"`

module purge
module load slurm/20.02.7
module load tensorflow2-py37-cuda10.2-gcc8/2.5.0  
module load python-mpi4py-3.0.3
module list

source venv/bin/activate
which python3
python3 --version

# srun python3 -m src.run
srun sh train.sh

echo "Job ended at:" `date +"%Y-%m-%d %H:%M:%S"`