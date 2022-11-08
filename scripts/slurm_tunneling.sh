#!/bin/bash
#SBATCH -p milanq
#SBATCH  --job-name=HK-ENViSEC
#SBATCH -n 1
#SBATCH --time=01-02:00:00
#SBATCH -o ./output/%j.out # STDOUT

case $(hostname) in
    talapas-ln*) echo 'this script should be run with sbatch'; exit 1;;
esac

#Load necessary modules
ulimit -s 10240

module purge
module load slurm/20.02.7
module load python-3.7.4
module list

#  source venv/bin/activate
#  srun python3 src/run_shallow.py

#Go to the folder you wanna run jupyter in
cd $HOME/ENViSEC/sec-iot
source venv/bin/activate

#Pick a random or predefined port
port=$(shuf -i 6000-9999 -n1)
hostport=$(shuf -i8000-64000 -n1)

node=$(hostname -s)
user=$(whoami)

echo "Starting Jupyter on $node"
echo "Port: $port"
echo "Hostport: $hostport"
echo "User: $user"

# port=8769
#Forward the picked port to the ex3 on the same port. Here log-x is set to be the ex3 login node.
ssh -N -L localhost:${port}:${node}:${hostport} ${user}@${loginnode}

#Start the notebook
jupyter-notebook --no-browser --port=${hostport} --port-retries=0 --ip='*' --NotebookApp.shutdown_no_activity_timeout=10000