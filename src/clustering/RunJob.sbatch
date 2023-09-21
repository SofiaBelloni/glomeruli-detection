#! /bin/bash

#SBATCH --job-name=auto_clustering
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s317626@studenti.polito.it
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=30G

module load intel/python/3/2019.4.088
module load nvidia/cudasdk/11.6
source /home/mla_group_02/visa/bin/activate
python /home/mla_group_02/visa/VISA/src/clustering/autoencoder.py
