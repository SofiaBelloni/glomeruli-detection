#! /bin/bash

#SBATCH --job-name=data_augmentation
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s317626@studenti.polito.it
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G

module load intel/python/3/2019.4.088
module load nvidia/cudasdk/11.6
source /home/mla_group_02/visa/bin/activate
python /home/mla_group_02/visa/VISA/src/data_augmentation/main.py