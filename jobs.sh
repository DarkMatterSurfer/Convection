#!/bin/bash
#SBATCH --account=b1094
#SBATCH --partition=ciera-std
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=IH1boundcondtestrun
cd $HOME/Convection
if [ -f /projects/b1094/software/dotfiles/.bashrc ]; then
    . /projects/b1094/software/dotfiles/.bashrc
fi
dedalus3
bash run.sh
echo "succeeded"