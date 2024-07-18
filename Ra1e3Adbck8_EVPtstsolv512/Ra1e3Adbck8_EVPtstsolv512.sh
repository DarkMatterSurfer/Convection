#!/bin/bash
#SBATCH --account=b1094
#SBATCH --partition=ciera-std
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=${0##*/}

config=options.cfg
source png2mp4.sh
source options.cfg
if [ -f /projects/b1094/software/dotfiles/.bashrc ]; then
    . /projects/b1094/software/dotfiles/.bashrc
fi
dedalus3
mpirun -n $nprocs python3 $solver $config
# mpirun -n $nprocs python3 ~/Convection/plotscripts/plot_cheb.py $config
# png2mp4 ${name}/frames/ ${name}/movie.mp4 120
# echo ${name}/movie.mp4
# code ${name}/movie.mp4