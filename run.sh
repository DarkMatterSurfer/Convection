#!/bin/bash

source png2mp4.sh
# rm -rf $name_frames ; rm -rf snapshots

# nprocs=$1
# ra=$2
# pr=$3
# st=$4
# lx=$5
# state=$6
# chckpoints_bckup/checkpoints_s[saturatted file].h5
first_string=$ra
second_string="p"


config=options.cfg
source $config
# if [ -d "$name" ]; then 
#     echo "DIRECTORY $name ALREADY EXISTS!"
#     echo "PRESS ENTER TO OVERIDE EXISTING SIMULATION SUITE"
#     read -p " "
#     rm -rf $name
# fi
mkdir $name
cp $config $name
# # mpiexec -n $nprocs python3 rayleigh_benard.py --sn=$name --st=$st --ra=$ra --pr=$pr --lx=$lx --state=$state
mpiexec -n $nprocs python3 rayleigh_benard_config.py $config
# mpiexec -n 1 python3 plotfluxbetter.py $config
mpiexec -n $nprocs python3 plot_snapshots.py $config
png2mp4 ${name}/frames/ ${name}/movie.mp4 120
code ${name}/movie.mp4
# rm -rf ${name}_frames ; rm -rf snapshots
