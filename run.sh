#!/bin/bash

source png2mp4.sh
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
cp $solver $name
# mpirun -n $nprocs python3 $solver $config
# mpirun -n $nprocs python3 ~/Convection/plotscripts/plotscalarmeasurement.py $config
mpirun -n $nprocs python3 ~/Convection/plotscripts/plot_cheb.py $config
png2mp4 ${name}/frames/ ${name}/movie.mp4 120
echo ${name}/movie.mp4
code ${name}/movie.mp4