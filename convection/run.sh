#!/bin/bash

source png2mp4.sh
# rm -rf $name_frames ; rm -rf snapshots

nprocs=$1
ra=$2
pr=$3
st=$4
lx=$5

name="$ra"
rm -rf ${name}_frames ; rm -rf snapshots
mpiexec -n $nprocs python3 rayleigh_benard.py --sn=$name --st=$st --ra=$ra --pr=$pr --lx=$lx
mpiexec -n $nprocs python3 plot_snapshots.py $name/*.h5 --output=${name}_frames --lx=$lx
png2mp4 ${name}_frames/ ${name}.mp4 120
code ${name}.mp4
