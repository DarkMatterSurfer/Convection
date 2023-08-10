#!/bin/bash

nprocs=$1
ra=$2
pr=$3
st=$4
lx=$5

name="$ra"

mpiexec -n $nprocs python3 rayleigh_benard.py --sn=$name --st=$st --ra=$ra --pr=$pr --lx=$lx
mpiexec -n $nprocs python3 Averaging_Flux.py 
