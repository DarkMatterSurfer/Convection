#!/bin/bash
config=options.cfg
template=jobs.sh
solverEVP=EVP_methods_CHEBBED.py
source $config
if [ -d "$name" ]; then 
    echo "DIRECTORY $name ALREADY EXISTS!"
    echo "PRESS ENTER TO OVERIDE EXISTING SIMULATION SUITE"
    read -p " "
    rm -rf $name
fi
mkdir $name

cp $config $name
cp $solverEVP $name
cp $solver $name
cp $template $name/$name.sh
cd $name 
sbatch $name.sh > submit_message.txt
cat submit_message.txt