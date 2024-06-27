#!/bin/bash
config=options.cfg
solver=rayleigh_benard_config.py
template=jobs.sh
source $config
if [ -d "$name" ]; then 
    echo "DIRECTORY $name ALREADY EXISTS!"
    echo "PRESS ENTER TO OVERIDE EXISTING SIMULATION SUITE"
    read -p " "
    rm -rf $name
fi
mkdir $name
cp $config $name
cp $solver $name
cp $template $name/$name.sh
cd $name 
sbatch $template > submit_message.txt
cat submit_message.txt