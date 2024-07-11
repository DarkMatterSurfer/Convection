#!/bin/bash
config=options.cfg
template=jobs.sh
EVPsolver=EVP_methods.py
source $config
if [ -d "$name" ]; then 
    echo "DIRECTORY $name ALREADY EXISTS!"
    echo "PRESS ENTER TO OVERIDE EXISTING SIMULATION SUITE"
    read -p " "
    rm -rf $name
fi
mkdir $name
cp $EVPsolver $name
cp $config $name
cp $solver $name
cp $template $name/$name.sh
cd $name 
sbatch $name.sh > submit_message.txt
cat submit_message.txt