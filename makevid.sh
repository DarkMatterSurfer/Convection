#!/bin/bash

source png2mp4.sh
png2mp4 ${1}/frames/ ${1}/movie.mp4 120
echo "${1}/movie.mp4"
