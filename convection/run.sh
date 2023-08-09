#!/bin/bash


# PNG encoding with ffmpeg
# Options:
#  -y         Overwrite output
#  -f image2pipe    Input format
#  -vcodec png     Input codec
#  -r $3        Frame rate
#  -i -        Input files from cat command
#  -f mp4       Output format
#  -vcodec libx254   Output codec
#  -pix_fmt yuv420p  Output pixel format
#  -preset slower   Prefer slower encoding / better results
#  -crf 20       Constant rate factor (lower for better quality)
#  -vf "scale..."   Round to even size
#  $2         Output file
rm -rf frames ; rm -rf snapshots

function png2mp4(){
  cat $1* | ffmpeg \
    -y \
    -f image2pipe \
    -vcodec png \
    -r $3 \
    -i - \
    -f mp4 \
    -vcodec libx264 \
    -pix_fmt yuv420p \
    -preset slower \
    -crf 20 \
    -vf "scale=trunc(in_w/2)*2:trunc(in_h/2)*2" \
    $2
}
function png2mp4_big(){ 
  echo $1* | xargs cat | ffmpeg \
  -y \
    -f image2pipe \
    -vcodec png \
    -r $3 \
    -i - \
    -f mp4 \
    -vcodec libx264 \
    -pix_fmt yuv420p \
    -preset slower \
    -crf 20 \
    -vf "scale=trunc(in_w/2)*2:trunc(in_h/2)*2" \
    $2
}

nprocs=8
mpiexec -n $nprocs python3 rayleigh_benard.py
mpiexec -n $nprocs python3 plot_snapshots.py snapshots/*.h5
png2mp4 frames/ $1 $2
code $1
