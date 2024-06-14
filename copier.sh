mkdirname=$1
questname=$2

mkdir $mkdirname ; scp -r quest:$questname $mkdirname/movie.mp4 120