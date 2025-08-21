#!/bin/bash
filename=$1
inputfile=$2
filename=$(echo $1 | sed 's/\..*$//')
now=$(date "+%s")
newfile="$filename$now.py"
srcfile="./src/$1"
srcdir="src"
echo "File = " $filename
echo "Date = " $now
echo "New file = " $newfile
echo "$filename$now"
mkdir -p ./results/
mkdir -p ./results/$filename/$filename$now
cp -r ./src/. ./results/$filename/$filename$now/
cp $inputfile ./results/$filename/$filename$now/$inputfile
python3 ./results/$filename/$filename$now/$1 $inputfile
