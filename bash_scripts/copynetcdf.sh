!/bin/bash

# text file to read file names
filename=$1

while IFS=: read -r f1 f2
do
  printf 'Input: %s\n' "$f1"
  cp "$f1" "$f2"
done < "$1"
