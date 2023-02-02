#!/bin/bash

filename=$1

module load cdo

while IFS=: read -r f1 f2
do
  printf 'Input: %s\n' "$f1"
  cdo mergetime "$f1" "$f2"
done < "$1"
