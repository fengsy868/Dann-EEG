#!/bin/sh

cd .

s=1

for ((k=1; k <= 10; k++))
do
	s=$(($s+10))
	echo "$s"
done
