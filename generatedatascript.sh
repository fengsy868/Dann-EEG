#!/bin/sh

cd .

for ((j=1; j <= 10000; j = j*10)) 
do
	for ((i=1; i <= 56; i++))
	do
		echo "$((j/10))"
		th eegdann.lua -hiddenLayerUnits "$i" -domainLambda "$j"
	done
done

# for ((i=1; i <= 56; i++))
# do
# 	echo "$i"
# 	th eegdann.lua -hiddenLayerUnits "$i" -domainLambda 5000 -learningRate 1
# done
