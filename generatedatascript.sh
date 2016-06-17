#!/bin/sh

cd .

for ((j=1; j <= 20; j = j+1)) 
do
	for ((i=1; i <= 56; i++))
	do
		echo "$j"
		th eegdann.lua -hiddenLayerUnits "$i" -domainLambda "$j"
	done
done

# for ((i=1; i <= 56; i++))
# do
# 	echo "$i"
# 	th eegdann.lua -hiddenLayerUnits "$i" -domainLambda 5000 -learningRate 1
# done
