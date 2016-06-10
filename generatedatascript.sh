#!/bin/sh

cd ~
cd Documents/TAO/DeepEGG/EEG/


# for ((j=100; j <= 10000; j = j*10)) 
# do
# 	for ((i=1; i <= 56; i++))
# 	do
# 		echo "$i"
# 		th eegdann.lua -hiddenLayerUnits "$i" -domainLambda "$j"
# 	done
# done

for ((i=1; i <= 56; i++))
do
	echo "$i"
	th eegdann.lua -hiddenLayerUnits "$i" -domainLambda 5000 -learningRate 1
done