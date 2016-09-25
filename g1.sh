#!/bin/sh

cd .

s=1
# for ((j=1; j <= 10000; j = j*10)) 
# do
# 	for ((i=1; i <= 56; i++))
# 	do
# 		for ((k=1; k <= 10; k++))
# 		do
# 			s=$(($s+1))
# 			echo "$s"
# 			th eegdann.lua -hiddenLayerUnits "$i" -domainLambda "$j" -tim "$k" -seed "$s"
# 		done
# 	done
# done


for ((i=1; i <= 56; i++))
do
	for ((k=1; k <= 10; k++))
	do
		s=$(($s+1))
		echo "$s"
		th e1.lua -hiddenLayerUnits "$i" -domainLambda 0.01 -tim "$k" -seed "$s"
	done
done
# for ((i=1; i <= 56; i++))
# do
# 	echo "$i"
# 	th eegdann.lua -hiddenLayerUnits "$i" -domainLambda 100 -learningRate 1
# done

# for ((j=1; j <= 10; j = j+1)) 
# do
# 	for ((i=1; i <= 56; i++))
# 	do
# 		echo "$j"
# 		th eegdann.lua -hiddenLayerUnits "$i" -domainLambda 100 -tim "$j"
# 	done
# done