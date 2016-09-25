#!/bin/sh

cd .

s=1
# for ((j=1; j <= 1000; j = j*10)) 
# do
# 	for ((i=1; i <= 56; i++))
# 	do
# 		for ((k=1; k <= 3; k++))
# 		do
# 			s=$(($s+1))
# 			echo "$s"
# 			th eegdann.lua -hiddenLayerUnits "$i" -domainLambda "$j" -tim "$k" -seed "$s"
# 		done
# 	done
# done


# for ((i=1; i <= 56; i++))
# do
# 	for ((k=1; k <= 10; k++))
# 	do
# 		s=$(($s+1))
# 		echo "$s"
# 		th eegdann.lua -hiddenLayerUnits "$i" -domainLambda 0.1 -tim "$k" -seed "$s"
# 	done
# done

# for (( j = 1; j <= 56; j++ )); do
# 	for ((i=1; i <= 5; i++))
# 	do
# 		echo "$j"
# 		echo "$i"
# 		th eegdann.lua -hiddenLayerUnits "$j" -tim "$i"
# 	done
# done
for ((i=1; i <= 56; i++))
do
	echo "$i"
	th dannsansdomain.lua -hiddenLayerUnits "$i" -maxEpoch 100 -tim 1
done