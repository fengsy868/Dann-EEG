#!/bin/sh

cd .

for ((i=650; i <= 6000; i++))
do
	th dann-learningcurve.lua -examplesize "$i"
	i=$(($i+50))
done