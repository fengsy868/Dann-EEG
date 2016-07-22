import csv
import numpy

listoflists = [[],[],[],[],[]]
csv_filepath = 'exp_22_07_16.csv'
with open(csv_filepath, "rb") as f:
	reader = csv.reader(f)
	header = reader.next()
	for row in reader:
		listoflists[int(row[8])-1].append(row) #here row[i] indicate the index of Times

for i in range(len(listoflists)):
	for j in range(len(listoflists[i])):
		listoflists[i][j] = map(float, listoflists[i][j])
	

print listoflists[1][1]
a = numpy.array(listoflists)

result = numpy.mean(a, axis=0)
result = result.tolist()

csv_targetpath = 'exp_22_07_16_average.csv'

with open(csv_targetpath, "wb") as file:
	wr = csv.writer(file)
	head = 'TargetDomLossValid,ValidMSE,SourceDomLossValid,Domainloss,LearningRate,HiddenUnits,DomainLambda,TrainMSE,Times,MaxEpoch\n'
	file.write(head)
	# wr.writerows(['Spam', 'Lovely Spam', 'Wonderful Spam'])
	wr.writerows(result)