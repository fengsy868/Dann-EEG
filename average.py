import csv
import numpy

listoflists = [[],[],[],[],[],[],[],[],[],[]]
csv_filepath = 'final_test.csv'
with open(csv_filepath, "rb") as f:
	reader = csv.reader(f)
	header = reader.next()
	for row in reader:
		listoflists[int(row[8])-1].append(row)

for i in range(len(listoflists)):
	for j in range(len(listoflists[i])):
		listoflists[i][j] = map(float, listoflists[i][j])
	

print listoflists[1][1]
a = numpy.array(listoflists)

result = numpy.mean(a, axis=0)
result = result.tolist()

csv_targetpath = 'average_value.csv'

with open(csv_targetpath, "wb") as file:
	wr = csv.writer(file)
	file.write('TargetDomLossValid,ValidMSE,SourceDomLossValid,Domainloss,LearningRate,HiddenUnits,DomainLambda,MaxEpoch,Times\n')
	# wr.writerows(['Spam', 'Lovely Spam', 'Wonderful Spam'])
	wr.writerows(result)