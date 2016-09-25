import csv
import numpy

listoflists = [[],[],[]] # size of times here
csv_filepath = 'exp_23_09_16.csv'
with open(csv_filepath, "rb") as f:
	reader = csv.reader(f)
	header = reader.next()
	for row in reader:
		listoflists[int(row[4])-1].append(row) #here row[i] indicate the index of Times

for i in range(len(listoflists)):
	for j in range(len(listoflists[i])):
		listoflists[i][j] = map(float, listoflists[i][j])
	

print listoflists[1][1]
a = numpy.array(listoflists)

result = numpy.mean(a, axis=0)
result = result.tolist()

csv_targetpath = 'exp_23_09_16_average.csv'

with open(csv_targetpath, "wb") as file:
	wr = csv.writer(file)
	head = 'ValidMSE,DomainlossTrain,DomainlossValid,DomainLambda,Times,TrainMSE,HiddenUnits\n'
	file.write(head)
	# wr.writerows(['Spam', 'Lovely Spam', 'Wonderful Spam'])
	wr.writerows(result)