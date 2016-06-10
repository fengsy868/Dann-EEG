local matio = require 'matio'

dataset = torch.Tensor(9,12000,56)
for k = 1,9 do
	local temp = matio.load('S'..tostring(k)..'_SR200_CE_EEG.ch.mat')
	local tensor = temp.EEG.samp
	print('Reading '..tostring(k)..' file')
	max = torch.max(tensor)
	min = torch.min(tensor)

	tensor = (tensor - min) / ( max - min)
	for i = 1, 12000 do
		for j = 1, 56 do
			dataset[k][i][j] = tensor[i][j]
		end
	end 
end

torch.save('EEGMat.torch', dataset)