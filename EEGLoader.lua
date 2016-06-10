dataset = torch.Tensor(9,12000,56)
for k = 1,9 do
   
   f = torch.DiskFile('EEG'..tostring(k)..'.dat' ,'r'):binary()
   
   timeslice = f:readInt()
   nSensors = f:readInt()
   print("Reading EEG"..tostring(k)..".dat")
   for i = 1,timeslice do
	  for j = 1,nSensors do
		 dataset[{k,i,j}] = f:readDouble()
	  end
   end
end
torch.save("EEG.torch",dataset)


