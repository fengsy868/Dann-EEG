--dann pour eeg sans domain classifier


require 'optim'
require 'nn'
require 'csvigo'

if not opt then
  print '==> processing options'
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Deep Learning - Telecom tutorial')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-learningRate', 1, 'learning rate at t=0')
  cmd:option('-batchSize', 50, 'mini-batch size (1 = pure stochastic)')
  cmd:option('-maxEpoch', 1000, 'maximum nb of epoch')
  cmd:option('-seed', 0, 'random seed')
  cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
  cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
  cmd:option('-hiddenLayerUnits', 20 , 'Number of hidden layer units')
  cmd:option('-tim', 1 , 'time of runs')
  cmd:text()
  opt = cmd:parse(arg or {})
end

gen = torch.Generator()
torch.manualSeed(gen, 0)
trainLogger = optim.Logger(paths.concat("results", opt.save))
trainLogger:setNames({'epoch','trainLoss','trainAccuracy','validLoss','validAccuracy'})

--------------------------------
--tensor to store the result
x = torch.Tensor()
--------------------------------
nTrainSource = 6000
nValidSource = 6000
EEGDataset1 = torch.load('Eyes-closed/Session1/EEG_Session1.torch')
nInputEEG = EEGDataset1:size(3)

sourceTrainSet = torch.Tensor(nTrainSource, nInputEEG)
sourceTrainSet:copy(EEGDataset1[1]:narrow(1,1,nTrainSource):float())
sourceTrainSetLabel = torch.Tensor(nTrainSource, nInputEEG)
sourceTrainSetLabel:copy(EEGDataset1[1]:narrow(1,1,nTrainSource):float())

sourceValidSet = torch.Tensor(nTrainSource,nInputEEG)
sourceValidSet:copy(EEGDataset1[1]:narrow(1,nTrainSource+1,nValidSource):float())
sourceValidSetLabel = torch.Tensor(nValidSource, nInputEEG)
sourceValidSetLabel:copy(EEGDataset1[1]:narrow(1,nTrainSource+1,nValidSource):float())

sourceInputs = torch.Tensor(opt.batchSize,sourceTrainSet:size(2)) 
sourceLabels = torch.Tensor(opt.batchSize,sourceTrainSet:size(2))                                        



-- -- concatente the source and target
-- trainSet = torch.Tensor(sourceTrainSet)
-- validSet = torch.Tensor(sourceValidSet)

-- concatente the source and target
validSet = torch.Tensor(sourceTrainSet)
trainSet = torch.Tensor(sourceValidSet)
-- Definition of the encoder
model = nn.Sequential()
model:add(nn.Linear(nInputEEG,opt.hiddenLayerUnits))
model:add(nn.Sigmoid())
model:add(nn.Linear(opt.hiddenLayerUnits,nInputEEG))
model:add(nn.Sigmoid())

	
-- Definition of the criterion
criterion = nn.MSECriterion()


params, gradParams = model:getParameters()

-- featExtractorParams = torch.ones(featExtractorParams:size())
-- labelPredictorParams = torch.ones(labelPredictorParams:size())
-- domainClassifierParams = torch.ones(domainClassifierParams:size())


inputs = torch.Tensor(opt.batchSize,trainSet:size(2))


-- Learning function
function train()

  local tick1 = sys.clock()

  -- It may help to shuffle the examples
  shuffle = torch.randperm(trainSet:size(1))

  for t = 1,trainSet:size(1),opt.batchSize do
   
    -- Define the minibatch
    for i = 1,opt.batchSize do
      inputs[i]:copy(trainSet[shuffle[t+i-1]])
      

      xlua.progress(t+i,sourceTrainSet:size(1))
    end

    -- Definition of the evaluation function (closure)
    local feval = function(x)
     
      if params ~=x then
			params:copy(x)
	  end
      gradParams:zero()

      local preds = model:forward(inputs)
      local Cost = criterion:forward(preds,inputs)
      local dfdo = criterion:backward(preds, inputs)
      model:backward(inputs, dfdo)

      return params, gradParams
    end
    optim.sgd(feval,params,opt)
  end
  print("tick" .. sys.clock()-tick1)
end

prevLoss = 10e12

tick2 = sys.clock()

for i = 1,opt.maxEpoch do
  model:evaluate()

  local trainPred = model:forward(trainSet)
  local trainLoss = criterion:forward(trainPred, trainSet) 
  

  print("EPOCH: " .. i)

  print(" + Train MSE ==> " .. trainLoss)

  local validPred = model:forward(validSet)
  local validLoss = criterion:forward(validPred, validSet) 

  print(" + Valid MSE ==> " .. validLoss)


  if opt.saveModel then
    if trainLoss < prevLoss then
      prevLoss = trainLoss
      torch.save("model.bin",model)
    else
      model = torch.load("model.bin")
    end
  end

  model:training()
  train()

  if i == opt.maxEpoch then
    file = csvigo.load("exp_25_09_16.csv")
    table.insert(file["ValidMSE"], validLoss)
    table.insert(file["TrainMSE"], trainLoss)
    table.insert(file["HiddenUnits"], opt.hiddenLayerUnits)
    table.insert(file["LearningRate"], opt.learningRate)
    table.insert(file["MaxEpoch"], opt.maxEpoch)
    table.insert(file["Times"], opt.tim)
    csvigo.save("exp_25_09_16.csv", file)
  end


  if prevLoss > trainLoss then
    prevLoss = trainLoss
    opt.learningRate = opt.learningRate * 1.2
  else
    prevLoss = trainLoss
    opt.learningRate = opt.learningRate / 2
  end

  -- -- opt.learningRate = 0.1 / i
  -- -- opt.learningRate = 0.1 / math.sqrt(i)
  -- table.insert(file["ValidMSE"], validLoss)
  -- table.insert(file["TrainMSE"], trainLoss)
  -- table.insert(file["HiddenUnits"], opt.hiddenLayerUnits)
  -- table.insert(file["LearningRate"], opt.learningRate)
  -- table.insert(file["Epoch"], i)
  -- table.insert(file["Exp"], '3')
  -- csvigo.save("exp_25_07_16_3.csv", file)
end
--exp1: lr = 1: 
--exp2: adaptive: 1573s = 26mins
--exp3: lr = 10:  1637s = 27mins

print('Time Total Used: ' .. sys.clock()-tick2 .. 's')

-- function mse( preds, targets )
--   count = 0
--   for i=1,preds:size(1) do
--     count = count + (preds[i]-targets[i])^2--math.sqrt((preds[i]-targets[i])^2)
--   end
--   -- print("Preds size ------->" .. preds:size(1))
--   return count/preds:size(1)
-- end

-- trainPred = model:forward(trainSet)

-- validPred = model:forward(validSet)

-- file = csvigo.load("exp_25_07_16_4.csv")
-- for i = 1, trainSet:size(1)+validSet:size(1) do
--   if i <= trainSet:size(1) then
--     table.insert(file["MSE"], mse(trainPred[i], trainSet[i]))
--   else
--     table.insert(file["MSE"], mse(validPred[i-trainSet:size(1)], validSet[i-trainSet:size(1)]))
--   end
--   table.insert(file["Index"], i)
--   table.insert(file["HiddenUnits"], opt.hiddenLayerUnits)
  
-- end
-- csvigo.save("exp_25_07_16_4.csv", file)

