require 'optim'
require 'nn'
---------------------------
-- Definition of DANN
--
--
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Deep Learning - Telecom tutorial')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-learningRate', 0.1, 'learning rate at t=0')
   cmd:option('-domainLambda', 0.1, 'regularization term for transfer learning')
   cmd:option('-batchSize', 500, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-maxEpoch', 100, 'maximum nb of epoch')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:text()
   opt = cmd:parse(arg or {})
end
torch.manualSeed(opt.seed)
trainLogger = optim.Logger(paths.concat("results", opt.save))
trainLogger:setNames({'epoch','trainLoss','trainAccuracy','validLoss','validAccuracy'})

--------------------------------------
-- Loading and normalizing the dataset

-- classes
classes = {'0','1','2','3','4','5','6','7','8','9'}

-- This matrix records the current confusion across classes
trainConfusion = optim.ConfusionMatrix(classes)
validConfusion = optim.ConfusionMatrix(classes)

-- Load the source dataset (MNIST)
local nTrainMnist = 50000
local nValidMnist = 10000
local mnist = require 'mnist'
local mnistDataset = mnist.traindataset()
local nInputMnist = mnistDataset.data:size(2) * mnistDataset.data:size(3)

sourceTrainSet = torch.Tensor(nTrainMnist,mnistDataset.data:size(2),mnistDataset.data:size(3))
sourceTrainSet:copy(mnistDataset.data:narrow(1,1,nTrainMnist):float():div(255.))
sourceTrainSetLabel = torch.Tensor(nTrainMnist)
sourceTrainSetLabel:copy(mnistDataset.label:narrow(1,1,nTrainMnist))
sourceTrainSetLabel:add(1)

sourceValidSet = torch.Tensor(nValidMnist,mnistDataset.data:size(2),mnistDataset.data:size(3))
sourceValidSet:copy(mnistDataset.data:narrow(1,nTrainMnist+1,nValidMnist):float():div(255.))
sourceValidSetLabel = torch.Tensor(nValidMnist)
sourceValidSetLabel:copy(mnistDataset.label:narrow(1,nTrainMnist+1,nValidMnist))
sourceValidSetLabel:add(1)

sourceInputs = torch.Tensor(opt.batchSize,sourceTrainSet:size(2),sourceTrainSet:size(3))
sourceLabels = torch.Tensor(opt.batchSize)

-- Load the target dataset (SVHN)
local nTrainSvhn = 50000
local nValidSvhn = 23257
local svhnDataset = torch.load('housenumbers/train_28x28.t7')
local nInputSvhn = svhnDataset.X:size(2) * svhnDataset.X:size(3)

targetTrainSet = torch.Tensor(nTrainSvhn,svhnDataset.X:size(2),svhnDataset.X:size(3))
targetTrainSet:copy(svhnDataset.X:narrow(1,1,nTrainSvhn):float())
targetTrainSetLabel = torch.Tensor(nTrainSvhn)
targetTrainSetLabel:copy(svhnDataset.y:narrow(2,1,nTrainSvhn))

targetValidSet = torch.Tensor(nValidSvhn,svhnDataset.X:size(2),svhnDataset.X:size(3))
targetValidSet:copy(svhnDataset.X:narrow(1,nTrainSvhn+1,nValidSvhn):float())
targetValidSetLabel = torch.Tensor(nValidSvhn)
targetValidSetLabel:copy(svhnDataset.y:narrow(2,nTrainSvhn+1,nValidSvhn))

targetInputs = torch.Tensor(opt.batchSize,targetTrainSet:size(2),targetTrainSet:size(3))
targetLabels = torch.Tensor(opt.batchSize)   

targetInputs = torch.Tensor(opt.batchSize,targetTrainSet:size(2),targetTrainSet:size(3))
targetLabels = torch.Tensor(opt.batchSize)

-- Definition of the feature extractor
featExtractor = nn.Sequential()
featExtractor:add(nn.Reshape(nInputMnist))
featExtractor:add(nn.Linear(nInputMnist,100))
featExtractor:add(nn.ReLU())
featExtractor:add(nn.Linear(100,100))

-- Definition of the label predictor
labelPredictor = nn.Sequential()
labelPredictor:add(nn.Linear(100,10))
labelPredictor:add(nn.LogSoftMax())

-- Definition of the domain classifier
domainClassifier = nn.Sequential()
domainClassifier:add(nn.GradientReversal())
domainClassifier:add(nn.Linear(100,1))
domainClassifier:add(nn.Sigmoid())

labelPredictorCriterion = nn.ClassNLLCriterion()
domainClassifierCriterion = nn.BCECriterion()

-- Retrieve the pointers to the parameters and gradParameters from the model for latter use
featExtractorParams,featExtractorGradParams = featExtractor:getParameters()
labelPredictorParams,labelPredictorGradParams = labelPredictor:getParameters()
domainClassifierParams,domainClassifierGradParams = domainClassifier:getParameters()
params = torch.Tensor(featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1))
params:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
params:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
params:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)
gradParams = torch.Tensor(featExtractorParams:size(1)+labelPredictorParams:size(1)+domainClassifierParams:size(1))
--

print("feat " .. tostring(featExtractorParams:size()))
print("label " .. tostring(labelPredictorParams:size()))
print("domain " .. tostring(domainClassifierParams:size()))

-- Learning function
function train()

   local tick1 = sys.clock()
   
   -- It may help to shuffle the examples
   shuffle = torch.randperm(sourceTrainSet:size(1))
   
   for t = 1,sourceTrainSet:size(1),opt.batchSize do
	  
	  xlua.progress(t,sourceTrainSet:size(1))
	  
	  -- Define the minibatch
	  for i = 1,opt.batchSize do
		 sourceInputs[i]:copy(sourceTrainSet[shuffle[t+i-1]])
		 sourceLabels[i] = sourceTrainSetLabel[shuffle[t+i-1]]
		 targetInputs[i]:copy(targetTrainSet[shuffle[t+i-1]])
		 targetLabels[i] = targetTrainSetLabel[shuffle[t+i-1]]
	  end

	  -- Definition of the evaluation function (closure)
	  local feval = function(x)
		 
		 --featExtractorParams:copy(x)
		 featExtractorParams:copy(x:narrow(1,1,featExtractorParams:size(1)))
		 labelPredictorParams:copy(x:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)))
		 domainClassifierParams:copy(x:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)))
		 
		 featExtractorGradParams:zero()
		 labelPredictorGradParams:zero()
		 domainClassifierGradParams:zero()

		 local feats = featExtractor:forward(sourceInputs)
		 local preds = labelPredictor:forward(feats)
		 local labelCost = labelPredictorCriterion:forward(preds,sourceLabels)

		 --print("Label cost ".. tostring(labelCost))

		 local labelDfdo = labelPredictorCriterion:backward(preds, sourceLabels)
		 local gradLabelPredictor = labelPredictor:backward(feats, labelDfdo)
		 featExtractor:backward(sourceInputs, gradLabelPredictor)

		 local domPreds = domainClassifier:forward(feats)
		 local domCost = domainClassifierCriterion:forward(domPreds,torch.Tensor(domPreds:size(1),1):fill(1)) -- TODO: ugly, replace with two unique allocations
		 local domDfdo = domainClassifierCriterion:backward(domPreds,torch.Tensor(domPreds:size(1),1):fill(1))
		 local gradDomainClassifier = domainClassifier:backward(feats,domDfdo,opt.domainLambda) --TODO: verify
		 featExtractor:backward(sourceInputs, gradDomainClassifier,opt.domainLambda)

		 --- Target propagation
		 local targetFeats = featExtractor:forward(targetInputs)
		 local targetDomPreds = domainClassifier:forward(targetFeats)
		 local targetDomCost = domainClassifierCriterion:forward(targetDomPreds,torch.Tensor(targetDomPreds:size(1),1):fill(0)) -- TODO: ugly, replace with two unique allocations
		 local targetDomDfdo = domainClassifierCriterion:backward(targetDomPreds,torch.Tensor(targetDomPreds:size(1),1):fill(0))
		 local targetGradDomainClassifier = domainClassifier:backward(targetFeats,targetDomDfdo,opt.domainLambda) --TODO: verify
		 featExtractor:backward(targetInputs, targetGradDomainClassifier,opt.domainLambda)

		 --print("Domain cost ".. tostring(domCost+targetDomCost))

		 params:narrow(1,1,featExtractorParams:size(1)):copy(featExtractorParams)
		 params:narrow(1,featExtractorParams:size(1),labelPredictorParams:size(1)):copy(labelPredictorParams)
		 params:narrow(1,featExtractorParams:size(1)+labelPredictorParams:size(1),domainClassifierParams:size(1)):copy(domainClassifierParams)
		 gradParams:narrow(1,1,featExtractorGradParams:size(1)):copy(featExtractorGradParams)
		 gradParams:narrow(1,featExtractorGradParams:size(1),labelPredictorGradParams:size(1)):copy(labelPredictorGradParams)
		 gradParams:narrow(1,featExtractorGradParams:size(1)+labelPredictorParams:size(1),domainClassifierGradParams:size(1)):copy(domainClassifierGradParams)

		 return params,gradParams		 
	  end
	  optim.sgd(feval,params,opt)

   end
   print("tick" .. sys.clock()-tick1)
end

prevLoss = 10e12
for i = 1,opt.maxEpoch do
   
   featExtractor:evaluate()
   labelPredictor:evaluate()
   domainClassifier:evaluate()

   local sourceFeats = featExtractor:forward(sourceTrainSet)
   local sourceTrainPred = labelPredictor:forward(sourceFeats)
   local sourceTrainLoss = labelPredictorCriterion:forward(sourceTrainPred, sourceTrainSetLabel) 
   local sourceDomPreds = domainClassifier:forward(sourceFeats)
   local sourceDomCost = domainClassifierCriterion:forward(sourceDomPreds,torch.Tensor(sourceDomPreds:size(1),1):fill(1)) -- TODO: ugly, replace with two unique allocations

   local targetFeats = featExtractor:forward(targetTrainSet)
   local targetDomPreds = domainClassifier:forward(targetFeats)
   local targetDomCost = domainClassifierCriterion:forward(targetDomPreds,torch.Tensor(targetDomPreds:size(1),1):fill(0)) -- TODO: ugly, replace with two unique allocations

   trainConfusion:batchAdd(sourceTrainPred, sourceTrainSetLabel)
   print("EPOCH: " .. i)
   print(trainConfusion)
   print(" + Train loss " .. sourceTrainLoss .. " " .. sourceDomCost+targetDomCost)

   local validPred = labelPredictor:forward(featExtractor:forward(sourceValidSet))
   local validLoss = labelPredictorCriterion:forward(validPred, sourceValidSetLabel) 

   local sourceFeats = featExtractor:forward(sourceValidSet)
   local sourceValidPred = labelPredictor:forward(sourceFeats)
   local sourceValidLoss = labelPredictorCriterion:forward(sourceValidPred, sourceValidSetLabel) 
   local sourceDomPreds = domainClassifier:forward(sourceFeats)
   local sourceDomCostValid = domainClassifierCriterion:forward(sourceDomPreds,torch.Tensor(sourceDomPreds:size(1),1):fill(1)) -- TODO: ugly, replace with two unique allocations

   local targetFeats = featExtractor:forward(targetValidSet)
   local targetDomPreds = domainClassifier:forward(targetFeats)
   local targetDomCostValid = domainClassifierCriterion:forward(targetDomPreds,torch.Tensor(targetDomPreds:size(1),1):fill(0)) -- TODO: ugly, replace with two unique allocations

   validConfusion:batchAdd(validPred, sourceValidSetLabel)
   print(validConfusion)
   print(" + Valid loss " .. validLoss .. " " .. sourceDomCostValid+targetDomCostValid)

   trainLogger:add{i, trainLoss, trainConfusion.totalValid * 100, validLoss, validConfusion.totalValid * 100}
   trainConfusion:zero()
   validConfusion:zero()

   if opt.saveModel then
	  if trainLoss < prevLoss then
		 prevLoss = trainLoss
		 torch.save("model.bin",model)
	  else
		 model = torch.load("model.bin")
	  end
   end

   featExtractor:training()
   labelPredictor:training()
   domainClassifier:training()
   train()
end

