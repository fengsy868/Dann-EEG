
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
   cmd:option('-domainLambda', 10, 'regularization term for transfer learning')
   cmd:option('-batchSize', 500, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-maxEpoch', 200, 'maximum nb of epoch')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-hiddenLayerUnits', 56 , 'Number of hidden layer units')
   cmd:text()
   opt = cmd:parse(arg or {})
end
torch.manualSeed(opt.seed)
trainLogger = optim.Logger(paths.concat("results", opt.save))
trainLogger:setNames({'epoch','trainLoss','trainAccuracy','validLoss','validAccuracy'})

classes = {'0','1'}
trainConfusion = optim.ConfusionMatrix(classes)
validConfusion = optim.ConfusionMatrix(classes)
trainConfusion:zero()
validConfusion:zero()
--------------------------------
--tensor to store the result
x = torch.Tensor()
--------------------------------
nTrainS1 = 6000
nValidS1 = 6000
EEGDataset1 = torch.load('EEG.torch')
nInputEEG = EEGDataset1:size(3)

sourceTrainSet = torch.Tensor(nTrainS1, nInputEEG)
sourceTrainSet:copy(EEGDataset1[1]:narrow(1,1,nTrainS1):float())
sourceTrainSetLabel = torch.Tensor(nTrainS1, nInputEEG)
sourceTrainSetLabel:copy(EEGDataset1[1]:narrow(1,1,nTrainS1):float())

sourceValidSet = torch.Tensor(nTrainS1,nInputEEG)
sourceValidSet:copy(EEGDataset1[1]:narrow(1,nTrainS1+1,nValidS1):float())
sourceValidSetLabel = torch.Tensor(nValidS1, nInputEEG)
sourceValidSetLabel:copy(EEGDataset1[1]:narrow(1,nTrainS1+1,nValidS1):float())

sourceInputs = torch.Tensor(opt.batchSize,sourceTrainSet:size(2)) 
sourceLabels = torch.Tensor(opt.batchSize,sourceTrainSet:size(2))                                        


nTrainS2 = 6000
nValidS2 = 6000
EEGDataset2 = torch.load('EEGMat.torch')
nInputEEG = EEGDataset2:size(3)

targetTrainSet = torch.Tensor(nTrainS2, nInputEEG)
targetTrainSet:copy(EEGDataset2[1]:narrow(1,1,nTrainS2):float())
targetTrainSetLabel = torch.Tensor(nTrainS2, nInputEEG)
targetTrainSetLabel:copy(EEGDataset2[1]:narrow(1,1,nTrainS2):float())

targetValidSet = torch.Tensor(nTrainS2,nInputEEG)
targetValidSet:copy(EEGDataset2[1]:narrow(1,nTrainS2+1,nValidS2):float())
targetValidSetLabel = torch.Tensor(nValidS2, nInputEEG)
targetValidSetLabel:copy(EEGDataset2[1]:narrow(1,nTrainS2+1,nValidS2):float())

targetInputs = torch.Tensor(opt.batchSize,targetTrainSet:size(2)) 
targetLabels = torch.Tensor(opt.batchSize,targetTrainSet:size(2)) 


-- Definition of the encoder
featExtractor = nn.Sequential()
--featExtractor:add(nn.Reshape(nInputEEG))
featExtractor:add(nn.Linear(nInputEEG,opt.hiddenLayerUnits))
-- featExtractor:add(nn.Sigmoid())
-- featExtractor:add(nn.Linear(5,opt.hiddenLayerUnits))
featExtractor:add(nn.ReLU())
-- featExtractor:add(nn.Linear(5,5))

-- Definition of the decoder
labelPredictor = nn.Sequential()
labelPredictor:add(nn.Linear(opt.hiddenLayerUnits,nInputEEG))
-- labelPredictor:add(nn.Sigmoid())
-- labelPredictor:add(nn.Linear(5,nInputEEG))
labelPredictor:add(nn.ReLU())

-- Definition of the domain classifier
domainClassifier = nn.Sequential()
domainClassifier:add(nn.GradientReversal(opt.domainLambda))
domainClassifier:add(nn.Linear(opt.hiddenLayerUnits,1))
domainClassifier:add(nn.ReLU())
-- domainClassifier:add(nn.LogSoftMax())


-- Definition of the criterion
labelPredictorCriterion = nn.MSECriterion()
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

featExtractorParams = torch.ones(featExtractorParams:size())
labelPredictorParams = torch.ones(labelPredictorParams:size())
domainClassifierParams = torch.ones(domainClassifierParams:size())

print("feat " .. tostring(featExtractorParams:size()))
print("label " .. tostring(labelPredictorParams:size()))
print("domain " .. tostring(domainClassifierParams))



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

       -- print("Label cost ".. tostring(labelCost))

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

       -- print("Domain cost ".. tostring(domCost+targetDomCost))

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

   -- trainConfusion:batchAdd(sourceDomPreds, torch.Tensor(sourceDomPreds:size(1),1):fill(1))
   print("EPOCH: " .. i)

   print("LearningRate: "..opt.learningRate)
   -- print(trainConfusion)
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

   -- validConfusion:batchAdd(targetDomPreds, torch.Tensor(targetDomPreds:size(1),1):fill(0))
   -- print(validConfusion)
   print(" + Valid loss " .. validLoss .. " " .. sourceDomCostValid+targetDomCostValid)

   -- trainLogger:add{i, trainLoss, trainConfusion.totalValid * 100, validLoss, validConfusion.totalValid * 100}
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

   if i == opt.maxEpoch then
      file = csvigo.load("exp2/exp_ReLU.csv")
      table.insert(file["ValidMSE"], validLoss)
      table.insert(file["SourceDomLossValid"], sourceDomCostValid)
      table.insert(file["TargetDomLossValid"], targetDomCostValid)
      table.insert(file["Domainloss"], sourceDomCostValid+targetDomCostValid)
      table.insert(file["DomainLambda"], opt.domainLambda)
      table.insert(file["HiddenUnits"], opt.hiddenLayerUnits)
      table.insert(file["LearningRate"], opt.learningRate)
      table.insert(file["MaxEpoch"], opt.maxEpoch)
      csvigo.save("exp2/exp_ReLU.csv", file)
   end

   -- file = csvigo.load("exp1/exp1.csv")
   -- if prevLoss > sourceTrainLoss then
   --    prevLoss = sourceTrainLoss
   --    opt.learningRate = opt.learningRate * 1.2
   -- else
   --    prevLoss = sourceTrainLoss
   --    opt.learningRate = opt.learningRate / 2
   -- end

   -- opt.learningRate = 0.1 / i
   -- opt.learningRate = 0.1 / math.sqrt(i)
   -- table.insert(file["ValidMSE"], validLoss)
   -- table.insert(file["Domainloss"], sourceDomCostValid+targetDomCostValid)
   -- table.insert(file["DomainLambda"], opt.domainLambda)
   -- table.insert(file["HiddenUnits"], opt.hiddenLayerUnits)
   -- table.insert(file["LearningRate"], opt.learningRate)
   -- table.insert(file["Epoch"], i)
   -- table.insert(file["Exp"], '7')
   -- table.insert(file["etacoe"], '1')
   -- csvigo.save("exp1/exp1.csv", file)


end


