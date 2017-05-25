--- DeepCloak: Masking Deep Neural Network Models for Robustness Against Adversarial Samples ---
--- Code: Ji Gao 3/1/2017 ---
require('optim')
require('nn')
require('./RandomMask')
require('MaskLayer')

cmd = torch.CmdLine()
cmd:text('Robust classifier using siamese network')
cmd:text('')
cmd:text("Options")
cmd:option("-gpu", 2, "use gpu num")
cmd:option("-power",10,"Adversarial strength(epsilon in fast gradient method)")
cmd:option("-model",'models/model_vgg_orig.t7', "Filename of original trained model")
cmd:option("-dataset",'cifar10.t7', "Filename of original trained model")
cmd:option("-layernum",54,"Number of layer modified")
cmd:option("-std", 17.067521638107, "Std of the dataset (Used in the adversarial generation)")
cmd:text()
opt = cmd:parse(arg)
cuda= true
opt.datafolder = 'mnist.t7'
opt.trainpath = 'train_32x32.t7'
opt.testpath = 'test_32x32.t7'
opt.normalize = false

if cuda then
	require('cutorch')
	require('cunn')
	require('cudnn')
	cutorch.setDevice(opt.gpu)
	-- torch.setdefaulttensortype('torch.CudaTensor')
end

local std = opt.std
local loss = nn.CrossEntropyCriterion():cuda()
local dataset = torch.load(opt.dataset)
print(dataset)
traindata = dataset.trainData.data
traindata = traindata:cuda()
trainlabels = dataset.trainData.labels
model = torch.load(opt.model)
data = dataset.testData.data
labels = dataset.testData.labels
data=data:cuda()
model=model:cuda()
model:training()

perturbed_train = torch.Tensor(traindata:size()):cuda()
for i = 1,traindata:size(1) do
    -- xlua.progress(i, traindata:size(1))
	local img = traindata[i]:clone()
	if cuda then
		img=img:cuda()
	end
	local img_adv = require('adversarial-fast')(model, loss, img:clone(), trainlabels[i], std, opt.power)
	perturbed_train[i]:copy(img_adv)
end

model = torch.load(opt.model)
model=model:cuda()
model:training()
perturbed = torch.Tensor(data:size()):cuda()
for i = 1,data:size(1) do
	local img = data[i]:clone()
	if cuda then
		img=img:cuda()
	end
	local img_adv = require('adversarial-fast')(model, loss, img:clone(), labels[i], std, opt.power)
	perturbed[i]:copy(img_adv)
end

dataset.perturbed_train = perturbed_train
dataset.perturbed = perturbed
torch.save(opt.model..'perturbedtrain'..opt.power..'.dat',dataset) --- store the data so can be directly loaded

criterion = nn.CrossEntropyCriterion()
criterion = criterion:cuda()
classes = {'0','1','2','3','4','5','6','7','8','9'}
confusion = optim.ConfusionMatrix(#classes)
confusion:zero()

local perturbed = dataset.perturbed
local layernum = opt.layernum
local featuresize = model:get(layernum).output:nElement()
local store = torch.zeros(featuresize):cuda()
model = torch.load(opt.model)
model=model:cuda()
model:evaluate()

local data_split = traindata:split(32,1)
local labels_split = trainlabels:split(32,1)
local perturbed_split = perturbed_train:split(32,1)


for i,v in ipairs(data_split) do
    -- disp progress
    -- xlua.progress(i, traindata:size(1)/32)
    local input = v
    local target = labels_split[i]
    local pred = model:forward(input)
    local err = criterion:forward(pred, target)
    local o1 = model:get(opt.layernum).output:clone()
    confusion:batchAdd(pred, target)
    local input2 = perturbed_split[i]
    local target2 = labels_split[i]
    local pred2 = model:forward(input2)
    local o2 = model:get(opt.layernum).output:clone()
    store:add(torch.abs(o2-o1):sum(1))
end
confusion:updateValids()
print(confusion.totalValid)

y = torch.sort(store,1,true)

data_split = data:split(32,1)
labels_split = labels:split(32,1)
perturbed_split = perturbed:split(32,1)
local res1,res2,nowmax
nowmax = 0
for i = 0,30 do  --- i: Percentage of nodes masked ---
    if i~=0 then
        mask_a = (torch.sign(y[featuresize*i*0.01] - store - 1e-8) + 1) / 2
    else
        mask_a = torch.ones(featuresize)
    end
    model:insert(nn.MaskLayer(featuresize,mask_a),layernum+1)
    model:evaluate()
    model:cuda()
    confusion:zero()
    for j,v in ipairs(data_split) do
        local input = v
        local target = labels_split[j]
        local pred = model:forward(input)
        local err = criterion:forward(pred, target)
        confusion:batchAdd(pred, target)
    end
    confusion:updateValids()
    res1 = confusion.totalValid
    
    confusion:zero()
    for j,v in ipairs(perturbed_split) do
        local img_adv = v:cuda()
        local target = labels_split[j]
        local pred = model:forward(img_adv)
        local err = criterion:forward(pred, target)
        confusion:batchAdd(pred, target)
    end
    confusion:updateValids()
    res2 = confusion.totalValid
    print(""..i.."\t"..res1.."\t"..res2.."")
    model:remove(layernum+1)
end
