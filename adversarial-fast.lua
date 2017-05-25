--- Original: https://github.com/e-lab/torch-toolbox/blob/master/Adversarial/adversarial-fast.lua
--- Modified by Ji Gao ---
require('nn')

local function adversarial_fast(model, loss, img, label, std, intensity)
   -- assert(loss.__typename == 'nn.ClassNLLCriterion')
   -- local intensity = intensity or 1
   local x = img:clone()
   local y = label
   -- consider x as batch
   local batch = false
   if x:dim() == 3 then
      x = x:view(1, x:size(1), x:size(2), x:size(3))
      batch = true
   end

   -- consider y as tensor
   if type(y) == 'number' then
      y = torch.Tensor({y}):typeAs(x)
   end

   -- compute output
   -- local y_hat = model:updateOutput(x)

   -- use predication as label if not provided
   local _, target = nil, y
   if target == nil then
      _, target = y_hat:max(y_hat:dim())
   end

   -- find gradient of input (inplace)
   model:zeroGradParameters()
   model:forward(x:cuda())
   local loss_ = loss:forward(model.output, y)
   local df_do = loss:backward(model.output:cuda(), y:cuda())
   model:backward(x:cuda(), df_do:cuda())
   local x_grad=model.modules[1].gradInput
   local noise = x_grad:sign():mul(intensity)
   
   if type(std) == 'number' then
      noise:div(std)
   else
      for c = 1, 3 do
         noise[{{},{c},{},{}}]:div(std[c])
      end
   end

   if batch then
      x = x:view(x:size(2), x:size(3), x:size(4))
   end

   -- return adversarial examples (inplace)
   return x:add(noise)

end

return adversarial_fast
