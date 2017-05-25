--- Author: Ji Gao ---
local MaskLayer, Parent = torch.class('nn.MaskLayer', 'nn.Module')

function MaskLayer:__init(size, mask)
    Parent.__init(self)
    self.train = true
    self.mask = torch.Tensor(size)
    self.mask:copy(mask)
end

function MaskLayer:updateOutput(input)
    self.output:resizeAs(input):copy(input)
    -- print(self.output)
    if not self.train then
        if self.output:dim() > self.mask:dim() then
            -- Batch
            -- print(self.mask:repeatTensor(self.output:size(1), 1))
            mask1 = torch.repeatTensor(self.mask,self.output:size(1), 1)
            self.output:cmul(mask1)
        else
            self.output:cmul(self.mask)
        end
    end
    return self.output
end

function MaskLayer:updateGradInput(input, gradOutput)
    if self.train then
        self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    else
        self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    end
    return self.gradInput
end

function MaskLayer:setmask(mask)
    self.mask:resizeAs(mask):copy(mask)
end

function MaskLayer:__tostring__()
    return string.format('%s', torch.type(self))
end
