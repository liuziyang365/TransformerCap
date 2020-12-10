import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    def __init__(self, mode='logit'):
        super(LanguageModelCriterion, self).__init__()
        self.mode = mode

    def forward(self, input, target, mask):

        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)].to(input)
        if self.mode == 'logit':
            input = F.log_softmax(input, dim=-1)
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        # Average over each token
        output = torch.sum(output) / torch.sum(mask)

        return output
