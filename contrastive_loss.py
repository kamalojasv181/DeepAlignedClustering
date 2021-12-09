import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        features = features.to(device)
        labels = labels.to(device)

        norm = features.norm(dim = 1)
        norm_features = torch.div(features.T, norm).T
        cosine_similarity_ij = torch.matmul(norm_features, norm_features.T)
        cosine_similarity_ij = torch.div(cosine_similarity_ij, self.temperature)

        exp_similarity = torch.exp(cosine_similarity_ij)
        tot_sum = torch.sub(torch.sum( exp_similarity , 1), torch.diagonal(exp_similarity, 0))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        mask = torch.sub(mask, torch.eye(batch_size).to(device))

        pos_num = torch.sum(mask, 1)
        pos_sum = torch.sum(torch.mul(mask, exp_similarity), 1)

        reference_ones = torch.ones(pos_num.shape[0])
        reference_ones = reference_ones.to(device)

        pos_num = torch.where(pos_num != 0, pos_num, reference_ones)

        exp_prob = torch.div(pos_sum, tot_sum)
        exp_prob = torch.div(exp_prob, pos_num)
        exp_prob = torch.mean(exp_prob)
        log_prob = torch.log(exp_prob)

        loss = -log_prob

        return loss