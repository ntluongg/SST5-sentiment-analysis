import torch
import torch.nn as nn

# credit to https://github.com/HobbitLong/SupContrast
# credit to https://github.com/JieyuZ2/LCL_loss
# credit to https://github.com/varsha33/LCL_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_matrix = labels_matrix.float()
        
        # Compute numerator (positive pairs)
        positive_mask = labels_matrix - torch.eye(len(labels), device=labels.device)
        positive_pairs = torch.exp(similarity_matrix) * positive_mask
        
        # Compute denominator (all pairs)
        denominator = torch.sum(torch.exp(similarity_matrix), dim=1)
        
        # Compute contrastive loss
        loss = -torch.log(
            torch.sum(positive_pairs, dim=1) / 
            (denominator - torch.exp(torch.diag(similarity_matrix)))
        )
        
        return torch.mean(loss)


class WeightedContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(WeightedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, weights=None, mask=None):
        """
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # index with same label = 1, else = 0
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_feature = features
        anchor_feature = contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # daignal = 0, else = 1
        logits_mask = 1 - torch.eye(batch_size).to(device)

        # mask diagnal and let index with same label = 1
        mask = mask * logits_mask
 
        weighted_mask = weights[:, labels.flatten()]
        # weighted_mask = weighted_mask * logits_mask
        pos_weighted_mask = weighted_mask * mask

        # compute log_prob with logsumexp
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits) * weighted_mask

        ## log_prob = x - max(x1,..,xn) - logsumexp(x1,..,xn) the equation
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (pos_weighted_mask * log_prob).sum(1) / mask.sum(1)

        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss