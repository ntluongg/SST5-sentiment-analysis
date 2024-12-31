import torch
import torch.nn as nn
import json

class FocalLoss(nn.Module):
    def __init__(self, alpha=torch.tensor([0.78509, 0.37755, 0.52877, 0.38107, 0.64011]), gamma=2):
        super(FocalLoss, self).__init__()
        self.register_buffer('alpha', alpha)  # Register alpha as a buffer to manage devices properly
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Move alpha to the same device as inputs
        if self.alpha.device != inputs.device:
            alpha = self.alpha.to(inputs.device)
        else:
            alpha = self.alpha
        
        # Compute CrossEntropyLoss with class weights
        BCE_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * BCE_loss
        return focal_loss.mean()

    
if __name__ == "__main__":

    # Path to the JSON file
    json_path = "focal_loss_weights.json"

    # Load the weights from the JSON file
    with open(json_path, "r") as json_file:
        weights_dict = json.load(json_file)

    # Convert the weights to a PyTorch tensor
    alpha = torch.tensor(weights_dict["alpha"])

    print("Loaded weights as PyTorch tensor:", alpha)
    criterion = FocalLoss(alpha=alpha, gamma=2)
