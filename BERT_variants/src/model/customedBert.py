import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

import loralib as lora

class CustomedBert(nn.Module):
    def __init__(self, config, num_labels, 
                 lora=False, full_finetune=False):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.hidden_dim = 768 if "base" in config else 1024
        
        self.encoder = BertModel.from_pretrained(self.config)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_labels)
        )
        
        if lora: self.lora_setup(8, 16)
        if not full_finetune:
            for param in self.encoder.parameters():
                param.requires_grad=False
        
    def lora_setup(self, lora_r, lora_alpha):
        """
        Apply LoRA to all the Linear layers in the Vision Transformer model.
        """
        # Step 1: Collect the names of layers to replace
        layers_to_replace = []
        
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Linear) :
                if 'qkv' in name or 'proj' in name:
                    # Collect layers for replacement (store name and module)
                    layers_to_replace.append((name, module))
        
        # Step 2: Replace the layers outside of the iteration
        for name, module in layers_to_replace:
            # Create the LoRA-augmented layer
            lora_layer = lora.Linear(module.in_features, module.out_features, r=lora_r, lora_alpha=lora_alpha)
            # Copy weights and bias
            lora_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.bias.data = module.bias.data.clone()

            # Replace the layer in the model
            parent_name, layer_name = name.rsplit('.', 1)
            parent_module = dict(self.tile_encoder.named_modules())[parent_name]
            setattr(parent_module, layer_name, lora_layer)
        
    def forward(self, input_ids, start_indexs, end_indexs, span_masks, attention_mask=None):
        # generate mask
        attention_mask = (input_ids != 1).long()
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs[0][:,0,:]
        return [self.classifier(features)]
        
        