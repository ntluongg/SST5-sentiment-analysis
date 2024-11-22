import torch.nn as nn
from transformers import BertModel

class ContrastiveBERTModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=5, embedding_dim=768):
        super(ContrastiveBERTModel, self).__init__()
        
        # BERT Base Model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Projection Head for Contrastive Learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        bert_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = bert_output.last_hidden_state[:, 0, :]
        
        # Projection for contrastive learning
        projected_embedding = self.projection_head(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits, projected_embedding