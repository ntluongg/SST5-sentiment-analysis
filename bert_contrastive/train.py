import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytreebank
import transformers
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.model_selection import train_test_split

from data import SSTDataset
from model import ContrastiveBERTModel
from loss import ContrastiveLoss
from config import Config

def train_contrastive_bert(model, train_loader, val_loader, optimizer, 
                            classification_criterion, contrastive_criterion, 
                            device, epochs=1):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, embeddings = model(input_ids, attention_mask)
            
            # Compute classification loss
            classification_loss = classification_criterion(logits, labels)
            
            # Compute contrastive loss
            contrastive_loss = contrastive_criterion(embeddings, labels)
            
            # Combined loss
            total_batch_loss = classification_loss + 0.5 * contrastive_loss
            
            # Backward pass
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits, _ = model(input_ids, attention_mask)
                val_classification_loss = classification_criterion(logits, labels)
                val_loss += val_classification_loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {total_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        print(f"Validation Accuracy: {correct_predictions/total_predictions:.4f}")

def main(): 
    # Initialize tokenizer and device
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = SSTDataset(split="train", binary=False, model_name="bert-base-uncased", max_length=Config.MAX_LENGTH)
    val_dataset = SSTDataset(split="dev", binary=False, model_name="bert-base-uncased", max_length=Config.MAX_LENGTH)
    test_dataset = SSTDataset(split="test", binary=False, model_name="bert-base-uncased", max_length=Config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = ContrastiveBERTModel().to(device)
    
    # Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    classification_criterion = nn.CrossEntropyLoss()
    contrastive_criterion = ContrastiveLoss()
    
    # Train
    train_contrastive_bert(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        classification_criterion, 
        contrastive_criterion, 
        device
    )

if __name__ == '__main__':
    main()