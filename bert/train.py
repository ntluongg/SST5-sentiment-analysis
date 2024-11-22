import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import pytreebank

from config import Config
from data import SSTDataset
from eval import evaluate_model

# Training Function
def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs):
    best_val_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            train_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {np.mean(train_losses):.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_bert_sst5_model.pth')

            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': loss,}, 'best_bert_sst5_model.pth')
    
    return model


def main():

     
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, 
        num_labels=Config.NUM_LABELS
    ).to(Config.DEVICE)
    
    # Create datasets and dataloaders

    train_dataset = SSTDataset(split="train", binary=False, model_name="bert-base-uncased", max_length=Config.MAX_LENGTH)
    val_dataset = SSTDataset(split="dev", binary=False, model_name="bert-base-uncased", max_length=Config.MAX_LENGTH)
    test_dataset = SSTDataset(split="test", binary=False, model_name="bert-base-uncased", max_length=Config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        Config.DEVICE, 
        Config.EPOCHS
    )
    
    # Evaluate model
    predictions, true_labels = evaluate_model(
        trained_model, 
        test_loader, 
        Config.DEVICE
    )

if __name__ == '__main__':
    main()
