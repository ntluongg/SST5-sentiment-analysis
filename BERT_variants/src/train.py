import os
import torch
from functools import partial

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

import data.utils as dutils
import loss
from options import parsing
from data.sst5 import SSTDataset
from eval import evaluate_model
from model import get_model_from_name

# Training Function
def train_model(args, model, 
                train_dataloader, val_dataloader,
                optimizer, criterion, device, epochs,
                Config=None, outdir='./', scheduler=None):
    
    best_val_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            optimizer.zero_grad()
            
            input_ids, labels, length, start_indexs, end_indexs, span_masks = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            start_indexs = start_indexs.to(device)
            end_indexs = end_indexs.to(device)
            span_masks = span_masks.to(device)
            
            outputs = model(input_ids, start_indexs, end_indexs, span_masks)
            
            a_ij = None
            labels = labels.view(-1)
            if len(outputs)==2:
                pred, a_ij = outputs
            elif len(outputs)==1:
                pred = outputs[0]
            else: assert(0)
            
            loss = criterion(pred, labels)
            if a_ij is not None: 
                loss = loss - 0.01 * a_ij.pow(2).sum(dim=1).mean()
            
            train_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, total=len(val_dataloader)):
                input_ids, labels, length, start_indexs, end_indexs, span_masks = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                start_indexs = start_indexs.to(device)
                end_indexs = end_indexs.to(device)
                span_masks = span_masks.to(device)
                
                outputs = model(input_ids, start_indexs, end_indexs, span_masks)
                
                labels = labels.view(-1)
                if len(outputs)==2:
                    pred, a_ij = outputs
                elif len(outputs)==1:
                    pred = outputs[0]
                else: assert(0)
                    
                # loss = criterion(outputs, labels)
                
                predictions = torch.argmax(pred, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
        if scheduler is not None: 
            scheduler.step(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {np.mean(train_losses):.4f}")
        print(f"Validation F1: {val_f1:.4f}")
        
        # Save best model
        os.makedirs(outdir, exist_ok=True)
        if val_f1 > best_val_f1:
            print("Update best F1: ", val_f1)
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(outdir, f"{args.model_name}_{Config.CHECKPOINT}"))
    
    return model


def main():

    # Argument
    args, Config = parsing()
    
    # Initialize tokenizer and model
    
    model_configs = {
        'config': Config.MODEL_NAME,
        'num_labels': Config.NUM_LABELS,
        'lora': args.lora,
        'full_finetune': args.full_finetune
    }
    
    model = get_model_from_name(args.model_name, model_configs).to(Config.DEVICE)
    
    for n, p in model.named_parameters():
        if p.requires_grad: 
            print(n, end='\t')
    print()
    
    # Create datasets and dataloaders

    train_dataset = SSTDataset(split="train", binary=False, model_name=Config.MODEL_NAME, max_length=Config.MAX_LENGTH)
    val_dataset = SSTDataset(split="dev", binary=False, model_name=Config.MODEL_NAME, max_length=Config.MAX_LENGTH)
    test_dataset = SSTDataset(split="test", binary=False, model_name=Config.MODEL_NAME, max_length=Config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=partial(dutils.collate_self_explain, fill_values=[1, 0, 0]))
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=partial(dutils.collate_self_explain, fill_values=[1, 0, 0]))
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=partial(dutils.collate_self_explain, fill_values=[1, 0, 0]))
    
    # Optimizer and Loss
    
    classifier_params = []
    others = []
    for n, pr in model.named_parameters():
        if 'output' in n or 'classifier' in n:
            classifier_params.append(pr)
        else:
            others.append(pr)
                
    optimizer = optim.AdamW(
        [{
            'params': classifier_params,
            'lr': Config.LEARNING_RATE
         },
         {
            'params': others
         }], lr=3e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    criterion = nn.CrossEntropyLoss()
    # criterion = loss.FocalLoss(gamma=5)
    
    # Train model
    trained_model = train_model(
        args,
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        Config.DEVICE, 
        Config.EPOCHS,
        Config,
        args.outdir,
        scheduler
    )
    
    # Evaluate model
    predictions, true_labels = evaluate_model(
        trained_model, 
        test_loader, 
        Config.DEVICE
    )

if __name__ == '__main__':
    main()
