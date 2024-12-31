import os
import torch
from functools import partial
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

import data.utils as dutils
from options import parsing
from data.sst5 import SSTDataset
from model import get_model_from_name


def load_model_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from a .pth checkpoint
    
    Args:
        model (BertForSequenceClassification): Pre-initialized model
        checkpoint_path (str): Path to the .pth checkpoint file
        device (torch.device): Device to load the model to
    
    Returns:
        BertForSequenceClassification: Model with loaded weights
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint)
        model.to(device)
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
        
        # Optional: Print additional checkpoint information
        if 'epoch' in checkpoint:
            print(f"Loaded model from epoch {checkpoint['epoch']}")
        
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise



def evaluate_model(model, test_dataloader, device):
    """
    Evaluate the model on the test dataset
    
    Args:
        model (BertForSequenceClassification): Trained model
        test_dataloader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
    
    Returns:
        tuple: Predictions and true labels
    """
    model.eval()
    test_predictions = []
    test_true_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
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
            
            predictions = torch.argmax(pred, dim=1)
            test_predictions.extend(predictions.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(test_true_labels, test_predictions))
    
    # Calculate and print accuracy
    accuracy = accuracy_score(test_true_labels, test_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    return test_predictions, test_true_labels

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    # Load model checkpoint
    checkpoint_path = os.path.join(args.outdir, f"{args.model_name}_{Config.CHECKPOINT}")  # Update this path
    model = load_model_checkpoint(model, checkpoint_path, device)
    
    # Create test dataset and dataloader
    test_dataset = SSTDataset(
        split="test", binary=False,
        model_name=Config.MODEL_NAME,
        max_length=Config.MAX_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        collate_fn=partial(dutils.collate_self_explain, fill_values=[1, 0, 0])
    )
    
    # Evaluate model
    predictions, true_labels = evaluate_model(
        model, 
        test_loader, 
        device
    )

if __name__ == '__main__':
    main()