import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from config import Config
from data import SSTDataset

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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids, 
                attention_mask=attention_mask
            )
            
            predictions = torch.argmax(outputs.logits, dim=1)
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
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, 
        num_labels=Config.NUM_LABELS
    ).to(device)
    
    # Load model checkpoint
    checkpoint_path = Config.CHECKPOINT  # Update this path
    model = load_model_checkpoint(model, checkpoint_path, device)
    
    # Create test dataset and dataloader
    test_dataset = SSTDataset(
        split="test", 
        binary=False, 
        model_name="bert-base-uncased", 
        max_length=Config.MAX_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    # Evaluate model
    predictions, true_labels = evaluate_model(
        model, 
        test_loader, 
        device
    )

if __name__ == '__main__':
    main()