import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model_checkpoint(model, checkpoint_path, device):
    """Load model weights from a .pth checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def predict_sentiment(model, tokenizer, text, device, num_labels=5):
    """
    Predict sentiment for a given text
    
    Args:
        model (BertForSequenceClassification): Trained BERT model
        tokenizer (BertTokenizer): BERT tokenizer
        text (str): Input text to classify
        device (torch.device): Device to run inference on
        num_labels (int): Number of sentiment classes
    
    Returns:
        dict: Prediction details including class and probabilities
    """
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        truncation=True, 
        max_length=128, 
        padding=True
    ).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Create label mapping (adjust based on your specific labeling)
    label_map = {
        0: 'Very Negative',
        1: 'Negative',
        2: 'Neutral',
        3: 'Positive',
        4: 'Very Positive'
    } if num_labels == 5 else {
        0: 'Negative',
        1: 'Positive'
    }
    
    return {
        'text': text,
        'predicted_class': label_map[predicted_class],
        'probabilities': probabilities.cpu().numpy()[0],
        'predicted_class_index': predicted_class
    }

def infer_from_file(file_path, model, tokenizer, device, num_labels=5):
    """
    Perform inference on texts from a file
    
    Args:
        file_path (str): Path to text file with sentences to classify
        model (BertForSequenceClassification): Trained BERT model
        tokenizer (BertTokenizer): BERT tokenizer
        device (torch.device): Device to run inference on
        num_labels (int): Number of sentiment classes
    
    Returns:
        list: List of prediction dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    predictions = []
    for text in texts:
        text = text.strip()
        if text:  # Skip empty lines
            prediction = predict_sentiment(
                model, 
                tokenizer, 
                text, 
                device, 
                num_labels
            )
            predictions.append(prediction)
    
    return predictions

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    model_name = 'bert-base-uncased'
    num_labels = 5  # Adjust based on your dataset
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = 'path/to/your/model_checkpoint.pth'
    model = load_model_checkpoint(model, checkpoint_path, device)
    
    # Inference on file
    input_file_path = 'path/to/your/input_texts.txt'
    results = infer_from_file(
        input_file_path, 
        model, 
        tokenizer, 
        device, 
        num_labels
    )
    
    # Print results
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Predicted Sentiment: {result['predicted_class']}")
        print(f"Probabilities: {result['probabilities']}\n")

if __name__ == '__main__':
    main()