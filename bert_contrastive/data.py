import pytreebank
import torch
import logging
from transformers import BertTokenizer
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSTDataset(Dataset):
    """Configurable SST Dataset with support for modern Transformers library."""
    def __init__(
        self, 
        split="train", 
        root=True, 
        binary=True, 
        model_name="bert-base-uncased", 
        max_length=128
    ):
        """
        Initialize the SST dataset.
        
        Args:
            split (str): Dataset split, one of [train, val, test]
            root (bool): If true, only use root nodes. Else, use all nodes.
            binary (bool): If true, use binary labels. Else, use fine-grained.
            model_name (str): Transformers model name for tokenization
            max_length (int): Maximum sequence length
        """
        logger.info(f"Loading SST {split} set")
        
        # Load SST dataset
        sst = pytreebank.load_sst()
        self.sst = sst[split]
        
        # Initialize tokenizer
        logger.info(f"Initializing tokenizer: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Configure dataset based on parameters
        self.max_length = max_length
        self.binary = binary
        self.root = root
        
        # Prepare data
        logger.info("Preparing dataset")
        self.data = self._prepare_data()
    
    def _get_label(self, label):
        """Convert label based on configuration."""
        if self.binary:
            if label < 2:
                return 0  # Negative
            elif label > 2:
                return 1  # Positive
            else:
                # Neutral case - decide how to handle
                return -1  # or raise ValueError
        return label  # Fine-grained label
    
    def _prepare_data(self):
        """Prepare dataset based on root and binary configurations."""
        processed_data = []
        
        if self.root:
            # Only use root nodes
            for tree in self.sst:
                # Skip neutral in binary case
                label = self._get_label(tree.label)
                if label != -1:
                    processed_data.append((tree.to_lines()[0], label))
        else:
            # Use all nodes
            for tree in self.sst:
                for label, line in tree.to_labeled_lines():
                    converted_label = self._get_label(label)
                    if converted_label != -1:
                        processed_data.append((line, converted_label))
        
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Returns tokenized input and label.
        
        Returns:
            dict: Contains input_ids, attention_mask, and label
        """
        text, label = self.data[index]
        
        # Tokenize with transformers
        encoding = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def main():
    # Example usage
    try:
        # Binary, root-only dataset
        binary_dataset = SSTDataset(
            split='train', 
            root=True, 
            binary=True,
            model_name='bert-base-uncased'
        )
        
        # Fine-grained, all nodes dataset
        fine_grained_dataset = SSTDataset(
            split='train', 
            root=False, 
            binary=False,
            model_name='bert-base-uncased'
        )
        
        # Print some stats
        print(f"Binary Dataset Size: {len(binary_dataset)}")
        print(f"Fine-grained Dataset Size: {len(fine_grained_dataset)}")
        
        # Demonstrate data retrieval
        print("\nSample from Binary Dataset:")
        sample_binary = binary_dataset[0]
        print(f"Input IDs shape: {sample_binary['input_ids'].shape}")
        print(f"Label: {sample_binary['labels']}")
        
        print("\nSample from Fine-grained Dataset:")
        sample_fine = fine_grained_dataset[0]
        print(f"Input IDs shape: {sample_fine['input_ids'].shape}")
        print(f"Label: {sample_fine['labels']}")
    
    except Exception as e:
        logger.error(f"Error in dataset processing: {e}")

if __name__ == '__main__':
    main()
