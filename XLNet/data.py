from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler


def create_dataloaders(train_inputs, train_masks, train_labels, val_inputs, val_masks, val_labels):
    """
    Create DataLoaders for training and validation datasets.
    """
    # Create TensorDatasets
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    val_data = TensorDataset(val_inputs, val_masks, val_labels)

    # Create Samplers
    train_sampler = RandomSampler(train_data)
    val_sampler = SequentialSampler(val_data)

    # Create DataLoaders
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=Config.BATCH_SIZE)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=Config.BATCH_SIZE)

    return train_dataloader, val_dataloader


def prepare_data(file_path):
    """
    Load, preprocess, and create DataLoaders for the dataset.
    """
    tokenizer = XLNetTokenizer.from_pretrained(Config.MODEL_NAME, do_lower_case=True)
    sentences, labels = load_data(file_path)
    input_ids, attention_masks, labels = preprocess_data(sentences, labels, tokenizer)

    # Split data into train and validation sets
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.1,
                                                                          random_state=42)
    train_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=0.1, random_state=42)

    # Convert to tensors
    train_inputs = torch.tensor(train_inputs)
    val_inputs = torch.tensor(val_inputs)
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)

    # Create DataLoaders
    train_dataloader, val_dataloader = create_dataloaders(train_inputs, train_masks, train_labels, val_inputs,
                                                          val_masks, val_labels)

    return train_dataloader, val_dataloader
