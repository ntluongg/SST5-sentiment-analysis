def train_model(model, train_dataloader, val_dataloader):
    """
    Train the XLNet model using DataLoader.
    """
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        print(f'Epoch {epoch + 1}')

        # Training loop
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch_inputs, batch_masks, batch_labels = tuple(t.to(Config.DEVICE) for t in batch)

            model.zero_grad()
            outputs = model(batch_inputs, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Training loss: {total_loss / len(train_dataloader)}')

        # Validation loop
        model.eval()
        eval_loss = 0
        for batch in val_dataloader:
            batch_inputs, batch_masks, batch_labels = tuple(t.to(Config.DEVICE) for t in batch)
            with torch.no_grad():
                outputs = model(batch_inputs, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
                loss = outputs[0]
                eval_loss += loss.item()

        print(f'Validation loss: {eval_loss / len(val_dataloader)}')

    return model
