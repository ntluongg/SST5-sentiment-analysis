def main():
    # File path to SST-5 dataset
    file_path = "sst5_train.tsv"

    # Prepare data
    train_dataloader, val_dataloader = prepare_data(file_path)

    # Load model
    model = load_model()

    # Train model
    model = train_model(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
