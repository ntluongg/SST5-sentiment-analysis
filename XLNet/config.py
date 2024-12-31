import torch

class Config:
    MODEL_NAME = "xlnet-base-cased"
    MAX_LEN = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_LABELS = 5  # SST-5 has 5 labels
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 4
