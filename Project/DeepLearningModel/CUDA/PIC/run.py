from preprocessing import read_data, preprocess_data, TextClassificationDataset, get_data_loaders
from model import TextClassifier
from train import train


def main():
    # Define constants
    TRAIN_FILE = "train.tsv"
    DEV_FILE = "dev.tsv"
    MODEL_NAME = "bert-base-cased"
    NUM_LABELS = 3
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 5e-5
    WARMUP_PROPORTION = 0.1
    WEIGHT_DECAY = 0.01

    # Read and preprocess data
    train_data = preprocess_data(read_data(TRAIN_FILE))
    dev_data = preprocess_data(read_data(DEV_FILE))

    # Load the tokenizer
    tokenizer = TextClassifier(MODEL_NAME, NUM_LABELS, MAX_LEN).tokenizer

    # Create data loaders
    train_loader, dev_loader = get_data_loaders(train_data, dev_data, tokenizer, batch_size=BATCH_SIZE)

    # Train the model
    model = TextClassifier(MODEL_NAME, NUM_LABELS, MAX_LEN).model
    model = train(model, train_loader, dev_loader, EPOCHS, BATCH_SIZE, LEARNING_RATE, WARMUP_PROPORTION, WEIGHT_DECAY)

    # Save the model
    torch.save(model.state_dict(), "text_classifier.pt")


if __name__ == '__main__':
    main()
