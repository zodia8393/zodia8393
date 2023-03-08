#NER PERSONAL INFORMATION CENSOR MODEL PROTOTYPE IMPROVED VERSION

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# define constants
MAX_LEN = 256
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {'contradiction': 0, 'neutral': 1, 'entailment': 2}


def read_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]  # skip header
    data = [line.strip().split('\t') for line in lines]
    return data


def preprocess_data(data):
    preprocessed_data = []
    for premise, hypothesis, label in data:
        premise = premise.replace('\n', ' ').strip()
        hypothesis = hypothesis.replace('\n', ' ').strip()
        label = LABEL_MAP[label]
        preprocessed_data.append((premise, hypothesis, label))
    return preprocessed_data


class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        premise, hypothesis, label = self.data[idx]
        encoded_pair = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=MAX_LEN,
            truncation_strategy='longest_first'
        )
        input_ids = encoded_pair['input_ids']
        token_type_ids = encoded_pair['token_type_ids']
        attention_mask = encoded_pair['attention_mask']
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)


def get_data_loaders(train_file, dev_file, tokenizer):
    train_data = preprocess_data(read_data(train_file))
    dev_data = preprocess_data(read_data(dev_file))

    train_dataset = TextClassificationDataset(train_data, tokenizer)
    dev_dataset = TextClassificationDataset(dev_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return train_loader, dev_loader


def main():
    # check if GPU is available
    if DEVICE == torch.device("cuda"):
        print('Using GPU for training.')
    else:
        print('Using CPU for training.')

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    # get data loaders
    train_loader, dev_loader = get_data_loaders('train.tsv', 'dev.tsv', tokenizer)

    # print sample batch
    sample_batch = next(iter(train_loader))
    print(sample_batch)


if __name__ == '__main__':
    main()

import torch
from transformers import BertForTokenClassification, BertTokenizerFast

# Check if a GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=3)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# Load the processed data
with open('processed_data.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Tokenize and align labels
tokenized_inputs = []
for line in lines:
    tokens = tokenizer.encode(line.strip(), add_special_tokens=True)
    labels = ['O'] * len(tokens)
    tokenized_inputs.append((tokens, labels))

# Convert tokenized inputs to PyTorch tensors
input_ids = torch.tensor([tokens for tokens, labels in tokenized_inputs], dtype=torch.long)
attention_mask = torch.tensor([[int(i > 0) for i in tokens] for tokens, labels in tokenized_inputs], dtype=torch.long)
labels = torch.tensor([model.config.label2id[label] for tokens, labels in tokenized_inputs for label in labels], dtype=torch.long)

# Put the tensors on the GPU if available
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
labels = labels.to(device)

# Evaluate the model on the processed data
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

# Convert predictions back to labels
id2label = {id: label for label, id in model.config.label2id.items()}
predicted_labels = []
for prediction, tokens in zip(predictions.tolist(), tokenized_inputs):
    predicted_labels.append([id2label[p] for p in prediction[1:len(tokens[0])-1]])

# Print the predicted labels
for line, predicted in zip(lines, predicted_labels):
    print(f"{line.strip()} --> {' '.join(predicted)}")

import torch
import transformers
import pandas as pd

# Load the model and tokenizer
model_name = "dslim/bert-base-NER"
model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Check if a GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define function to tokenize text
def tokenize_text(text):
    tokenized_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt"
    )
    return tokenized_text

# Define function to censor entities in text
def censor_entities(text, entity_labels):
    tokenized_text = tokenize_text(text)
    input_ids = tokenized_text["input_ids"].to(device)
    attention_mask = tokenized_text["attention_mask"].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    label_indices = torch.argmax(output[0], axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    output = []
    for token, label_idx in zip(tokens, label_indices[0]):
        label = model.config.id2label[label_idx]
        if label in entity_labels:
            token = "[CENSORED]"
        output.append(token)
    censored_text = tokenizer.convert_tokens_to_string(output)
    return censored_text

# Load the data and preprocess it
data = pd.read_csv("data.csv")
data = data.fillna("")
data["text"] = data["text"].apply(lambda x: x.replace("\n", " ").strip())
data["censored_text"] = data["text"].apply(lambda x: censor_entities(x, ["PER", "PHONE", "EMAIL"]))

# Save the censored data
data.to_csv("censored_data.csv", index=False)
