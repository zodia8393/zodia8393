import torch
from torch.utils.data import Dataset, DataLoader

# Define constants
MAX_LEN = 256
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
