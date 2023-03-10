import torch
from transformers import AutoTokenizer


class TextClassifier:
    def __init__(self, model_name, num_labels, max_len=256):
        self.model_name = model_name
        self.max_len = max_len
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', self.model_name, num_labels=self.num_labels)
        self.model.to(self.device)

    def classify(self, text):
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoded_dict['input_ids'].to(self.device)
        attention_mask = encoded_dict['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask=attention_mask)

        logits = output[0].squeeze()
        probabilities = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probabilities, dim=-1).item()

        return predicted_label
