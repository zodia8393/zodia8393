import torch
import transformers


class TokenClassifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = self.model.config.id2label

    def classify(self, text):
        tokenized_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids = tokenized_text["input_ids"].to(self.device)
        attention_mask = tokenized_text["attention_mask"].to(self.device)
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
        label_indices = torch.argmax(output[0], axis=2)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = [self.id2label[label_idx] for label_idx in label_indices[0]]
        return list(zip(tokens, labels))
