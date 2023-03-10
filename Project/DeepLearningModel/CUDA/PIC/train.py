import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


def train(model, train_data, dev_data, epochs, batch_size, learning_rate, warmup_proportion=0.1, weight_decay=0.01):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    # Set optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Train loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()

        for batch in tqdm(train_loader):
            # Move batch to device
            batch = {key: val.to(device) for key, val in batch.items()}

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(**batch)
            loss = output.loss

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()
            scheduler.step()

        # Evaluate on dev set
        model.eval()
        num_correct = 0
        num_total = 0

        with torch.no_grad():
            for batch in tqdm(dev_loader):
                # Move batch to device
                batch = {key: val.to(device) for key, val in batch.items()}

                # Forward pass
                output = model(**batch)
                logits = output.logits

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                labels = batch["label"]
                num_correct += torch.sum(predictions == labels).item()
                num_total += len(labels)

        accuracy = num_correct / num_total
        print(f"Dev accuracy: {accuracy:.4f}")

    return model
