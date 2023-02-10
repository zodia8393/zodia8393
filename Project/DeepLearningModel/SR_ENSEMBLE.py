#음성인식 Pre-trained Model Ensemble 

import DeepSpeech
import PyTorch_Kaldi
import SpeechRecognition
import wave
import contextlib
import os
import pandas as pd

def ensemble_models(speech_file):
    with contextlib.closing(wave.open(speech_file,'r')) as speech_input:
        frames = speech_input.getnframes()
        rate = speech_input.getframerate()
        speech_data = speech_input.readframes(frames)
        
        prediction1 = DeepSpeech.predict(speech_data, rate)
        prediction2 = PyTorch-Kaldi.predict(speech_data, rate)
        prediction3 = SpeechRecognition.predict(speech_data, rate)
        
        final_prediction = majority_voting(prediction1, prediction2, prediction3)
        return final_prediction
    
def majority_voting(prediction1, prediction2, prediction3):
    predictions = [prediction1, prediction2, prediction3]
    return max(set(predictions), key = predictions.count)


path = '/path/to/directory'
files = os.listdir(path)

wav_files = [f for f in files if f.endswith('.wav')]

results = []
for wav_file in wav_files:
    speech_file = os.path.join(path, wav_file)
    result = ensemble_models(speech_file)
    results.append([wav_file, result])

df = pd.DataFrame(results, columns=['wav_file', 'result'])
df.to_excel('results.xlsx', index=False)


#DeepSpeech Model Pre-train Code
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Load pre-trained DeepSpeech model
model = torch.hub.load('mozilla/DeepSpeech', 'deepspeech', pretrained=True)

# Define the loss function
criterion = nn.CTCLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load the training data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the model
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Print the training progress
    if (epoch+1) % print_every == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

# Save the fine-tuned model
torch.save(model.state_dict(), 'deepspeech_finetuned.pth')

    
