#음성인식 Pre-trained Model Ensemble 

import DeepSpeech
import PyTorch_Kaldi
import SpeechRecognition
import wave
import contextlib

def ensemble_models(speech_file):
    with contextlib.closing(wave.open(speech_file,'r')) as speech_input:
        frames = speech_input.getnframes()
        rate = speech_input.getframerate()
        speech_data = speech_input.readframes(frames)
        
        prediction1 = DeepSpeech.predict(speech_data, rate)
        prediction2 = PyTorch_Kaldi.predict(speech_data, rate)
        prediction3 = SpeechRecognition.predict(speech_data, rate)
        
        final_prediction = majority_voting(prediction1, prediction2, prediction3)
        return final_prediction
    
def majority_voting(prediction1, prediction2, prediction3):
    predictions = [prediction1, prediction2, prediction3]
    return max(set(predictions), key = predictions.count)
