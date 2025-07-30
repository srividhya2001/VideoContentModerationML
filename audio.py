import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Load the pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio
audio_path = "audio.wav"
speech, sr = librosa.load(audio_path, sr=16000) # Assuming 16kHz sampling rate

# Process audio
input_values = processor(speech, return_tensors="pt", padding="longest", do_normalize=True, sampling_rate=sr).input_values # (Batch size, sequence length)

# Extract embeddings
with torch.no_grad():
    embeddings = model(input_values).last_hidden_state # (Batch size, sequence length, hidden_size)

print(embeddings)
