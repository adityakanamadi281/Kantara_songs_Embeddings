from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import librosa



feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")



audio_files = [r"C:\Users\adity\Kantara_songs_Embeddings\Audio\Bramhakalasha_part2.wav",
                r"C:\Users\adity\Kantara_songs_Embeddings\Audio\Bramhakalasha.wav",
                r"C:\Users\adity\Kantara_songs_Embeddings\Audio\karma.wav",
                r"C:\Users\adity\Kantara_songs_Embeddings\Audio\Varaha_Roopam.wav",
                r"C:\Users\adity\Kantara_songs_Embeddings\Audio\rebel.wav"]


embeddings = [] 

for file in audio_files:
    print(f"Processing: {file}")

    # Load one file at a time
    audio, rate = librosa.load(file, sr=16000)

    # Extract features
    inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=rate)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)   # mean pooling

    embeddings.append(emb)

    print(f"Embedding shape: {emb.shape}\n")


final_embeddings = torch.stack(embeddings)
print("Final embeddings shape:", final_embeddings.shape)

# Save the Embeddings
# torch.save(final_embeddings, r"C:\Users\adity\Kantara_songs_Embeddings\kantara_songs_Embeddings.pt")

final_embeddings = final_embeddings.squeeze(1) 
final_embeddings_list = final_embeddings.numpy().tolist()
print("Final embeddings shape:", final_embeddings_list.shape)

items = []
for i, file in enumerate(audio_files):
    items.append({
        "id": f"audio_{i}",
        "embedding": final_embeddings_list[i],
        "metadata": {"filename": file}
    })






#  Vector Database Integration 

import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection(name="audio_embeddings")


collection.add(
    embeddings=final_embeddings_list,
    metadatas=[{"filename": file} for file in audio_files],
    ids=[f"audio_{i}" for i in range(len(audio_files))]
)

print("Stored in ChromaDB!")

