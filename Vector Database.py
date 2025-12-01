import chromadb
import glob 
import os 


audio_folder = "Audio"
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))




final_embeddings= r"C:\Users\adity\Kantara_songs_Embeddings\kantara_songs_Embeddings.pt"


final_embeddings = final_embeddings.squeeze(1) 
final_embeddings_list = final_embeddings.numpy().tolist()


items = []
for i, file in enumerate(audio_files):
    items.append({
        "id": f"audio_{i}",
        "embedding": final_embeddings_list[i],
        "metadata": {"filename": file}
    })





client = chromadb.Client()
collection = client.get_or_create_collection(name="audio_embeddings")


collection.add(
    embeddings=final_embeddings_list,
    metadatas=[{"filename": file} for file in audio_files],
    ids=[f"audio_{i}" for i in range(len(audio_files))]
)

print("Stored in ChromaDB!")



