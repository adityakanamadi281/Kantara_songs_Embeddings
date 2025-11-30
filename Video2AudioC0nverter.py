from moviepy.editor import VideoFileClip




input_paths = [r"C:\Users\adity\Kantara_songs_Embeddings\Video_Songs\Bramhakalasha_part2.webm",
               r"C:\Users\adity\Kantara_songs_Embeddings\Video_Songs\Bramhakalasha.webm",
               r"C:\Users\adity\Kantara_songs_Embeddings\Video_Songs\karma.webm",
               r"C:\Users\adity\Kantara_songs_Embeddings\Video_Songs\Varaha_Roopam.webm"]


output_paths = [r"C:\Users\adity\Kantara_songs_Embeddings\Audio\Bramhakalasha_part2.wav",
                r"C:\Users\adity\Kantara_songs_Embeddings\Audio\Bramhakalasha.wav",
                r"C:\Users\adity\Kantara_songs_Embeddings\Audio\karma.wav",
                r"C:\Users\adity\Kantara_songs_Embeddings\Audio\Varaha_Roopam.wav"]



for input_path, output_path in zip(input_paths, output_paths):
    print(f"Processing: {input_path}")

    video = VideoFileClip(input_path)
    video.audio.write_audiofile(output_path, codec="pcm_s16le")
    video.close()


