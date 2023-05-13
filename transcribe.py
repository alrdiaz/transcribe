import whisper

import moviepy.editor as mp

# Provide the path to the video file and the language code ('es' for Spanish)
video_path = "/home/alejandro/Vídeos/Videograbación 2023-05-04 15:35:21.mp4"
language_code = "es"

my_clip = mp.VideoFileClip(video_path)

my_clip.audio.write_audiofile(r"my_result.mp3")


model = whisper.load_model("medium")
# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("./my_result.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions(fp16=False, output_timestamps=True)
result = whisper.decode(model, mel, options)


result = model.transcribe("./my_result.mp3")
# print the recognized text
# # Write the transcript to a text file
with open("transcript.txt", "w") as file:
    file.write(result["text"])


print("Transcription saved to transcript.txt")
