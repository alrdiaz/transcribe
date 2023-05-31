import whisper

import moviepy.editor as mp
from datetime import datetime
from progress.bar import ShadyBar
from progress.spinner import Spinner


# print start moment
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("--------------------------------------------------------------------")
print("Start at: ", current_time)
print("--------------------------------------------------------------------")


# Provide the path to the video file and the language code ('es' for Spanish)
# path in windows
video_path = r"C:\Users\alromero\Videos\parte2.mp4"

# path in linux 
# video_path = "/home/alejandro/Vídeos/Videograbación 2023-05-04 15:35:21.mp4"

language_code = "es"

my_clip = mp.VideoFileClip(video_path)

my_clip.audio.write_audiofile(r"my_result.mp3")


model = whisper.load_model("medium")
# load audio and pad/trim it to fit 30 seconds

spinner = Spinner("Loading model to audio ")
spinner.next()
audio = whisper.load_audio("./my_result.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)
spinner.finish()


# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

print("--------------------------------------------------------------------")

# decode the audio
spinner2 = Spinner("Decoding audio ")
spinner2.next()
options = whisper.DecodingOptions(language=language_code,fp16=False)
result = whisper.decode(model, mel, options)


result = model.transcribe("./my_result.mp3")
spinner2.finish()
# print the recognized text
# print(result)


# Calculate timestamps for pauses and speaker changes
timestamps = []
segmentsRange = len(result["segments"])

bar = ShadyBar("Processing text segments", max=segmentsRange, suffix="%(percent)d%%")
for sentence in result["segments"]:
    hoursStart=sentence["start"]
    hhStart=str(hoursStart/3600).split(".")[0]
    minutesStart=float("."+(str(hoursStart/3600).split(".")[1]))*60
    mmStart=str(minutesStart).split(".")[0]
    ssStart=str(round(float("."+(str(minutesStart).split(".")[1]))*60))
    
    hoursEnd=sentence["end"]
    hhEnd=str(hoursEnd/3600).split(".")[0]
    minutesEnd=float("."+(str(hoursEnd/3600).split(".")[1]))*60
    mmEnd=str(minutesEnd).split(".")[0]
    ssEnd=str(round(float("."+(str(minutesEnd).split(".")[1]))*60))
    
    timestamps.append(
        str(hhStart+":"+mmStart+":"+ssStart)
        + " - "
        + str(hhEnd+":"+mmEnd+":"+ssEnd)
        + ":"
        + str(sentence["text"])
    )
    bar.next()
bar.finish()

print("--------------------------------------------------------------------")
# print(timestamps)


# # Write the transcript to a text file
timestampsRange = len(timestamps)
with open("transcript.txt", "w", encoding='utf-8') as file:
    bar2 = ShadyBar("Processing text file", max=timestampsRange, suffix="%(percent)d%%")
    for timestamp in timestamps:
        file.write(f"{timestamp}\n")
        bar2.next()
bar2.finish()

print("--------------------------------------------------------------------")
print("Transcription saved to transcript.txt")
# print start moment
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("--------------------------------------------------------------------")
print("Finish at: ", current_time)
print("--------------------------------------------------------------------")
