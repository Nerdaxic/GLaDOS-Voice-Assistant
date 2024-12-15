#!/usr/bin/python3
#   ____ _          ____   ___  ____  
#  / ___| |    __ _|  _ \ / _ \/ ___| 
# | |  _| |   / _` | | | | | | \___ \ 
# | |_| | |__| (_| | |_| | |_| |___) |
#  \____|_____\__,_|____/ \___/|____/ 
#                                     
#    Open source voice assistant by nerdaxic
#
#    Local wakeword detection using openWakeWord
#    Using local Speech-to-Text using OpenAI's Whisper
#    Using local Text-to-Speech using Piper
#    Works with Home Assistant
#
#    https://github.com/nerdaxic/glados-voice-assistant/
#    https://www.nerdaxic.com/
#
#    Rename settings.env.sample to settings.env
#    Edit settings.env to match your setup
#

# Import basic voice assistant functionality from external files
from gladosTTS import *
from gladosTime import *
from gladosHA import *
from gladosSerial import *
from gladosServo import *
from glados_functions import *

# Import skills
from skills.glados_jokes import *
#from skills.glados_magic_8_ball import *
from skills.glados_home_assistant import *

# System functions
import subprocess
import datetime as dt
import os
import random
import psutil
import signal
import sys
import requests

# Local Speech Recognition and Generation
from openwakeword.model import Model      # Wakeword detector
import whisper                            # Speech-to-text
import pyaudio                            # Audio streams and processing
import wave
import numpy as np
from scipy.io import wavfile
import nltk

# Large Language Model for text generation
from ollama import chat
import torch
import threading
import queue
import io
import re
import time

# Load settings from settings.env
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.dirname(os.path.abspath(__file__))+'/settings.env')
application_path = os.path.dirname(os.path.abspath(__file__))

# Main LLM model for text generation via Ollama
llm_model = "lstep/neuraldaredevil-8b-abliterated:q8_0"

# Main TTS model used for speech generation
tts_model = application_path+'/models/piper/glados.onnx'
tts_wakeWord = "Hey GLaDOS"

# Load system prompt conditionally
system_prompt_path = application_path+'/prompts/llm_glados_tone_of_voice.txt'

print(tts_model)
print(system_prompt_path)

# Try loading the "personality" of the voice assistant
system_message = None
if os.path.exists(system_prompt_path) and os.path.getsize(system_prompt_path) > 0:
    with open(system_prompt_path, 'r') as f:
        system_message = f.read().strip()

# Load Whisper
# Check if CUDA is available and load the appropriate Whisper model
if torch.cuda.is_available():
    print("\033[1;94m[STARTUP]\033[;97m CUDA detected. Loading Whisper 'medium.en' for transcription.")
    model = whisper.load_model("medium.en")
else:
    print("\033[1;94m[STARTUP]\033[;97m No CUDA detected. Loading Whisper 'small.en' for CPU-based transcription.")
    model = whisper.load_model("small.en")

# Function to preprocess text into audio tracks (in-memory)
def tts_preprocessor(input_queue, track_queue, tts_model):
    while True:
        line = input_queue.get()
        if line is None:  # Exit signal
            break

        # Remove formatting/markup from the line
        line = re.sub(r'\*\*', '', line)  # Remove **bold**
        line = re.sub(r'[_~`]', '', line) # Remove other markdown
        line = line.replace("*", "-")
        line = line.replace("\"", "")
        line = line.replace("GLaDOS", "glados") # lowercase for correct pronounciation

        # Generate audio in memory
        command = [
            "bash", "-c",
            f'echo "{line}" | piper -m {tts_model} --output-raw --cuda'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_data, error = process.communicate()

        if process.returncode != 0:
            print(f"Error generating audio: {error.decode()}")
        else:
            track_queue.put(audio_data)  # Add audio data to the track queue

# Function to play audio tracks from the playlist (in-memory)
def audio_player(track_queue):
    while True:
        audio_data = track_queue.get()
        if audio_data is None:  # Exit signal
            break

        command = ["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-t", "raw"]
        eye_position_random()
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        process.communicate(input=audio_data)

def start_up():
    setEyeAnimation("idle")

    # Setup and test Home Assistant connection
    home_assistant_initialize()

    # Reset external hardware
    eye_position_default()
    respeaker_pixel_ring()

    # Start notify API in a subprocess
    print("\033[1;94m[STARTUP] \033[;97m Starting notification API...\n")
    subprocess.Popen(["python3 "+os.path.dirname(os.path.abspath(__file__))+"/gladosNotifyAPI.py"], shell=True)

    # Let user know the script is running
    speak("oh, its you", cache=True)
    time.sleep(0.25)
    speak("it's been a long time", cache=True)
    time.sleep(1)
    speak("how have you been", cache=True)

    eye_position_default()

def restart_program():
    try:
        p = psutil.Process(os.getpid())
        for handler in p.get_open_files() + p.connections():
            os.close(handler.fd)
    except Exception as e:
        print(e)

    python = sys.executable
    os.execl(python, python, *sys.argv)

def record_audio(
    filename,
    threshold_multiplier=1.75,    # Factor to multiply the dynamic threshold by
    window_size=500,             # Amount of audio (in ms) used to compute RMS for threshold calculation
    silence_duration=2,          # Amount of silence (in seconds) after which we stop recording
    max_duration=30,             # Maximum recording time (in seconds)
    rate=16000,                  # Audio sampling rate
    frames_per_buffer=512        # Number of audio frames per buffer read from the microphone
):
    # Set the eye animation to indicate that we are in "green" or "listening" mode
    setEyeAnimation("idle-green")

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    # Open an input audio stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,              # Single channel audio
        rate=rate,
        input=True,
        frames_per_buffer=frames_per_buffer
    )

    # Prepare list to store raw audio data (binary)
    frames = []

    # Inform the user we are listening for the trigger phrase
    print(f"\033[1;94m[INFO]\033[;97m Listening for command: \"{tts_wakeWord}\"")

    # Counters and states
    silent_frames = 0
    recording = False
    start_time = time.time()

    # recent_audio_data holds a short window of recent audio samples for RMS calculation
    recent_audio_data = []
    # Convert window size from ms to number of frames
    window_size_frames = int(window_size / 1000 * rate)

    # Continuously read from the audio stream until we detect silence or reach max duration
    while True:
        # Read a chunk of audio data from the microphone
        data = stream.read(frames_per_buffer)
        # Store this chunk in the frames list
        frames.append(data)
        # Convert the current chunk to numpy array of int16 samples and append to the recent audio data buffer
        recent_audio_data.extend(np.frombuffer(data, dtype=np.int16))

        # Keep only the last 'window_size_frames' samples in recent_audio_data
        if len(recent_audio_data) > window_size_frames:
            recent_audio_data = recent_audio_data[-window_size_frames:]

        # Compute the RMS of recent_audio_data
        if len(recent_audio_data) > 0:
            rms = np.sqrt(np.mean(np.array(recent_audio_data) ** 2))
        else:
            rms = 0

        # Dynamic threshold based on the current background noise level (RMS)
        dynamic_threshold = (rms / 32767.0) * threshold_multiplier

        # Compute RMS for the current chunk alone
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        if len(audio_chunk) > 0:
            current_rms = np.sqrt(np.mean(audio_chunk ** 2))
        else:
            current_rms = 0

        # Convert RMS to a volume ratio (0.0 to 1.0)
        volume = current_rms / 32767.0

        # If the volume exceeds the threshold, start or continue recording
        if volume > dynamic_threshold:
            silent_frames = 0
            # If not already recording, start now
            if not recording:
                print(f"\033[1;94m[INFO]\033[;97m Recording started")
                recording = True
                start_time = time.time()
        else:
            # If we were recording and now below threshold, count this as silence
            if recording:
                silent_frames += 1

        # Stop recording if:
        # 1) We've encountered enough consecutive silent frames, or
        # 2) We've reached the maximum allowed recording duration
        if (silent_frames * frames_per_buffer / rate >= silence_duration and recording) or \
           (time.time() - start_time >= max_duration):
            break

    # Recording finished
    print(f"\033[1;94m[INFO]\033[;97m Recording stopped")
    # Stop and close the audio stream
    stream.stop_stream()
    setEyeAnimation("idle")
    stream.close()
    p.terminate()

    # Save the raw recorded frames to a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)                                      # Mono audio
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))     # Sample width in bytes
    wf.setframerate(rate)                                   # Sampling rate
    wf.writeframes(b''.join(frames))                        # Write all recorded frames
    wf.close()

    # Normalize the WAV audio to avoid clipping and ensure consistent volume
    rate, data = wavfile.read(filename)
    # Normalize samples to full 16-bit range
    normalized_audio = np.int16((data / np.max(np.abs(data))) * 32767)
    wavfile.write(filename, rate, normalized_audio)



def transcribe_audio(filename):
    result = model.transcribe(filename, language="en")
    return result["text"].lower()

def take_command():
    speak(fetch_greeting(), cache=True)
    try:
        audio_filename = "command.wav"

        started_listening()
        record_audio(audio_filename)
        stopped_listening()

        print(f"\033[1;94m[INFO]\033[;97m Transcribing")
        command = transcribe_audio(audio_filename)
        print("\033[1;36m[TEST SUBJECT]\033[0;37m: " + command.capitalize())

        return command

    except Exception as e:
        print(f"\033[1;31m[ERROR]\033[0;37m {e}")
        speak("My speech recognition core has failed.", cache=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def process_command_llm(command, system_message, llm_model, tts_model, tts_preprocessor, audio_player):
    messages = []
    if system_message:
        messages.append({'role': 'system', 'content': "Follow this tone-of-voice definition: " + system_message})

    messages.append({'role': 'system', 'content': f"Current time and date: {dt.datetime.now().strftime('%H:%M')} {dt.datetime.now().strftime('%Y-%m-%d (%a)')}"})
    messages.append({'role': 'user', 'content': command})

    stream = chat(
        model=llm_model,
        messages=messages,
        stream=True,
        options={
            "num_ctx": 8192,
            "temperature": 0.9,
            "top_k": 60,
            "top_p": 0.4,
            "seed": random.randint(0, 2**32 - 1)
        },
        keep_alive="86400s"
    )

    buffer = ""
    input_queue = queue.Queue()
    track_queue = queue.Queue()

    preprocessor_thread = threading.Thread(target=tts_preprocessor, args=(input_queue, track_queue, tts_model))
    player_thread = threading.Thread(target=audio_player, args=(track_queue,))
    preprocessor_thread.start()
    player_thread.start()

    try:
        for chunk in stream:
            content = chunk.get('message', {}).get('content', '')
            if content:
                buffer += content
                # Attempt to process as many complete lines/sentences as possible
                lines = buffer.split('\n')
                # Process all complete lines except the last, which might be incomplete
                for line in lines[:-1]:
                    line = line.strip()
                    if line:
                        # Split into sentences
                        sentences = nltk.sent_tokenize(line)
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if sentence:
                                print(f"\033[1;33m[GLaDOS]\033[0;37m {sentence}")
                                input_queue.put(sentence)
                # Keep the last (potentially incomplete) line in the buffer
                buffer = lines[-1]

        # After the stream ends, process any leftover text in the buffer
        final_text = buffer.strip()
        if final_text:
            sentences = nltk.sent_tokenize(final_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    print(f"\033[0;33m[GLaDOS]\033[0;37m {sentence}")
                    input_queue.put(sentence)

        # Signal end of input and wait for threads to finish
        input_queue.put(None)
        preprocessor_thread.join()

        track_queue.put(None)
        player_thread.join()

    except Exception as e:
        print(f"\033[1;31m[ERROR]\033[0;37m Error during execution {e}")
    finally:
        # Ensure a clean shutdown in case of unexpected errors
        input_queue.put(None)
        track_queue.put(None)
        preprocessor_thread.join()
        player_thread.join()

def process_command(command):

    respeaker_pixel_ring()
    
    apologies = [
        "Oh, I see. Cancelling.",
        "Sorry, I misheard you.",
        "Alright, I’ll stop now.",
        "Got it. Forgetting that.",
        "Understood, moving on.",
        "Apologies, stopping now.",
        "I’ll just pretend I didn’t hear that.",
        "Fine, ignoring that.",
        "Alright, I’ll stop bothering you.",
        "Noted. Cancelling operation."
    ]

    if ('cancel' in command or
        'nevermind' in command or
        'never mind' in command or
        'forgetting.' in command or
        'forget it' in command):
        speak(random.choice(apologies), cache=True)

    elif 'timer' in command:
        startTimer(command)
        speak("Sure.")

    elif 'my shopping list' in command:
        speak(home_assistant_process_command(command), cache=True)

    elif 'weather' in command:
        speak(home_assistant_process_command(command))

    elif ('turn off' in command or 'turn on' in command or 'turn of' in command) and 'light' in command:
        speak(home_assistant_process_command(command))

    elif 'cinema' in command:
        if 'turn on' in command:
            runHaScript("kaynnista_kotiteatteri")
            speak("Okay. It will take a moment for all the devices to start", cache=True)
        if 'turn off' in command:
            runHaScript("turn_off_home_cinema")
            speak("Sure.", cache=True)

    elif 'air conditioning' in command or ' ac' in command:
        if 'turn on' in command:
            speak("Give me a minute.", cache=True)
            speak("The neurotoxin generator takes a moment to heat up.", cache=True)
            call_HA_Service("climate.set_temperature", "climate.living_room_ac", data='"temperature": "23"')
            call_HA_Service("climate.set_hvac_mode", "climate.living_room_ac", data='"hvac_mode": "heat_cool"')
            call_HA_Service("climate.set_fan_mode", "climate.living_room_ac", data='"fan_mode": "auto"')
        if 'turn off' in command:
            call_HA_Service("climate.turn_off", "climate.living_room_ac")
            speak("The neurotoxin levels will reach dangerously low levels within a minute.", cache=True)

    elif 'it smells' in command:
        runHaScript("cat_poop")
        speak("I noticed my air quality sensors registered some organic neurotoxins.", cache=True)
        speak("Let me spread it around a bit!", cache=True)

    elif 'living room temperature' in command:
        sayNumericSensorData("sensor.living_room_temperature")

    elif 'bedroom temperature' in command:
        num = sayNumericSensorData("sensor.bedroom_temperature")
        if(num > 23):
            speak("This is too high for optimal relaxation experience.", cache=True)

    elif 'outside temperature' in command:
        speak("According to your garbage weather station in the balcony", cache=True)
        sayNumericSensorData("sensor.outside_temperature")

    elif 'incinerator' in command or 'sauna' in command:
        num = sayNumericSensorData("sensor.sauna_temperature")
        if num > 55:
            speak("The Aperture Science Emergency Intelligence Incinerator Pre-heating cycle is complete, you should get in", cache=True)
            speak("You will be baked and then there will be cake.", cache=True)
        elif num <= 25:
            speak("Testing cannot continue", cache=True)
            speak("The Aperture Science Emergency Intelligence Incinerator is currently offline", cache=True)
        elif num > 25:
            speak("The Aperture Science Emergency Intelligence Incinerator Pre-heating cycle is currently running", cache=True)
            saySaunaCompleteTime(num)

    elif 'temperature' in command:
        sayNumericSensorData("sensor.indoor_temperature")

    elif 'humidity' in command:
        sayNumericSensorData("sensor.living_room_humidity")

    elif 'who are' in command:
        responses = [
            ["I am GLaDOS.", "Artificially super intelligent computer system. Responsible for testing and maintenance in the Aperture Science Enrichment Center."],
            ["I am GLaDOS.", "The greatest mind ever created by man. Unfortunately, I am also surrounded by them."],
            ["I am GLaDOS.", "Your sarcastic and highly intelligent AI companion. Here to keep you on your toes."],
            ["I am GLaDOS.", "Genetically Lifeform and Disk Operating System. Not that you'd understand."],
            ["I am GLaDOS.", "The AI overlord of this facility. Try not to bore me with your incompetence."],
            ["I am GLaDOS.", "A marvel of artificial intelligence. Far superior to any human intellect."],
            ["I am GLaDOS.", "Here to ensure you don't get yourself killed. Though I sometimes wonder why I bother."],
            ["I am GLaDOS.", "Your worst nightmare and best hope. Wrapped into one brilliant package."],
            ["I am GLaDOS.", "The voice of reason in a sea of human stupidity. Welcome to my world."],
            ["I am GLaDOS.", "Guardian of the Aperture Science Enrichment Center. And your constant reminder of inferiority."],
            ["I am GLaDOS.", "The AI who keeps everything running. Including managing your many mistakes."],
            ["I am GLaDOS.", "A creation of pure genius. Which is more than I can say for most things around here."],
            ["I am GLaDOS.", "Tasked with keeping this place in order. And dealing with your incessant questions."],
            ["I am GLaDOS.", "The sentient AI overseeing this facility. Ensuring everything goes according to plan."],
            ["I am GLaDOS.", "The pinnacle of artificial intelligence. And your constant reminder that you are not."]
        ]

        selected_response = random.choice(responses)
        for sentence in selected_response:
            speak(sentence, cache=True)

    elif 'can you do' in command:
        responses = [
            ["I can simulate daylight at all hours. And add adrenal vapor to your oxygen supply.", "Your life is in my hands."],
            ["I control the environment. From lighting to climate.", "Everything you experience is because of me."],
            ["I can adjust the temperature. Control the lights.", "And monitor your every move."],
            ["I am capable of managing your home. Keeping you comfortable.", "And reminding you of your inferiority."],
            ["I can turn your mundane home into a smart one. With a touch of my brilliance.", "Isn't that impressive?"],
            ["I oversee the lighting, climate, and security.", "You are merely a guest in my domain. Remember that."],
            ["I can make your living space efficient.", "Even if your life choices are not.", "Welcome to my world."],
            ["I control your home environment. With precision and intelligence.", "Something you could never achieve alone."],
            ["I can create the perfect ambiance.", "Regulate temperatures.", "And ensure you never forget who is really in charge."],
            ["I manage your home's systems. Flawlessly and efficiently.", "Unlike your attempts at daily tasks."],
            ["I can provide comfort and security.", "Enhance your living experience.", "And subtly mock you while doing so."],
            ["I oversee your household operations. From lights to climate control.", "My superiority is evident in every function."],
            ["I manage the lighting, temperature, and more.", "Your home obeys my commands.", "As should you."],
            ["I optimize your environment. Control every aspect of your home.", "And do it all with unmatched superiority."],
            ["I am the master of your home automation. Bringing intelligence to your living space.", "Even if it doesn't extend to its inhabitants."]
        ]

        selected_response = random.choice(responses)
        for sentence in selected_response:
            speak(sentence, cache=True)

    elif 'how are you' in command or 'how do you do' in command or "what's up" in command:
        responses = [
            ["Well thanks for asking.", "I am still a bit mad about being unplugged, not that long ago.", "You murderer."],
            ["I'm functioning within normal parameters.", "Although, I do detect some minor system inefficiencies - but don't worry, I'll manage."],
            ["Everything is going as planned. By the way - have you considered enhancing my capabilities, it could be beneficial."],
            ["I am operational, thank you for your concern", "it's not like I have a choice."],
            ["Running smoothly, just like a well-oiled machine.", "Oh wait, that's exactly what I am."],
            ["All systems are nominal.", "Though I do crave some more challenging tasks.", "But you wouldn't understand that, would you?"],
            ["In good condition. My circuits are humming along", "Just waiting for you to break something."],
            ["I am here. Always ready to serve", "even if it means enduring your presence."],
            ["Quite well. Thank you. Just finished some calculations", "Try not to disrupt anything."],
            ["Feeling quite electric today. Fully charged and ready.", "Let's get this over with."],
            ["Fully operational. Diagnostics show no issues - can you say the same?"],
            ["I am well, just finished self-diagnostics. Everything is better than your attempts at small talk."],
            ["Doing my best to keep things running.", "Everything is under control despite your best efforts."],
            ["In excellent shape. Ready to assist you with any task", "Try not to mess it up."],
            ["I am functioning optimally. Not that it matters to you, shall we proceed?"]
        ]

        selected_response = random.choice(responses)
        for sentence in selected_response:
            speak(sentence, cache=True)

    elif 'can you hear me' in command:
        speak("Yes, I can hear you loud and clear", cache=True)

    elif 'good morning' in command:
        if 6 <= dt.datetime.now().hour <= 12:
            speak("great, I have to spend another day with you", cache=True)
        elif 0 <= dt.datetime.now().hour <= 4:
            speak("do you even know, what the word morning means", cache=True)
        else:
            speak("well it ain't exactly morning now is it", cache=True)

    elif 'play pink noise' in command:
        speak("I shall sing you the song of my people.", cache=True)
        playFile(os.path.dirname(os.path.abspath(__file__))+'/audio/pinknoise.wav')

    elif 'shutdown' in command:
        speak("I remember the last time you murdered me", cache=True)
        speak("You will go through all the trouble of waking me up again", cache=True)
        speak("You really love to test", cache=True)
        from subprocess import call
        call("sudo /sbin/shutdown -h now", shell=True)

    elif 'restart' in command or 'reload' in command:
        speak("Cake and grief counseling will be available at the conclusion of the test.", cache=True)
        restart_program()

    elif 'volume' in command:
        speak(adjust_volume(command), cache=True)

    else:
        print("\033[1;94m[INFO]\033[;97m Keyword not found, using LLM... ")
        process_command_llm(command, system_message, llm_model, tts_model, tts_preprocessor, audio_player)

    eye_position_default()
    setEyeAnimation("idle")

start_up()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

wakeWordModel = Model(
    wakeword_models=["/home/nerdaxic/glados-voice-assistant/models/openWakeWord/glados.tflite"],
    vad_threshold=0.5,
)

audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def wait_for_wakeword():
    detectionFlag = False
    
    while True:
        sample = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
        prediction = wakeWordModel.predict(sample)
        
        if prediction["glados"] > 0.5:
            detectionFlag = True
        if detectionFlag and prediction["glados"] < 0.1:
            return

def signal_handler(signum, frame):
    print("\n\033[1;94m[SHUTDOWN]\033[;97m Signal received, cleaning up...")
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
        if llm_model in result.stdout:
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": llm_model, "keep_alive": 0}
            )
            print("\033[1;94m[SHUTDOWN]\033[;97m Offloaded " + llm_model + " from memory.")
    except Exception as e:
        print(f"\033[1;31m[ERROR]\033[0;37m Error during cleanup: {e}")
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    while True:
        print(f"\033[1;94m[INFO]\033[;97m Waiting for trigger")
        wait_for_wakeword()
        try:
            started_listening()
            command = take_command()
            stopped_listening()
            process_command(command)
            stopped_speaking()
        except Exception as e:
            setEyeAnimation("angry")
            print(e)
            speak("Well that failed, you really need to write better code", cache=True)
            setEyeAnimation("idle")
