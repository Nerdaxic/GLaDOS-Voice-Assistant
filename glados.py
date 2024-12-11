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
#    Works with Home Assistant
#
#    https://github.com/nerdaxic/glados-voice-assistant/
#    https://www.nerdaxic.com/
#
#    Rename settings.env.sample to settings.env
#    Edit settings.env to match your setup
#
# Import basic voice assistant functionality
from gladosTTS import *
from gladosTime import *
from gladosHA import *
from gladosSerial import *
from gladosServo import *
from glados_functions import *

# Import skills
from skills.glados_jokes import *
from skills.glados_magic_8_ball import *
from skills.glados_home_assistant import *

# System functions
import subprocess
import datetime as dt
import os
import random
import psutil

# Local Speech Recognition and Generation
from openwakeword.model import Model     # Wakeword detector
import whisper                            # Speech-to-text
import pyaudio                            # Audio streams and processing
import wave
import numpy as np
from scipy.io import wavfile

# LLM
import threading
import queue
import io
import re
from ollama import chat

# Load settings to variables from setting file
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.dirname(os.path.abspath(__file__))+'/settings.env')

# Initialize whisper model
model = whisper.load_model("large-v3-turbo") #medium is too slow on CPU, small.en is OK, base is too stupid.









# Load system prompt conditionally
system_prompt_path = "/home/nerdaxic/glados-voice-assistant/prompts/llm_glados_tone_of_voice.txt"
system_message = None

if os.path.exists(system_prompt_path) and os.path.getsize(system_prompt_path) > 0:
    with open(system_prompt_path, 'r') as file:
        system_message = file.read().strip()

# Function to preprocess text into audio tracks (in-memory)
def tts_preprocessor(input_queue, track_queue, model_path):
    while True:
        line = input_queue.get()
        if line is None:  # Exit signal
            break

        # Remove formatting/markup from the line
        line = re.sub(r'\*\*', '', line)  # Remove **bold** markup
        line = re.sub(r'[_~`]', '', line)  # Remove other markdown characters
        line = line.replace("*", "-")
        line = line.replace("\"", "")

        # Generate audio in memory
        command = [
            "bash", "-c",
            f'echo "{line}" | piper -m {model_path} --output-raw --cuda'
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

        # Play the audio data directly from memory
        command = ["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-t", "raw"]
        eye_position_random()
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        process.communicate(input=audio_data)











def start_up():

    # Show regular eye-texture, this stops the initial loading animation
    setEyeAnimation("idle")

    # Setup and test Home Assistant connection
    home_assistant_initialize()

    # Reset external hardware
    eye_position_default()
    respeaker_pixel_ring()

    # Start notify API in a subprocess
    print("\033[1;94mINFO:\033[;97m Starting notification API...\n")
    subprocess.Popen(["python3 "+os.path.dirname(os.path.abspath(__file__))+"/gladosNotifyAPI.py"], shell=True)

    # Let user know the script is running
    speak("oh, its you", cache=True)
    time.sleep(0.25)
    speak("it's been a long time", cache=True)
    time.sleep(1)
    speak("how have you been", cache=True)

    print("\nWaiting for keyphrase: "+os.getenv('TRIGGERWORD').capitalize())

    eye_position_default()

# Reload Python script after doing changes to it
def restart_program():
    try:
        p = psutil.Process(os.getpid())
        for handler in p.get_open_files() + p.connections():
            os.close(handler.fd)
    except Exception as e:
        print(e)

    python = sys.executable
    os.execl(python, python, *sys.argv)

def record_audio(filename, threshold_multiplier=1.75, window_size=500, silence_duration=2, max_duration=30, rate=16000,
                 frames_per_buffer=512):
    """Records audio until silence (below a dynamic threshold) is detected.

    Args:
        filename (str): The path to save the recorded audio file.
        threshold_multiplier (float, optional): Multiplier for the running average to set the dynamic threshold.
                                                Higher values mean less sensitive to background noise. Defaults to 1.5.
        window_size (int, optional):  Number of milliseconds to consider for the running average. Defaults to 500.
        silence_duration (float, optional): Duration of silence in seconds to trigger the end of recording. 
                                            Defaults to 1.
        max_duration (float, optional): The maximum recording duration in seconds. Defaults to 60.
        rate (int, optional): The sampling rate for audio recording. Defaults to 44100.
        frames_per_buffer (int, optional): The number of frames per buffer for audio recording. Defaults to 512.
    """

    # Wait for a short period to allow mechanical noise to settle
    setEyeAnimation("idle-green")

    # Setup pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=frames_per_buffer)
    frames = []

    print("Waiting for command...")
    silent_frames = 0
    recording = False
    start_time = time.time()

    recent_audio_data = []
    window_size_frames = int(window_size / 1000 * rate)  # Window size in frames

    while True:
        data = stream.read(frames_per_buffer)
        frames.append(data)
        recent_audio_data.extend(np.frombuffer(data, dtype=np.int16))

        # Keep the recent audio data within the window size
        if len(recent_audio_data) > window_size_frames:
            recent_audio_data = recent_audio_data[-window_size_frames:]

        # Calculate RMS for dynamic threshold
        if len(recent_audio_data) == 0:
            rms = 0
        else:
            rms = np.sqrt(np.mean(np.array(recent_audio_data) ** 2))
        #rms = np.sqrt(np.mean(np.array(recent_audio_data) ** 2))
        dynamic_threshold = rms / 32767.0 * threshold_multiplier

        # Calculate current volume
        current_rms = np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16) ** 2))
        volume = current_rms / 32767.0

        if volume > dynamic_threshold:
            silent_frames = 0
            if not recording:
                print("Recording started...")
                recording = True
                start_time = time.time()
        else:
            if recording:
                silent_frames += 1

        # Stop recording conditions
        if (silent_frames * frames_per_buffer / rate >= silence_duration and recording) or \
                (time.time() - start_time >= max_duration):
            break

    print("Recording stopped.")
    stream.stop_stream()
    setEyeAnimation("idle")
    stream.close()
    p.terminate()

    # Save the recorded audio to a file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Read the recorded audio file
    rate, data = wavfile.read(filename)

    # Normalize the audio
    normalized_audio = np.int16((data / np.max(np.abs(data))) * 32767)
    
    # Save the preprocessed audio
    wavfile.write(filename, rate, normalized_audio)


# Use OpenAI Whisper to do local Speech-to-text
def transcribe_audio(filename):
    result = model.transcribe(filename, language="en")
    return result["text"].lower()

# Listen and process the voice command
def take_command():

    # "Greet" user and let them know the assistant is listening
    speak(fetch_greeting(), cache=True)

    try:
        audio_filename = "command.wav"
        # Tell Home Assistant that recording is going on
        # Think of: Pause Spotify
    
        started_listening()
        record_audio(audio_filename)
        stopped_listening()

        # Speech-to-Text
        # Extract text from the voice recoding
        print("Got it... Transcribing...")
        command = transcribe_audio(audio_filename)
        print("\n\033[1;36mTEST SUBJECT:\033[0;37m: " + command.capitalize() + "\n")
        
        # Remove possible triggerword from the recording
        #if os.getenv('TRIGGERWORD') in command:
        #    command = command.replace(os.getenv('TRIGGERWORD'), '')

        # Pass the voice command
        return command

    except Exception as e:
        print(f"An error occurred: {e}")
        speak("My speech recognition core has failed.", cache=True)


def process_command_llm(command):

    # Construct initial messages for the chat
    messages = []
    if system_message:
        messages.append({'role': 'system', 'content': "Follow this tone-of-voice definition: " + system_message})
    
    messages.append({'role': 'system', 'content': f"Current time and date: {dt.datetime.now().strftime('%H:%M')} {dt.datetime.now().strftime('%Y-%m-%d (%a)')}"})
    messages.append({'role': 'user', 'content': command})
    #print(messages)

    # Initialize the Ollama chat stream
    stream = chat(
        model='tarruda/neuraldaredevil-8b-abliterated:fp16',
        #model='llama3.2',
        messages=messages,
        stream=True,
        options = {
          "num_ctx": 8192,
          "temperature": 0.9,
          "top_k": 60,
          "top_p": 0.4,
          "seed": random.randint(0, 2**32 - 1)  # Random integer within 32-bit unsigned range
        },
        #keep_alive="0s"
        keep_alive="600s"

    )

    buffer = ""  # Buffer to accumulate partial lines
    input_queue = queue.Queue()  # Queue for lines to be processed
    track_queue = queue.Queue()  # Queue for audio tracks
    model_path = "/home/nerdaxic/glados-voice-assistant/models/piper/glados.onnx"

    # Start the preprocessing and playback threads
    preprocessor_thread = threading.Thread(target=tts_preprocessor, args=(input_queue, track_queue, model_path))
    player_thread = threading.Thread(target=audio_player, args=(track_queue,))
    preprocessor_thread.start()
    player_thread.start()

    try:
        for chunk in stream:
            content = chunk.get('message', {}).get('content', '')
            if content:
                buffer += content
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()  # Remove extra whitespace
                    if line:  # Only process non-empty lines
                        print(f"Processing line: {line}")
                        input_queue.put(line)

        # Handle any leftover text in the buffer after the stream ends
        if buffer.strip():  # Check for non-empty leftover text
            print(f"Processing leftover buffer: {buffer.strip()}")
            input_queue.put(buffer.strip())

        # Ensure all TTS items are processed before shutdown
        input_queue.put(None)  # Signal end of input
        preprocessor_thread.join()  # Wait for preprocessing to finish

        track_queue.put(None)  # Signal end of audio playback
        player_thread.join()  # Wait for player to finish
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Ensure clean shutdown in case of unexpected errors
        input_queue.put(None)
        track_queue.put(None)
        preprocessor_thread.join()
        player_thread.join()


# Process the command
def process_command(command):

    # If user cancels the triggering
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

    #elif 'time' in command:
    #    readTime()

    #elif ('should my ' in command or 
    #    'should i ' in command or
    #    'should the ' in command or
    #    'shoot the ' in command):
    #    speak(magic_8_ball(), cache=True)

    #elif 'joke' in command:
    #    speak(fetch_joke(), cache=True)

    elif 'my shopping list' in command:
        speak(home_assistant_process_command(command), cache=True)

    elif 'weather' in command:
        speak(home_assistant_process_command(command))

    ##### LIGHTING CONTROL ###########################

    elif ('turn off' in command or 'turn on' in command or 'turn of' in command) and 'light' in command:
        speak(home_assistant_process_command(command))


    ##### DEVICE CONTROL ##########################
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
                

    ##### SENSOR OUTPUT ###########################

    elif 'living room temperature' in command:
        sayNumericSensorData("sensor.living_room_temperature")

    elif 'bedroom temperature' in command:
        num = sayNumericSensorData("sensor.bedroom_temperature")
        if(num>23):
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
    
    ##### PLEASANTRIES ###########################

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

    ##### Utilities#########################

    # Used to calibrate ALSAMIX EQ 
    elif 'play pink noise' in command:
        speak("I shall sing you the song of my people.", cache=True)
        playFile(os.path.dirname(os.path.abspath(__file__))+'/audio/pinknoise.wav')

    # TODO: Reboot, Turn off
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
    
    ##### FAILED ###########################

    else:
        #setEyeAnimation("angry")
        print("Command not recognized, handing it over to LLM")
        #speak("I have no idea what you meant by that.")
        process_command_llm(command)
        #log_failed_command(command)

    print("\n"+command)
    respeaker_pixel_ring() # Reset respeaker leds
    print("\nWaiting for trigger...")
    eye_position_default()
    setEyeAnimation("idle")

start_up()

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

# Load wakeword detection model
wakeWordModel = Model(
    wakeword_models=["/home/nerdaxic/glados-voice-assistant/models/openWakeWord/glados.tflite"],
    vad_threshold=0.5,
)

# Setup audio for wakeword detection
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)


# Loop where to process mic audio and wait for keyword to be cleared from buffer
def wait_for_wakeword():
    
    detectionFlag = False
    
    # Keep listening for wakeword
    while True:
        sample = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
        prediction = wakeWordModel.predict(sample)
        
        if prediction["glados"] > 0.5:
            # Wakeword was detected
            detectionFlag = True
        if detectionFlag and prediction["glados"] < 0.1:
            # Wakeword is no longer detected
            return

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":

    while True:
        
        wait_for_wakeword()

        try:
            # Listen for command
            started_listening() # Home Assistant trigger
            command = take_command()
            stopped_listening() # Home Assistant trigger
            
            # Execute command
            process_command(command)
            stopped_speaking() # Home Assistant trigger
            
        except Exception as e:
            # Something failed
            setEyeAnimation("angry")
            print(e)
            speak("Well that failed, you really need to write better code", cache=True)
            setEyeAnimation("idle")