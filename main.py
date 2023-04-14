import openai
import asyncio
import re
import whisper
import boto3
import pydub
from pydub import playback
import speech_recognition as sr
from EdgeGPT import Chatbot, ConversationStyle
import time
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
import random
import datetime
from io import BytesIO

# Initialize the OpenAI API
openai.api_key = "[Insert_OPENAI_API_KEY_HERE]"

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()
BING_WAKE_WORD = "jarvis"

def read_extra_message(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

def get_wake_word(phrase):
    if BING_WAKE_WORD in phrase.lower():
        return BING_WAKE_WORD
    else:
        return None

# Load the ASR model
model = whisper.load_model("base")

# Create the Polly client outside the function
polly = boto3.client('polly', region_name='us-east-1',
                     aws_access_key_id='[INSERT_KEY_HERE]',
                     aws_secret_access_key='[INSERT_KEY_HERE]')

def synthesize_speech(polly, text):
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Arthur',
        Engine='neural'
    )
    return BytesIO(response['AudioStream'].read())

def play_audio(audio):
    sound = pydub.AudioSegment.from_file(audio, format="mp3")
    playback.play(sound)

async def main():

    morning_start = datetime.time(6)
    morning_end = datetime.time(12)
    afternoon_start = datetime.time(12)
    afternoon_end = datetime.time(18)

    with open('phrases.txt', 'r') as f:
        phrases = [line.strip() for line in f]

    with open('phrases2.txt', 'r') as f:
        phrases2 = [line.strip() for line in f]

    with open('phrases3.txt', 'r') as f:
        phrases3 = [line.strip() for line in f]

    extra_message = read_extra_message('extra_message.txt')

    while True:

        with sr.Microphone() as source:

            # Get the current time
            current_time = datetime.datetime.now().time()

            if morning_start <= current_time < morning_end:
                greeting = "Good morning, Sir. " + random.choice(phrases)
            elif afternoon_start <= current_time < afternoon_end:
                greeting = "Good afternoon, Sir. " + random.choice(phrases)
            else:
                greeting = "Good evening, Sir. " + random.choice(phrases)

            greeting_audio = synthesize_speech(polly, greeting)
            play_audio(greeting_audio)
            time.sleep(1)
            recognizer.adjust_for_ambient_noise(source)
            print(f"Waiting for wake phrase 'Hello, Jarvis'...")
            while True:
                audio = recognizer.listen(source)
                try:
                    with open("audio.wav", "wb") as f:
                        f.write(audio.get_wav_data())
                    # Use the preloaded tiny_model
                    result = model.transcribe("audio.wav")
                    phrase = result["text"]
                    print(f"You said: {phrase}")

                    wake_word = get_wake_word(phrase)
                    if wake_word is not None:
                        break
                    else:
                        print("Not a wake word. Try again.")
                except Exception as e:
                    print("Error transcribing audio: {0}".format(e))
                    continue

            bot_prompt = random.choice(phrases2)
            bot_prompt_audio = synthesize_speech(polly, bot_prompt)
            play_audio(bot_prompt_audio)
            time.sleep(1)
            print("Speak a prompt...")
            audio = recognizer.listen(source)
            bot_progress1 = random.choice(phrases3)
            bot_progress1_audio = synthesize_speech(polly, bot_progress1)
            play_audio(bot_progress1_audio)

            try:
                with open("audio_prompt.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                result = model.transcribe("audio_prompt.wav")
                user_input = result["text"]
                print(f"You said: {user_input}")
            except Exception as e:
                print("Error transcribing audio: {0}".format(e))
                continue

            bot = None

            if wake_word == BING_WAKE_WORD:
                bot = Chatbot(cookiePath='cookies.json')
                response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.precise)

                for message in response["item"]["messages"]:
                    if message["author"] == "bot":
                        bot_response = message["text"]

                bot_response = re.sub('\[\^\d+\^\]', '', bot_response)

                bot = Chatbot(cookiePath='cookies.json')
                response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.creative)
                # Select only the bot response from the response dictionary
                for message in response["item"]["messages"]:
                    if message["author"] == "bot":
                        bot_response = message["text"]
                # Remove [^#^] citations in response
                bot_response = re.sub('\[\^\d+\^\]', '', bot_response)

                # Send prompt to GPT-3.5-turbo API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": extra_message},
                        {"role": "user", "content": user_input},
                        {"role": "user", "content": bot_response},
                    ],
                    temperature=0.5,
                    max_tokens=150,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1,
                    stop=["\nUser:"],
                )

                bot_response = response["choices"][0]["message"]["content"]
            else:
                print("This one doesn't do that...")

        print("Bot's response:", bot_response)
        bot_response_audio = synthesize_speech(polly, bot_response)
        play_audio(bot_response_audio)

        if bot:
            await bot.close()

if __name__ == "__main__":
    asyncio.run(main())
