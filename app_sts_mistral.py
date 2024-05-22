#from langchain_community.chat_models import ChatOpenAI
#from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from openai import OpenAI
import streamlit as st
import wave
import time
import pyaudio
import torch
import numpy as np
#from openvoice import se_extractor
import whisper
#import argparse
import openai
from pynput import keyboard
import os
from dotenv import load_dotenv
from openvoice.api import BaseSpeakerTTS #ToneColorConverter
#from audio_recorder_streamlit import audio_recorder
from audiorecorder import audiorecorder
#import pydub

load_dotenv()

## OPENAI STUFF
#headers = {"Authorization": "Bearer " + os.getenv("OPENAI_API_KEY") }
#openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize the OpenAI client with the api key
#client = OpenAI(base_url="http://localhost:8501", api_key=os.getenv("OPENAI_API_KEY"))
#client = OpenAI()

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")
# System instructions for the model
system_instructions1 = "[SYSTEM] Answer as Real Chatgpt 4o', Keep conversation very short, clear, friendly and concise. The text provided is a request for a specific type of response from you, the virtual assistant. You will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. [USER]"
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")


os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# for Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")



# Define the name of the log file
chat_log_filename = "chatbot_conversation_log_sts_mistral.txt"

# Function to play audio using PyAudio
def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')

    # Create a PyAudio instance
    p = pyaudio.PyAudio()

    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

output_parser = StrOutputParser()
# Command line arguments
#parser = argparse.ArgumentParser()
#parser.add_argument("--share", action='store-time',default=False, help="make link public")
#args = parser.parse_args()

# Model and device setup
en_ckpt_base = '/Users/neha/Downloads/langchain/OpenVoice/checkpoints/base_speakers/EN' 
ckpt_converter = '/Users/neha/Downloads/langchain/OpenVoice/checkpoints/converter'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
output_dir = 'outputs_sts'
os.makedirs(output_dir, exist_ok=True)


# Load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
#tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', devide=device)
#tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
sampling_rate = en_base_speaker_tts.hps.data.sampling_rate
mark = en_base_speaker_tts.language_marks.get("english", None)

asr_model = whisper.load_model("base.en")

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)

# Main processing function
def process_and_play(prompt, style, audio_file_path):
    tts_model = en_base_speaker_tts
    
    # Process text and generate audio
    try:
        src_path = f'{output_dir}/tmp.wav'
        tts_model.tts(prompt, src_path, speaker=style, language='English')

        save_path = f'{output_dir}/tmp.wav'
        print("Audio generated successfully.")
        play_audio(save_path)

    except Exception as e:
        print(f"Error during audio generation: {e}")

def chatgpt_streamed(user_input, system_message, conversation_history, bot_name):
    """
    Function to send a query to OpenAI's GPT-3.5-Turbo model, stream the response and print the full line in yellow colour.
    Logs the conversation to a file.
    """
    messages = [{"role":"system", "content": system_message}] + conversation_history +[{"role":"user","content":user_input}]
    
    generate_kwargs = dict(
        temperature=0.7,
        max_new_tokens=512,
        top_p=0.95,
        repetition_penalty=1,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = system_instructions1 + user_input + "[OpenGPT 4o]"

    streamed_completion = client.text_generation(
        formatted_prompt,
        stream=True,
        **generate_kwargs,
        details=True, 
        return_full_text=False
    )
    output = ""
    for response in streamed_completion:
        if not response.token.text == "</s>":
            output += response.token.text

    return output
"""
    #inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    #outputs = model.generate(inputs, max_new_tokens=20)
    #print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    full_response = ""
    line_buffer = ""

    with open(chat_log_filename, "a") as log_file: # Open the log file in append mode
        for chunk in streamed_completion:
            delta_content = chunk.choices[0].delta.content

            if delta_content is not None:
                line_buffer += delta_content

                if '\n' in line_buffer:
                    lines = line_buffer.split('\n')
                    for line in lines[:-1]:
                        print(line)
                        full_response += line + '\n'
                        st.write(f"{bot_name}: {line}\n") # Log the line with the bot's name
                        log_file.write(f"{bot_name}: {line}\n") # Log the line with the bot's name
                    line_buffer = lines[-1]

        if line_buffer:
            print(line_buffer)
            full_response += line_buffer
            log_file.write(f"{bot_name}: {line_buffer}\n")
            st.write(f"{bot_name}: {line_buffer}\n")

    return full_response
"""
# convert user audio input to text
def st_audio_to_text():
    st.title("Audio Recorder")
    audio = audiorecorder("Click to record", "Click to stop recording")
    if len(audio) > 0:
        audio.export(out_f="/Users/neha/Downloads/langchain/OpenVoice/outputs_sts/audio.wav", format="wav")

        data = "/Users/neha/Downloads/langchain/OpenVoice/outputs_sts/audio.wav"
        #audio_file = open(data, "rb")
        #transcript = openai.Audio.transcribe("whisper-1", audio_file)
        result = asr_model.transcribe(data, fp16 = False)['text']
        st.write("You: "+ result)
        return result

# New function to handle a conversation with a user
def conversation():
    system_message = "You are a helpful assistant. Please respond to the user queries. Make the responses short and concise."
    #conversation_history = [{'role': 'system', 'content': system_message}]
    conversation_history = []
    #st.write("Press any key and start speaking")
    #user_input = ""
    user_input = st_audio_to_text()
    flag = True
    while flag:
        
        if user_input:
            if user_input.lower() == "exit": # Say 'exit' to end the conversation
                flag = False
                break 

            conversation_history.append({'role': 'user', 'content': user_input})

            #response = client.chat.completions.create(model="local-model", messages=conversation_history)
            chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot")
            #chatbot_response = response.choices[0].message.content
            conversation_history.append({'role': 'assistant', 'content': chatbot_response})
            #st.write(conversation_history)
            process_and_play(chatbot_response, "default", "/Users/neha/Downloads/langchain/OpenVoice/outputs/tmp.wav")

            #st.write("Press any key and start speaking")
            user_input = ""
            #user_input = record_and_transcribe_audio()
        else:
            flag=False
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

## streamlit framework
st.title('Speech to Speech Demo with OpenAI API')
conversation()