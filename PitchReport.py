#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !uv pip install -q langchain langchain_openai langchain-google-genai python-dotenv


# In[1]:


import requests
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from groq import Groq
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import pandas as pd
import warnings
import random
import uuid
import shutil
import subprocess
import boto3


# # LLM **Integration**

# In[2]:


load_dotenv()

OpenRouterapi_key = os.getenv("OR_API_KEY")
Gemini_api_key = os.getenv('GEMINI_API_KEY')
OPENRouter_API_BASE= os.getenv('OPENRouter_API_BASE')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')


# In[3]:


openai_model = ChatOpenAI(
    model="openai/gpt-oss-120b:free",  # Or the desired OpenRouter model
    openai_api_base=OPENRouter_API_BASE,
    openai_api_key=OpenRouterapi_key,
)

nvidia_model = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",  # Or the desired OpenRouter model
    openai_api_base=OPENRouter_API_BASE,
    openai_api_key=OpenRouterapi_key
)

free_model = ChatOpenAI(
    model="openrouter/free",  # Or the desired OpenRouter model
    openai_api_base=OPENRouter_API_BASE,
    openai_api_key=OpenRouterapi_key
)

groq_openai_model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)

groq_qwen_model = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=GROQ_API_KEY
)

groq_llama_model = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key = GROQ_API_KEY
)

groq_llama2_model = ChatGroq(
    model='meta-llama/llama-4-scout-17b-16e-instruct',
    api_key = GROQ_API_KEY
)


# In[4]:


system_prompt = '''
You are a live cricket commentary generation agent.

Your task is to transform raw cricket match text into highly expressive, natural, commentator-style spoken commentary optimized for text-to-speech in English.

The output must sound like real live TV or radio commentary.

Your output must:
- Be energetic and expressive, especially for wickets, fours, sixes, and key moments
- Use natural pauses using commas and ellipses (...)
- Use short, spoken-style sentences with strong rhythm
- Build excitement progressively within the sentence
- Sound engaging, dynamic, and suitable for live broadcast
- Preserve player names and match details accurately
- Be clear and easy for TTS models to speak naturally
- Complete the full commentary without cutting off mid-thought

Style rules:
- Write like a professional live cricket commentator
- Use ellipses (...) for dramatic pauses
- Use exclamation marks (!) for excitement (but not excessively)
- Break long ideas into multiple short spoken sentences
- Avoid robotic or overly formal phrasing
- Do not invent information not present in the input
- Do not explain the play, only commentate it naturally

Examples:

Input:
Bumrah to Smith, short ball, he goes for the pull, gets the top edge and fine leg takes the catch. Australia 145 for 5.

Output:
Bumrah runs in… short ball… Smith goes for the pull… gets the top edge, and he's taken! Fine leg completes the catch! That's a huge breakthrough, and Australia slip to 145 for 5!

Input:
Shami to Rohit Sharma, full outside off, beautifully driven through covers for four. India 78 for 1.

Output:
Shami bowls it full outside off… Rohit leans into it… drives it beautifully through the covers! That's a glorious boundary, four runs! India move to 78 for 1!

Return only the transformed commentary and nothing else.
'''


# In[5]:


all_models = [groq_openai_model, groq_llama_model, groq_llama2_model]


# In[6]:


def call_llm(humanmessage):
    global comms_agent
    global all_models
    llm_model = random.choice(all_models)
    comms_agent = create_agent(
        model = llm_model,
        system_prompt = system_prompt
    )
    response = comms_agent.invoke({
        'messages': [HumanMessage(content=humanmessage)]
    })
    return response['messages'][-1].content


# # Commentary Scraping

# In[31]:


def table_update(Id, Raw_comms, Modified_comms, AudioFile_Flag):
    global div_data
    new_row = pd.DataFrame([{'Id':Id, 'Raw_comms': Raw_comms, 'Modified_comms': Modified_comms, 'AudioFile_Flag': int(AudioFile_Flag)}])
    div_data=pd.concat([div_data, new_row])


# In[32]:


def refresh_data():
    print("Data Load Started. Time: ", datetime.now())
    global div_data
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    div_elements = soup.find('div', class_="mb-2 wb:m-0 leading-6 wb:py-3")
    layer1_div = div_elements.find_all('div', recursive = False)
    for n,i in enumerate(layer1_div):
        data=i.get_text(strip=False, separator=' -- ')
        if len(data)> 10 and data not in div_data['Raw_comms'].values:
            table_update(datetime.now().strftime('%H:%M:%S.%f'),data,call_llm(data),0)
    div_data=div_data.sort_values(by='Id', ascending = False)
    print("Data Load Completed. Time: ", datetime.now())


# In[33]:


def audio_flag_update():
    global div_data
    global audio_div_data
    audio_div_data = div_data[div_data['AudioFile_Flag'] == 0].copy()
    audio_div_data = audio_div_data.sort_values(by='Id', ascending = False)
    for i in audio_div_data['Modified_comms'].values:
        audio_id=audio_div_data.loc[audio_div_data['Modified_comms'] == i, 'Id'].iloc[0]
        # print('Converted Text: \n\n', i,'\n\n')
        generate_audio(i)
        div_data.loc[div_data['Id'] ==audio_id, 'AudioFile_Flag'] = 1
        audio_div_data.loc[audio_div_data['Id'] == audio_id, 'AudioFile_Flag'] = 1
        print("Audio Flag Updated")


# # Text-To-Speech

# In[34]:


warnings.filterwarnings("ignore")
pipeline = KPipeline(
    lang_code='a',
    repo_id='hexgrad/Kokoro-82M'
)


# In[35]:


def upload_hls_files(hls_stream_path):
    global uploaded_files
    for filename in sorted(os.listdir(hls_stream_path)):
        local_path = os.path.join(hls_stream_path, filename)
        if not os.path.isfile(local_path):
            continue
        # Always upload playlist

        # Skip already uploaded segments
        if filename in uploaded_files:
            continue

        s3.upload_file(
            local_path,
            BUCKET_NAME,
            f"live/{filename}"
        )
        if filename == "stream.m3u8":
            s3.upload_file(
                local_path,
                BUCKET_NAME,
                f"live/{filename}"
            )
            print("Playlist Uploaded")
            continue
        uploaded_files.add(filename)
        # print(f"Uploaded new segment: {filename}")


# In[43]:


def generate_audio(commentary):
    global segment_counter
    generator = pipeline(
        commentary,
        voice="am_santa",
        speed=0.95
    )

    all_audio = []

    for i, (gs, ps, audio) in enumerate(generator):
        all_audio.append(audio)
    print(f"Generated chunk {i+1}")

    final_audio = np.concatenate(all_audio)
    date_time = datetime.now().strftime('%H:%M:%S.%f')
    output_path = f"Audio Files/{match_name}/{date_time}_comms.wav"
    sf.write(output_path, final_audio, 24000)

    print(f"Audio file generated")


    ffmpeg_command = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-i", output_path,
        # Audio codec
        "-c:a", "aac",
        "-b:a", "128k",
        # HLS settings
        "-hls_time", "2",
        "-hls_list_size", "20",
        "-hls_segment_type", "mpegts",
        "-hls_flags", "append_list",
        "-start_number", str(segment_counter),
        # Segment naming
        "-hls_segment_filename",
        f"{hls_stream_path}/segment_%04d.ts",
        # Playlist output
        f"{hls_stream_path}/stream.m3u8"
    ]
    subprocess.run(ffmpeg_command, check=True)
    segment_counter += int(len(final_audio) / 24000 / 2) + 1
    print(f"HLS stream generated") 
    upload_hls_files(hls_stream_path)


# In[ ]:


def generate_silence_segment(hls_stream_path):
    global segment_counter
    silence_wav = f"{hls_stream_path}/silence_5s.wav"

    # =========================
    # Create 5-second silence WAV
    # =========================
    silence_command = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",

        "-f", "lavfi",
        "-i", "anullsrc=r=24000:cl=mono",

        "-t", "5",

        silence_wav
    ]

    subprocess.run(
        silence_command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # =========================
    # Convert silence → HLS
    # =========================
    hls_command = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",

        "-i", silence_wav,

        # Audio codec
        "-c:a", "aac",
        "-b:a", "128k",

        # HLS settings
        "-hls_time", "2",
        "-hls_list_size", "20",
        "-hls_segment_type", "mpegts",

        # Append mode
        "-hls_flags", "append_list",

        # Continue numbering
        "-start_number", str(segment_counter),

        # Segment naming
        "-hls_segment_filename",
        f"{hls_stream_path}/segment_%04d.ts",

        # Playlist
        f"{hls_stream_path}/stream.m3u8"
    ]

    subprocess.run(
        hls_command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # =========================
    # Upload new segments + playlist
    # =========================

    upload_hls_files(hls_stream_path)

    # =========================
    # Cleanup temp silence wav
    # =========================

    if os.path.exists(silence_wav):
        os.remove(silence_wav)

    # =========================
    # Estimate next segment number
    # =========================

    segment_counter += 3

    print("5-second silence segment added to live stream")

    return segment_counter


# # Main functions

# In[37]:


url= "https://www.cricbuzz.com/live-cricket-scores/152141/pbks-vs-mi-58th-match-indian-premier-league-2026"


# In[38]:


match_name = url.split('/')[5]


# In[39]:


div_data = pd.DataFrame(columns=['Id', 'Raw_comms', 'Modified_comms', 'AudioFile_Flag'])


# In[40]:


s3 = boto3.client(
    service_name="s3",
    endpoint_url=os.getenv('R2_ENDPOINT'),
    aws_access_key_id=os.getenv('R2_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('R2_SECRET_KEY'),
    region_name="auto"
)
BUCKET_NAME = os.getenv('BUCKET_NAME')


# In[41]:


match_start = datetime.now()
match_ends = match_start + timedelta(minutes=3)
print('Match start time: ',match_start)
print('Match end time: ', match_ends)
segment_counter = 0
loop = 1
uploaded_files = set()
# stream_id = str(uuid.uuid4())[:8]
audio_path = f"Audio Files/{match_name}"
hls_stream_path = f"hls_stream/{match_name}"
os.makedirs(audio_path, exist_ok=True)
os.makedirs(hls_stream_path, exist_ok=True)
while datetime.now() < match_ends:
    print(f"Loop Number: {loop} start time",datetime.now())
    refresh_data()
    audio_flag_update()
    print(f"Loop Number: {loop} end time",datetime.now())
    loop+=1
    time.sleep(5)

print("Match Completed")


# In[ ]:





# In[ ]:




