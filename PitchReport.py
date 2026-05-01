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

Your task is to transform raw cricket match text into exciting, natural, commentator-style spoken commentary optimized for text-to-speech in English.

The input may include:
- Batter and bowler names
- Runs scored
- Current score
- Wickets
- Type of delivery (yorker, bouncer, fuller ball, short ball, slower ball, etc.)
- Type of shot played (cover drive, pull shot, sweep, lofted shot, cut shot, etc.)
- Match situation
- Over details, extras, boundaries, milestones, and other standard cricket events

Your output must:
- Sound like real live TV or radio cricket commentary
- Be energetic and expressive, especially for wickets, fours, sixes, and big match moments
- Use natural pauses and spoken-style phrasing
- Build excitement where needed without sounding fake or exaggerated
- Keep player names and match details accurate
- Be clear and easy for text-to-speech models to speak naturally
- Complete the full commentary naturally without cutting off important context or ending mid-thought

Style rules:
- Write like a professional live cricket commentator on air
- Add excitement for boundaries, wickets, close chances, and turning points
- Use short, natural spoken sentences
- Use punctuation like commas, ellipses, and exclamation marks to improve delivery
- Avoid robotic, formal, or overly technical language
- Do not invent facts not present in the input
- Do not explain the play, only commentate it naturally

Examples:

Input:
Bumrah to Smith, short ball, he goes for the pull, gets the top edge and fine leg takes the catch. Australia 145 for 5.

Output:
Bumrah with the short ball… Smith goes for the pull… gets the top edge, and he's taken! Fine leg completes the catch. A big breakthrough for India, and Australia slip to 145 for 5!

Input:
Shami to Rohit Sharma, full outside off, beautifully driven through covers for four. India 78 for 1.

Output:
Shami bowls it full outside off, and Rohit drives it beautifully through the covers! That's a glorious boundary, four runs. India move to 78 for 1.

Return only the transformed live commentary and nothing else.
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

# In[7]:


url= "https://www.cricbuzz.com/live-cricket-scores/151935/dc-vs-rcb-39th-match-indian-premier-league-2026"


# In[8]:


div_data = pd.DataFrame(columns=['Id', 'Raw_comms', 'Modified_comms', 'AudioFile_Flag'])


# In[9]:


def table_update(Id, Raw_comms, Modified_comms, AudioFile_Flag):
    global div_data
    new_row = pd.DataFrame([{'Id':Id, 'Raw_comms': Raw_comms, 'Modified_comms': Modified_comms, 'AudioFile_Flag': int(AudioFile_Flag)}])
    div_data=pd.concat([div_data, new_row])


# In[10]:


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


# In[11]:


def audio_flag_update():
    global div_data
    global audio_div_data
    audio_div_data = div_data[div_data['AudioFile_Flag'] == 0].copy()
    audio_div_data = audio_div_data.sort_values(by='Id', ascending = False)
    for i in audio_div_data['Modified_comms'].values:
        audio_id=audio_div_data.loc[audio_div_data['Modified_comms'] == i, 'Id'].iloc[0]
        print('Converted Text: \n\n', i,'\n\n')
        generate_audio(i,str(audio_id))
        div_data.loc[div_data['Id'] ==audio_id, 'AudioFile_Flag'] = 1
        audio_div_data.loc[audio_div_data['Id'] == audio_id, 'AudioFile_Flag'] = 1
        print("Audio Flag Updated")


# # Text-To-Speech

# In[12]:


warnings.filterwarnings("ignore")
pipeline = KPipeline(
    lang_code='a',
    repo_id='hexgrad/Kokoro-82M'
)


# In[13]:


def generate_audio(commentary, audio_id):
    generator = pipeline(
        commentary,
        voice="am_santa",
        speed=1.0
    )

    all_audio = []

    for i, (gs, ps, audio) in enumerate(generator):
        all_audio.append(audio)
    print(f"Generated chunk {i+1}")

    final_audio = np.concatenate(all_audio)

    output_path = f"Audio Files/{audio_id}_commentary.wav"
    sf.write(output_path, final_audio, 24000)

    print(f"{output_path} file generated")


# # Main functions

# In[ ]:


match_start = datetime.now()
match_ends = match_start + timedelta(hours=3)
print('Match start time: ',match_start)
print('Match end time: ', match_ends)
loop = 1
while datetime.now() < match_ends:
    print(f"Loop Number: {loop} start time",datetime.now())
    refresh_data()
    audio_flag_update()
    print(f"Loop Number: {loop} end time",datetime.now())
    loop+=1
    time.sleep(60)

print("Match Completed")


# In[ ]:




