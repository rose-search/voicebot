import streamlit as st
from audiorecorder import audiorecorder
import numpy as np
import openai
from gtts import gTTS
import base64
import os
from datetime import datetime

def STT(audio):
    filename = "input.mp3"
    wav_file = open(filename, "wb")
    wav_file.write(audio.tobytes())
    wav_file.close()

    audio_file = open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()
    
    os.remove(filename)
    return transcript["text"]

def ask_gpt(prompt, model):
    response = openai.ChatCompletion.create(model=model, messages=prompt)
    system_message = response["choices"][0]["message"]
    return system_message["content"]

def TTS(response):
    filename = "output.mp3"
    tts = gTTS(text=response, lang="ko")
    tts.save(filename)

    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="True">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    os.remove(filename)

def main():
    st.set_page_config(
        page_title="해린의 보이스봇 프로그램",
        layout = "wide"
    )

    flag_start = False

    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role":"system", "content":"You are a thoughtful assistant. Respond to all input in 25 words and answer in korean"}]
    if "check_audio" not in st.session_state:
        st.session_state["check_audio"] = []

    st.header("해린의 보이스봇 프로그램")

    st.markdown("---")

    with st.expander("보이스봇(voicebot) 프로그램에 관하여", expanded=True):
        st.write(
            """
            - 이해린이 만든 보이스봇 프로그램입니다. 
            - UI는 Streamlit을 활용했습니다.
            - STT(Speach-To-Text)는 OpenAI의 Whisper AI를 활용했습니다.
            - 답변은 OpenAI의 GPT 모델을 활용했습니다.
            - TTS(Text-To-Speach)는 구글의 Google Translate TTS를 활용했습니다.
            - 참고 도서 : 김준성·브라이스 유·안상준, 진짜 챗GPT API 활용법, 위키북스(2023), p.60-107
            """
        )

        st.markdown("")
    
    with st.sidebar:
        openai.api_key = st.text_input(label="OpenAI API Key", placeholder="Enter your API Key", value="", type ="password")
        st.markdown("---")
        
        model = st.radio(label="GPT 모델", options=["gpt-3.5-turbo", "gpt-4(유료 사용자)"])
        st.markdown("---")

        if st.button(label="초기화"):
            st.session_state["chat"] = []
            st.session_state["messages"] = [{"role":"system", "content":"You are a thoughtful assistant. Respond to all input in 25 words and answer in korean"}]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("질문하기")
        audio = audiorecorder("클릭하여 녹음하기", "녹음 중...")
        if len(audio) > 0 and not np.array_equal(audio, st.session_state["check_audio"]):
            st.audio(audio.tobytes())
            question = STT(audio)

            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"] = st.session_state["chat"] + [("user", now, question)]
            st.session_state["messages"] = st.session_state["messages"] + [{"role":"user", "content":question}]
            st.session_state["check_audio"] = audio
            flag_start = True

    with col2:
        st.subheader("질문/답변")
        if flag_start:
            response = ask_gpt(st.session_state["messages"], model)

            st.session_state["messages"] = st.session_state["messages"] + [{"role":"system", "content":response}]
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"] = st.session_state["chat"] + [("bot", now, response)]

            for sender, time, message in st.session_state["chat"]:
                if sender == "user":
                    st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                    st.write("")
                else:
                    st.write(f'<div style="display:felx;align-items:center;justify-content:flex-end;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                    st.write("")
            TTS(response)

if __name__ == "__main__":
    main()