import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit.components.v1 as components
import os
import sys
import cv2
# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import pickle
import tensorflow as tf
import sklearn
from audio_recorder_streamlit import audio_recorder
st.set_page_config(page_title="SER web-app", page_icon=":speech_balloon:", layout="wide")
Y = np.load(r"C:\Users\utkar\OneDrive\Desktop\SER\Y.npy", allow_pickle=True)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(np.array(Y).reshape(-1,1))
scaler = pickle.load(open(r"C:\Users\utkar\OneDrive\Desktop\SER\scaler.pkl",'rb'))
model = keras.models.load_model(r"C:\Users\utkar\OneDrive\Desktop\SER\model.h5")
def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)

def get_mfccs(audio, limit):
    data, sr = librosa.load(audio)
    a = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, :a.shape[1]] = a
    return mfccs

def get_predict_feat(path):
    d,sr=librosa.load(path=path,duration=2.5,offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    if(result.shape != 2376):
        elements_to_add = 2376 - len(result)
        additional_elements = np.random.rand(elements_to_add)
        result = np.append(result, additional_elements)
    result = np.reshape(result,newshape = (1,2376))
    i_result = scaler.transform(result)
    final_result = np.expand_dims(i_result, axis = 2)
    return final_result

emotions1 = {1:'Neutral', 2:'Calm',3:'Happy',4:'Sad',5:'Angry',6:"Fear",7:'Disgust',8:'Surprise'}
def prediction(path1):
    res = get_predict_feat(path1)
    predictions = model.predict(res)
    y_pred = encoder.inverse_transform(predictions)
    return y_pred[0][0]
def save_audio_by_upload(file):
    if file.size > 40000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0
def save_audio_by_record(file):
    # if file.size > 40000000:
    #     return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, 'audio_bytes'), "wb") as f:
        f.write(file)
    return 0
def main():
    st.sidebar.subheader("Menu")
    mood = None
    website_menu = st.sidebar.selectbox("Menu", ("Emotion Recognition", "Project description",
                                                 "Leave feedback", "Relax"))
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if website_menu == "Emotion Recognition":
        st.sidebar.subheader("Model")
        model_type = st.sidebar.selectbox("How would you like to predict?", ("By Uploading an audio file", "By recording your own Voice"))
        if(model_type == "By Uploading an audio file"):
            st.markdown("## Upload the file")
            with st.container():
                col1, col2 = st.columns(2)
                # audio_file = None
                # path = None
                with col1:
                    audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg'])
                    if audio_file is not None:
                        if not os.path.exists("audio"):
                            os.makedirs("audio")
                        path = os.path.join("audio", audio_file.name)
                        if_save_audio = save_audio_by_upload(audio_file)
                        if if_save_audio == 1:
                            st.warning("File size is too large. Try another file.")
                        elif if_save_audio == 0:
                            # extract features
                            # display audio
                            st.audio(audio_file, format='audio/wav', start_time=0)
                            try:
                                wav, sr = librosa.load(path, sr=44100)
                                Xdb = get_melspec(path)[1]
                                mfccs = librosa.feature.mfcc(y=wav, sr=sr)
                                mood = prediction(os.path.abspath(path))
                                # # display audio
                                # st.audio(audio_file, format='audio/wav', start_time=0)
                            except Exception as e:
                                audio_file = None
                                st.error(f"Error {e} - wrong format of the file. Try another .wav file.")
                        else:
                            st.error("Unknown error")
                # with col2:
                #     import matplotlib.pyplot as plt
                #     if audio_file is not None:
                #         fig = plt.figure(figsize=(10, 2))
                #         fig.set_facecolor('#d1d1e0')
                #         plt.title("Wave-form")
                #         librosa.display.waveshow(y = np.array(wav),sr=sr)
                #         plt.gca().axes.get_yaxis().set_visible(False)
                #         plt.gca().axes.get_xaxis().set_visible(False)
                #         plt.gca().axes.spines["right"].set_visible(False)
                #         plt.gca().axes.spines["left"].set_visible(False)
                #         plt.gca().axes.spines["top"].set_visible(False)
                #         plt.gca().axes.spines["bottom"].set_visible(False)
                #         plt.gca().axes.set_facecolor('#d1d1e0')
                #         st.write(fig)
                #     else:
                #         pass
                if audio_file is not None:
                    st.markdown("## Analyzing...")
                    if not audio_file == "test":
                        st.sidebar.subheader("Audio file")
                        file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
                        st.sidebar.write(file_details)

                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = plt.figure(figsize=(10, 2))
                            fig.set_facecolor('#d1d1e0')
                            plt.title("MFCCs")
                            librosa.display.specshow(data = mfccs, sr=sr, x_axis='time')
                            plt.gca().axes.get_yaxis().set_visible(False)
                            plt.gca().axes.spines["right"].set_visible(False)
                            plt.gca().axes.spines["left"].set_visible(False)
                            plt.gca().axes.spines["top"].set_visible(False)
                            st.write(fig)
                        with col2:
                            fig2 = plt.figure(figsize=(10, 2))
                            fig2.set_facecolor('#d1d1e0')
                            plt.title("Mel-log-spectrogram")
                            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
                            plt.gca().axes.get_yaxis().set_visible(False)
                            plt.gca().axes.spines["right"].set_visible(False)
                            plt.gca().axes.spines["left"].set_visible(False)
                            plt.gca().axes.spines["top"].set_visible(False)
                            st.write(fig2)
                st.markdown('## Prediction..')
                if mood:
                    st.success("You're in " + mood + ' mood')
                else:
                    st.warning("Upload the audio file in proper format.")
        else:
            st.markdown("## Record your Voice")
            with st.container():
                col1, col2 = st.columns(2)
                # audio_file = None
                # path = None
                with col1:
                    audio_bytes = audio_recorder()
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                    # The recording will stop automatically
                    # 2 sec after the utterance end
                    if audio_bytes:
                        if not os.path.exists("audio"):
                            os.makedirs("audio")
                        path = os.path.join("audio", 'audio_bytes')
                        if_save_audio = save_audio_by_record(audio_bytes)
                        if if_save_audio == 1:
                            st.warning("File size is too large. Try another file.")
                        elif if_save_audio == 0:
                            # extract features
                            # display audio
                            try:
                                wav, sr = librosa.load(path, sr=44100)
                                Xdb = get_melspec(path)[1]
                                mfccs = librosa.feature.mfcc(y=wav, sr=sr)
                                mood = prediction(os.path.abspath(path))
                                # # display audio
                                # st.audio(audio_file, format='audio/wav', start_time=0)
                            except Exception as e:
                                audio_file = None
                                st.error(f"Error {e} - wrong format of the file. Try another .wav file.")
                                
                # with col2:
                #     if audio_bytes is not None:
                #         fig = plt.figure(figsize=(10, 2))
                #         fig.set_facecolor('#d1d1e0')
                #         plt.title("Wave-form")
                #         #librosa.display.waveshow(y = wav, sr=44100)
                #         plt.gca().axes.get_yaxis().set_visible(False)
                #         plt.gca().axes.get_xaxis().set_visible(False)
                #         plt.gca().axes.spines["right"].set_visible(False)
                #         plt.gca().axes.spines["left"].set_visible(False)
                #         plt.gca().axes.spines["top"].set_visible(False)
                #         plt.gca().axes.spines["bottom"].set_visible(False)
                #         plt.gca().axes.set_facecolor('#d1d1e0')
                #         st.write(fig)
                #     else:
                #         pass
                if audio_bytes is not None:
                    st.markdown("## Analyzing...")
                    if not audio_bytes == "test":
                        st.sidebar.subheader("Audio file")
                        file_details = {"Filename": 'Your Recorded Voice'}
                        st.sidebar.write(file_details)

                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = plt.figure(figsize=(10, 2))
                            fig.set_facecolor('#d1d1e0')
                            plt.title("MFCCs")
                            librosa.display.specshow(data = mfccs, sr=sr, x_axis='time')
                            plt.gca().axes.get_yaxis().set_visible(False)
                            plt.gca().axes.spines["right"].set_visible(False)
                            plt.gca().axes.spines["left"].set_visible(False)
                            plt.gca().axes.spines["top"].set_visible(False)
                            st.write(fig)
                        with col2:
                            fig2 = plt.figure(figsize=(10, 2))
                            fig2.set_facecolor('#d1d1e0')
                            plt.title("Mel-log-spectrogram")
                            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
                            plt.gca().axes.get_yaxis().set_visible(False)
                            plt.gca().axes.spines["right"].set_visible(False)
                            plt.gca().axes.spines["left"].set_visible(False)
                            plt.gca().axes.spines["top"].set_visible(False)
                            st.write(fig2)
                st.markdown('## Prediction..')
                if mood:
                    st.success("You're in " + mood + ' mood')
                else:
                    st.warning("Firstly record something.")
    elif website_menu == "Project description":
        import pandas as pd
        import plotly.express as px
        st.title("Project description")

        st.subheader("Theory")
        link = '[Theory behind - Medium article]' \
               '(https://talbaram3192.medium.com/classifying-emotions-using-audio-recordings-and-python-434e748a95eb)'
        st.markdown(link + ":clap::clap::clap: Tal!", unsafe_allow_html=True)
        with st.expander("See Wikipedia definition"):
            components.iframe("https://en.wikipedia.org/wiki/Emotion_recognition",
                              height=320, scrolling=True)

        st.subheader("Dataset")
        txt = """

            Datasets used in this project
            * Crowd-sourced Emotional Mutimodal Actors Dataset (**Crema-D**)
            * Ryerson Audio-Visual Database of Emotional Speech and Song (**Ravdess**)
            * Surrey Audio-Visual Expressed Emotion (**Savee**)
            * Toronto emotional speech set (**Tess**)    
            """
        st.markdown(txt, unsafe_allow_html=True)

        # df = pd.read_csv(r"C:\Users\utkar\OneDrive\Desktop\SER\speech recognition features\emotion (4).csv")
        # st.dataframe(df)


    

    elif website_menu == "Leave feedback":
        st.subheader("Leave feedback")
        user_input = st.text_area("Your feedback is greatly appreciated")
        user_name = st.selectbox("Choose your personality", ["checker1", "checker2", "checker3", "checker4"])

        if st.button("Submit"):
            st.success(f"Message\n\"\"\"{user_input}\"\"\"\nwas sent")

            if user_input == "log123456" and user_name == "checker4":
                with open("log0.txt", "r", encoding="utf8") as f:
                    st.text(f.read())
            elif user_input == "feedback123456" and user_name == "checker4":
                with open("log.txt", "r", encoding="utf8") as f:
                    st.text(f.read())
            else:
                log_file(user_name + " " + user_input)
                thankimg = Image.open(r"C:\Users\utkar\OneDrive\Desktop\SER\sticky.png")
                st.image(thankimg)

    else:
        import requests
        import json

        url = 'http://api.quotable.io/random'
        if st.button("get random mood"):
            with st.container():
                col1, col2 = st.columns(2)
                n = np.random.randint(1, 1000, 1)[0]
                with col1:
                    quotes = {"Good job and almost done": "checker1",
                              "Great start!!": "checker2",
                              "Please make corrections base on the following observation": "checker3",
                              "DO NOT train with test data": "folk wisdom",
                              "good work, but no docstrings": "checker4",
                              "Well done!": "checker3",
                              "For the sake of reproducibility, I recommend setting the random seed": "checker1"}
                    if n % 5 == 0:
                        a = np.random.choice(list(quotes.keys()), 1)[0]
                        quote, author = a, quotes[a]
                    else:
                        try:
                            r = requests.get(url=url)
                            text = json.loads(r.text)
                            quote, author = text['content'], text['author']
                        except Exception as e:
                            a = np.random.choice(list(quotes.keys()), 1)[0]
                            quote, author = a, quotes[a]
                    st.markdown(f"## *{quote}*")
                    st.markdown(f"### ***{author}***")
                with col2:
                    st.image(image=f"https://picsum.photos/800/600?random={n}")




if __name__ == '__main__':
    main()