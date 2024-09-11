import re
import os
import time
import numpy as np
import random
import glob
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from espnet2.bin.tts_inference import Text2Speech
from scipy.io.wavfile import write
from scipy.io import wavfile
from torchaudio.compliance import kaldi
from xvector_jtubespeech import XVector

def extract_xvector(model, wav_path):
    _, wav = wavfile.read(wav_path)
    # extract mfcc
    wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    mfcc = kaldi.mfcc(wav, num_ceps=24, num_mel_bins=24) # [1, T, 24]
    mfcc = mfcc.unsqueeze(0)

    # extract xvector
    xvector = model.vectorize(mfcc) # (1, 512)
    xvector = xvector.to("cpu").detach().numpy().copy()[0]
    return xvector

def preprocess(text):
    text = str(text).replace(" ", "").replace("　", "").strip()
    text = re.sub("[0-10]+", "", text, 1)
    df = pd.read_excel("../../data/clean/train/train_text/用語集_with_pronunciation.xlsx")
    for i, record in df.iterrows():
        if record["Unnamed: 4"] == "y":
            continue
        if str(record["Unnamed: 3"]) == "nan":
            continue
        keyword = str(record["Unnamed: 3"]).split("。")[0]
        pronunce = record["Unnamed: 4"]
        if keyword in text:
            text = text.replace(keyword, pronunce)

    return text.split("。")

model = "./train.total_count.best.pth"
conf = "./config.yaml"
text2speech = Text2Speech.from_pretrained(
    train_config=conf,
    model_file=model,
    device="cuda",
    speed_control_alpha=1.0,
    noise_scale=0.333,
    noise_scale_dur=0.333,
)
texts_path = Path("../../data/clean/train/train_text/")
keyword_df = pd.read_excel(str(texts_path / "用語集_with_pronunciation.xlsx"))

# prepare xvector
xvector_model = torch.hub.load("sarulab-speech/xvector_jtubespeech", "xvector", trust_repo=True)
wav_100 = glob.glob("/home/sakai/espnet/egs2/jvs/tts1/downloads/jvs_ver1/*/nonpara30/wav24kHz16bit/")
xvectors = []
for wav_path in wav_100:
    wav_path = glob.glob(os.path.join(wav_path, "*"))[0]
    xvector = extract_xvector(xvector_model, wav_path)
    xvectors.append(xvector)
mix_xvectors = []
for i in range(0, len(xvectors)):
    for j in range(i+1, len(xvectors)):
        mix_xvector = xvectors[i] / 2 + xvectors[j] / 2
        mix_xvectors.append(mix_xvector)

xvectors += mix_xvectors
with open(str(texts_path / "clean_created_texts_by_manually.txt"), 'r') as f:
    for i, line in tqdm(enumerate(f)):
        if not line:
            continue
        xvector = random.choice(xvectors)
        input = preprocess(line)
        wav_list = []
        for j, sentence in enumerate(input):
            pause = np.zeros(30000, dtype=np.float32)
            try:
                with torch.no_grad():
                    wav = text2speech(text=sentence, spembs=xvector)["wav"]
            except:
                print(sentence)
                continue
            wav_list.append(np.concatenate([wav.view(-1).cpu().numpy(), pause]))
        if len(wav_list) > 1:
            final_wav = np.concatenate(wav_list)
        elif len(wav_list) == 1:
            final_wav = wav_list[0]
        else:
            continue
        write(f"../../data/clean/train/train_audio/{i}.wav", rate=text2speech.fs, data=final_wav)


