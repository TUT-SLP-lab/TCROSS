import glob
import json
import os
from pydub import AudioSegment
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
import jiwer
import evaluate
import ginza
import spacy
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import (AutoProcessor, Seq2SeqTrainer,
                        Seq2SeqTrainingArguments, WhisperFeatureExtractor,
                        WhisperForConditionalGeneration, WhisperProcessor,
                        WhisperTokenizer)

def after_process(text):
    text = text.replace("。", "").replace("、", "").replace(" ", "").replace("\n", "")

    return text
    
# processor = AutoProcessor.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="Japanese", task="transcribe")
# model = WhisperForConditionalGeneration.from_pretrained("./whisper-large-hi/checkpoint-5000", local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained("./whisper-medium/checkpoint-5000", local_files_only=True)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")
model.config.suppress_tokens = []

dataset = DatasetDict()
test_audio_par = "../../data/clean/eval/short/wavs"
test_audio = []
test_sentences = []
with open("../../data/clean/eval/short/manifest.json", "r") as f:
    for line in f.readlines():
        data = json.loads(line)
        audio_filepath = data["audio_filepath"]
        text = data["text"]
        audio_filepath = os.path.join(test_audio_par, audio_filepath)
        test_audio.append(audio_filepath)
        test_sentences.append(text)
dataset["test"] = Dataset.from_dict({"audio": test_audio, "sentences": test_sentences}) .cast_column("audio", Audio(sampling_rate=16000))

refs = []
preds = []
for i in range(len(dataset["test"])):
    reference = dataset["test"][i]["sentences"]
    inputs = processor(dataset["test"][i]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features

    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    transcription = after_process(transcription)
    reference = after_process(reference)

    print("pred:", transcription)
    print("reference:", reference)

    if len(transcription) > 0 and len(reference) > 0:
        preds.append(transcription)
        refs.append(reference)

output = jiwer.process_characters(refs, preds)
print(output.cer * 100)
