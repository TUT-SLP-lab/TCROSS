import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import ginza
import spacy
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from pydub import AudioSegment
from transformers import (AutoProcessor, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, WhisperFeatureExtractor,
                          WhisperForConditionalGeneration, WhisperProcessor,
                          WhisperTokenizer)

os.environ["WANDB_PROJECT"] = "tcross_whisper_finetune"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
# model_name = "Watarungurunnn/whisper-large-v3-ja"
# model_name = "vumichien/whisper-small-ja"
# model_name = "openai/whisper-large-v3"
# model_name = "openai/whisper-large"
# model_name = "openai/whisper-small"
model_name = "openai/whisper-medium"

processor = WhisperProcessor.from_pretrained(model_name, language="Japanese", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def check_dataset(audio: List, reference: List):
    for i, ref in enumerate(reference):
        if ref == "":
            audio.pop(i)
            reference.pop(i)

    for i, ref in enumerate(reference):
        data = AudioSegment.from_file(audio[i], "wav")
        if len(data)==0:
            audio.pop(i)
            reference.pop(i)

    assert len(audio) == len(reference)
    
    return audio, reference


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    metric = evaluate.load('cer')
    nlp = spacy.load("ja_ginza")
    ginza.set_split_mode(nlp, "C")

    # replace -100 with the pad_token_id
    label_ids[label_ids==-100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # pred_str = [" ".join([str(i) for i in nlp(j)]) for j in pred_str]
    # label_str = [" ".join([str(i) for i in nlp(j)]) for j in label_str]
    pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
    label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{model_name}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1, # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    # report_to=["tensorboard"],
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
    # remove_unused_columns=False,
)

dataset = DatasetDict()
train_audio = glob.glob("../../data/clean/train/train_audio/*.wav") 
tmp_dict = {}
for path in train_audio:
    num = path.split("/")[-1].split(".")[0].split("_")[-1]
    tmp_dict[path] = int(num)
train_audio = [item[0] for item in sorted(tmp_dict.items(), key=lambda x:x[1])]

train_sentences = []
with open("../../data/clean/train/train_text/clean_created_texts_by_manually.txt", "r") as f:
    for sentence in f.readlines():
        sentence = re.sub("^[0-9]*.", "", sentence).replace(" ", "")
        print(sentence)
        train_sentences.append(sentence)
train_audio, train_sentences = check_dataset(train_audio, train_sentences)
dataset["train"] = Dataset.from_dict({"audio": train_audio, "sentence": train_sentences}).cast_column("audio", Audio(sampling_rate=16000))

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
test_audio, test_sentences = check_dataset(test_audio, test_sentences)
dataset["test"] = Dataset.from_dict({"audio": test_audio, "sentence": test_sentences}).cast_column("audio", Audio(sampling_rate=16000))

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)

model = WhisperForConditionalGeneration.from_pretrained(model_name)

model.generation_config.language = "ja"
model.generation_config.task = "transcribe"

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor = processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
print("training start")
trainer.train()
print("finished train!!")

for i in range(len(dataset["train"])):
    inputs = processor(dataset["train"][i]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features

    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(transcription)

