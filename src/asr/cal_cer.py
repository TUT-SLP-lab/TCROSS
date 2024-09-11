import json

import jiwer
import numpy as np
from jiwer import cer
from pydub import AudioSegment

with open('../eval_audio/manifest.json', 'r') as f:
    manifest = json.load(f)

names = ['VTS_02_1', 'VTS_02_2', 'VTS_02_3']
sounds = {}
sounds['./VTS_02_1.wav'] = AudioSegment.from_file('../eval_audio/VTS_02_1.wav', 'wav')
sounds['./VTS_02_2.wav'] = AudioSegment.from_file('../eval_audio/VTS_02_2.wav', 'wav')
sounds['./VTS_02_3.wav'] = AudioSegment.from_file('../eval_audio/VTS_02_3.wav', 'wav')

groundtruth = ''
pred = ''
for name in names:
    for data in manifest:
        if name in data['audio_filepath']:
            groundtruth += data['text']
    with open(f'../transcribes/whisper_all_audio/{name}_whisper_result.txt', 'r') as f:
        for line in f:
            pred += str(line.split(':')[-1].strip())

output = cer(groundtruth, pred)
print(f"cer: {name} {output}")
