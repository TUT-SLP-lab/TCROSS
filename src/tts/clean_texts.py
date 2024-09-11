import re

text = ''
with open('./created_texts.txt', 'r') as f:
    for line in f:
        line = re.sub('[0-9]*\.', '', line)
        text += line.strip()
        text += '\n'

with open('./clean_created_texts.txt', 'w') as f:
    f.write(text)
