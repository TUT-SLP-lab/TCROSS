from typing import Optional
from dotenv import load_dotenv
import os
from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

openai_key = os.getenv("OPENAI_KEY")
access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

def make_prompt(key_word, n):
    messages = [
        {"role": "system", "content": "あなたは優秀な循環器の専門医です"},
        {"role": "user", "content": 
            f"""
            あなたにキーワードを与えるので、私にこのキーワードを含む文を生成して下さい。
            条件：
            1. 文を生成するたびに、いつも戦闘に現れないようにキーワードの文中の位置を調整してください.
            2. 生成される文は生成された文と重複しないで下さい.
            3. 生成された文は日本語だけで, ローマ字と翻訳は必要ありません.
            4. このキーワードを使って, "医者"の視点で, {n}個の日本語文を生成してください.
            キーワード: {key_word}
            """
         },
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt

def download_model(model_id: str, revision:Optional[str]=None):
    UC_VOLUME = "./"

    rev_dir = ("--" + revision) if revision else ""
    local_dir = f"/tmp/{model_id}{rev_dir}"
    uc_dir = f"/models--{model_id.replace('/', '--')}"
    
    snapshot_location = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=access_token,
    )

    dbutils.fs.cp(f"file:{local_dir}", f"{UC_VOLUME}{uc_dir}{rev_dir}", recurse=True)

model_id = "google/gemma-7b-it"
download_model(model_id)
# import kagglehub
#
# path = kagglehub.model_download("google/gemma/pyTorch/7b-it")
# print(path)
#
# exit()
model_path = "./models--google--gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
def generate_text(prompt: str):
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    outputs = model.generate(input_ids=input.to(model.device), max_new_tokens=150)

    print(tokenizer.decode(outputs[0]))
prompt = make_prompt("呼吸器", 1)
