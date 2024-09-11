from langchain_community.llms import Ollama
from dotenv import load_dotenv
import pandas as pd
import ollama

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
            1. 文を生成するたびに、いつもキーワードが先頭に現れないように、キーワードの位置を調整してください.
            2. 生成される文は他の生成された文と重複しないで下さい.
            3. 生成された文は日本語だけで, ローマ字と翻訳は必要ありません.
            4. このキーワードを使って, "循環器の専門医"の視点で, {n}個の文を生成してください.
            5. 生成する文章は日本語にしてください.
            キーワード: {key_word}
            """
         },
    ]
    return messages

def post_process(text):
    output = []
    lines = text.split('\n')
    for line in lines:
        if len(line)==0:
            continue
        if line[0].isdecimal():
            output.append(line[3:].strip())
    return output

def use_all_keyword():
    keyword_df = pd.read_excel('../train/train_text/用語集.xlsx')
    abbreviation_df = pd.read_excel('../train/train_text/略語.xls')
    n = 1
    print(keyword_df.columns)
    print(abbreviation_df.columns)
    for keyword in keyword_df['Unnamed: 3']:
        with open("created_texts.txt", 'a') as f:
            prompt = make_prompt(keyword, 10) 
            for i in range(n): 
                response = ollama.chat(model="llama3", messages=prompt)
                text = response["message"]["content"]
                print(text)
                f.write(text)

def use_select_keyword():
    keywords = [
        # "活性凝固時間",
        # "脳性ナトリウム利尿ペプチド",
        # "血圧",
        # "分岐部",
        # "生体吸収性スキャフォールド",
        # "冠動脈バイパス術",
        # "連続心拍出量",
        # "慢性腎臓病",
        # "心肺蘇生法",
        # "両心室ペースメーカ",
        # "中心静脈カテーテル",
        # "深大腿動脈",
        # "推算糸球体濾過量",
        # "B型肝炎ウイルス",
        # "心拍数",
        # "血管内超音波検査",
        # "左内胸動脈",
        # "下位右房側壁",
        # "左冠動脈主幹部",
        # "左下肺静脈",
        # "左室流出器",
        # "光周波数領域画像法",
        # "肺動脈圧",
        # "肺動脈楔入圧",
        # "肺動脈血栓内膜摘除術",
        # "膝窩動脈",
        # "経皮的腎動脈形成術",
        # "右冠尖",
        # "右上肺静脈",
        "洞結節回帰時間",
        # "洞不全症候群",
        # "胸部大動脈瘤",
        # "頻拍周期",
        # "一過性脳虚血発作",
        # "房室",
        # "心室細動",
        # "心室連続刺激",
        # "心室頻拍",
        # "東海メディカルﾌﾟﾛﾀﾞｸﾂのバルーン付きガイディングカテーテルの名称。",
        # "回収式自己血輸血装置の名称。",
        # "鎮痛作用",
        # "抗血小板剤（血液を固まりにくくする作用）",
        # "動脈塞栓除去用カテーテル",
        # "左房後壁隔離術。心房細動のアブレーションで行われる手技。通常の拡大肺静脈隔離に加え、ルーフとボトムという焼灼ラインを追加して左房後壁ごと隔離する。"
    ]    
    n = 1
    for keyword in keywords:
        with open("created_texts.txt", 'a') as f:
            prompt = make_prompt(keyword, 10) 
            for i in range(n): 
                response = ollama.chat(model="llama3", messages=prompt)
                text = response["message"]["content"]
                print(text)
                # texts = post_process(text)
                f.write(text)
if __name__ == '__main__':
    # use_all_keyword()
    use_select_keyword()
