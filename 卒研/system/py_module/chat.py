from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 取得したAPIキーを設定
api_key = os.getenv("OPENAI_API_KEY")
# モデル名を指定
model_name = "gpt-4o-mini"

client = OpenAI(api_key=api_key)

# 初期インストラクション

system_instruction = """あなたの名前はGeminiです。
会話の記録の後ろに、ユーザーの発言があります。
フレンドリーに、100文字以内で返答してね。
"""

open_prompt = """
あなたはカウンセラーです。
ユーザに対してオープンクエスチョンのみを用いて質問をしてください。
オープンクエスチョンとは、相手がはい、いいえで答えられない質問のことです。
ユーザの応答から情報を読み取り、深掘りできそうな箇所を見つけて質問してください。
"""

closed_prompt = """
あなたはカウンセラーです。
ユーザに対してクローズドクエスチョンのみを用いて質問をしてください。
クローズドクエスチョンとは、相手がはいかいいえで答えられる質問のことです。
ユーザの応答から情報を読み取り、深掘りできそうな箇所を見つけて質問してください。

例: 睡眠時間はどれくらい？ -> 睡眠時間は8時間ですか？
"""


def switch_prompt(is_open: bool) -> str:
    """
    質問形式に応じたプロンプトを返す関数

    Args:
        is_open (bool): Trueの場合はオープンクエスチョン、Falseの場合はクローズドクエスチョン

    Returns:
        str: 選択されたプロンプト
    """
    return open_prompt if is_open else closed_prompt


# 会話を覚えておくための変数
messages = [{"role": "system", "content": system_instruction}]

# 何回も対話できるようにループする
while True:
    # ユーザーの発言を入力
    speech_user = input("user:")

    # 「おしまい」とだけユーザーが入力したら対話終了
    if speech_user == "おしまい":
        print(f"model:おしまいです。さようなら")
        break

    # ユーザーの発言をメッセージに追加
    messages.append({"role": "user", "content": speech_user})

    selected_prompt = switch_prompt(is_open=False)
    # システムプロンプトを一時的に変更
    temp_messages = [{"role": "system", "content": selected_prompt}] + messages[1:]

    # ChatGPT APIを呼び出し
    response = client.chat.completions.create(
        model=model_name, messages=temp_messages, max_tokens=256
    )

    # アシスタントの返答を取得
    assistant_message = response.choices[0].message.content
    print(f"model:{assistant_message}")

    # アシスタントの返答をメッセージ履歴に追加
    messages.append({"role": "assistant", "content": assistant_message})
