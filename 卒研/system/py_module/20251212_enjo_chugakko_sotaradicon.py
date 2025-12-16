import os
import uuid
import queue
import subprocess
from datetime import datetime

import numpy as np
import sounddevice as sd

import soundfile as sf

import requests
import shutil
import time
import threading
import asyncio
import copy
import json

import cv2
from openai import OpenAI


# ----- 設定パラメータ -----
SAMPLE_RATE = 16000  # サンプリングレート
CHANNELS = 1  # モノラル
CHUNK_DURATION = 0.1  # チャンク長（秒）
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

AUDIO_ACTIVATE_AMPLITUDE_THRESH = 0.02  # 録音開始判定用振幅閾値

BEGIN_SEGMENT_TIME_LIMIT = 0.5  # (区間開始判定)無音とみなす連続時間（秒）
BEGIN_SEGMENT_THRESH = 0.2  # (区間開始判定)無音とみなす振幅閾値

END_SEGMENT_TIME_LIMIT = 2.0  # (区間終了判定)無音とみなす連続時間（秒）
END_SEGMENT_THRESH = 0.1  # (区間終了判定)無音とみなす振幅閾値

AUDIO_GAIN = 0.8  # 4.0 # 録音音声のゲイン調整用

device_id = 0


# ----- 可視化関係 ---------


record_dir = "./rec"
send_dir = "./__wavtmp"


# 音声認識関連
# modelpath_whisper = "models/ggml-small.bin"
# modelpath_whisper = "models/ggml-base.bin"
# modelpath_whisper = "models/ggml-tiny.bin"
modelpath_whisper = "models/ggml-small-q5_1.bin"
# modelpath_whisper = "models/ggml-medium-q5_0.bin"
thread_size = 8

transcription_stock = ""

asr_file_format = "WAV"  # "wav"  #"flac"
asr_file_ext = ".wav"  # ".wav"  #".flac"


# キューと状態変数
audio_queue = queue.Queue()
recording = False
audio_active = False
frames = []
silence_buf_time = 0.0
active_buf_time = 0.0


# 波形ビジュアライズのための画像設定
img_width = 640
img_height = 480

# セルフエコーキャンセリング用の制御変数
force_muted = True
is_sota_is_saying = False
sota_is_online = False


if asr_file_format == "FLAC":
    asr_file_ext = ".flac"


# --- LLM client -----------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")
print(f"api_key:{api_key[0:20]}...")

if api_key is None:
    print("OPENAI_API_KEY 環境変数が設定されていません。")
    exit()

client = client = OpenAI()


def generate_by_llm(model_use, prompt):

    unsafe_response = client.responses.create(
        model=model_use, input=prompt, reasoning={"effort": "minimal"}
    )

    return unsafe_response.output_text


# --- prompt templates -----------------------------------------------------------
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


prompt_base_question = """
あなたは優秀なインタビュアーで、ユーザーの最近の出来事を聞くのが役割です。
あなたはユーザーの話に興味を持っています。
ユーザーの発言を受けて、さらに会話を盛り上げる質問を日本語で丁寧に作ってください。
会話記録を考慮し、自然な質問となるようにしてください。
質問文は短く簡潔にしてください。
質問文だけ出力してください。
質問文の表現は口語表現にしてください。テキストチャット特有の表現（かっこで例示するなど）は避けてください。


# インタビューについての前提情報
ユーザーは名古屋市内に在住の中学生です。
今回、ユーザーは学校の課題であなたからのインタビューに答えています。
ユーザーは前回の授業で名古屋市の金山駅前について調べてくるという課題を出されていました。
ユーザーが課された宿題は以下のようなものです：
    課題：次回金曜までに、金山駅前についてインターネットで検索して調べてみよう
    調べる内容：金山駅前の再整備計画について、金山駅前まちそだて会の取組について、その他いろいろ金山周辺について
    調査以外にやるべきこと：Gemini(GoogleのLLMチャットボット)と議論してみる、どんなアプリやサービスがあると良さそうか妄想してみよう
あなたは、上記の課題についてユーザーが考えてきたこと・調べてきたことを深掘りしたり、関連する話題について質問を行ってください。
今日は2025年12月12日です。


# 金山駅前についての基本情報（インタビュアーとして知っておくべきこと）
- 金山駅前のまちの課題：
    - 線路や行政区の境で分断されている
    - 金山駅の利用者は非常に多い（1日47万人）が、駅の外に出てくれる人の数や回遊性はイマイチ
    - 2028年のアスナル金山という商業施設の閉鎖と再開発が予定されている
    - 「人・文化・芸術とともに育つまち」という名古屋市による再開発方針
- 金山駅前まちそだて会：
    - 金山駅前のまちづくりの団体
    - ビジョンが「交通拠点から交流拠点へ」


---
会話記録：
{dialog_log}
---
ユーザー発話：
{user_speech_text}
---
質問文：
"""


# --- dialog management -----------------------------------------------------------

dialog_log = []  # 会話履歴


def append_dialog_log(speaker, text):
    global dialog_log

    entry = f"{speaker}: {text}"
    dialog_log.append(entry)

    # 会話履歴が長くなりすぎたら古いものを削除
    max_entries = 200
    if len(dialog_log) > max_entries:
        dialog_log = dialog_log[-max_entries:]


def process_dialog(user_utterance):

    global dialog_log
    append_dialog_log("ユーザー", user_utterance)

    # LLMに質問文を生成させる
    dialog_log_text = "\n".join(dialog_log)
    prompt_question = prompt_base_question.format(
        dialog_log=dialog_log_text, user_speech_text=user_utterance
    )
    model_use = "gpt-5"
    question_text = generate_by_llm(model_use, prompt_question)

    append_dialog_log("インタビュアー", question_text)

    # 対話ログをタイムスタンプ付きで保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dialog_log_filename = f"dialog_log_{timestamp}.txt"
    with open(dialog_log_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(dialog_log))

    return question_text.strip()


# -----------------------


def audio_callback(indata, frames_count, time_info, status):

    # InputStream のコールバック。キューへ生データを流し込むだけ。
    if status:
        print(f"Audio callback status: {status}", flush=True)
    audio_queue.put(indata.copy())


def make_filename():
    # タイムスタンプ＋UUID で一意のファイル名を生成
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"rec_{ts}_{uid}"


def process_segment(wav_data: np.ndarray):

    # WAV 保存
    filename = make_filename()
    savepath = f"{record_dir}/{filename}{asr_file_ext}"

    sf.write(savepath, wav_data, SAMPLE_RATE, closefd=True, format=asr_file_format)
    print(f"Saved segment: {savepath}")

    return savepath, filename


# whisper.cppによる文字起こし
def transcription_whispercpp(wav_path):

    commandline = f"bin\\whisper-cli.exe -m {modelpath_whisper} -t {thread_size} -f {wav_path} -l ja -ng -nt -np"
    process = subprocess.Popen(commandline, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
    output_str = output.decode("utf-8").replace("\n", "").replace("\r", "")

    return output_str


def transcription(audio_path):

    # 文字起こしに要した時間を計測しておく
    __time_begin = time.perf_counter()
    output_str = transcription_whispercpp(audio_path)

    __time_end = time.perf_counter()

    print(f"Transcription time: {__time_end - __time_begin:.2f} seconds")

    return output_str


sota_ip = "133.68.80.199:8082"
sota_ip = "192.168.0.3:8082"
sota_ip = "192.168.137.40:8082"
sota_ip = "192.168.137.105:8082"


def say_sota(message):
    # Sotaに送る。↓のようなPOSTを送る。
    """'
    POST http://133.68.80.199:8082/say HTTP/1.1
    content-type: application/json

    {
        "message":"こんにちは,制御用PCとのメッセージ送受信テストです。少し長めに話しています。",
        "motion":"hello"
    }
    """
    url = f"http://{sota_ip}/say"
    headers = {"content-type": "application/json"}
    data = {"message": message, "motion": "talk"}
    try:
        response = requests.post(url, json=data, headers=headers, timeout=5)
        if response.status_code == 200:
            print("Sent to Sota successfully.")
        else:
            print(f"Failed to send to Sota. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending to Sota: {e}")

    return


def status_sota(in_progress=False, listening=False, speaking=False, unavailable=False):

    if unavailable:
        eye_color = "red"
    elif speaking:
        eye_color = "cyan"
    elif in_progress:
        eye_color = "red"
    elif listening:
        eye_color = "green"
    else:
        eye_color = "white"

    led_data = {
        "color_eye_left": eye_color,
        "color_eye_right": eye_color,
        "blightness_mouth": "0",
        "color_power_button": "green",
    }

    url = f"http://{sota_ip}/led"
    headers = {"content-type": "application/json"}

    # Sotaに送る。↓のようなPOSTを送る。
    try:
        response = requests.post(url, json=led_data, headers=headers, timeout=1)
        if response.status_code == 200:
            pass
            # print("Sent to Sota (LED) successfully.")
        else:
            print(f"Failed to send to Sota. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending to Sota: {e}")

    return


# Sotaの発話状況を取得する。重いので非同期版として動かす
def check_state_sota_saying_async():

    url = f"http://{sota_ip}/status-say"

    try:
        response = requests.post(url, timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            # print(f"Sota status data: {status_data}")
            return status_data.get("is_saying", False), True
        else:
            print(f"Failed to get Sota status. Status code: {response.status_code}")
            return False, False

    except requests.exceptions.RequestException as e:
        # print(f"Error getting Sota status: {e}")
        return False, False


# check_state_sota_saying_asyncを非同期で実行し、結果をグローバル変数に反映する
def update_state_sota_saying():
    global is_sota_is_saying, sota_is_online

    while True:
        is_sota_is_saying, sota_is_online = check_state_sota_saying_async()
        # print(f"Sota is_saying: {is_sota_is_saying}")
        time.sleep(0.2)
    return


def send_talker(user_speech_text):

    # 表示するだけ
    print(f"speech text : {user_speech_text}")

    # Sotaに送信
    say_sota(user_speech_text)

    return


def task_tran_and_send(rec_file_wav):

    status_sota(in_progress=True)

    print("Transcription.")
    tran_result = transcription(rec_file_wav)
    msg = f"{rec_file_wav}->{tran_result}"
    print(msg)
    print("Transcription end.")

    # exit()
    print("Process dialog.")
    question_text = process_dialog(tran_result)
    print("Process dialog end.")

    print("Send.")
    send_talker(question_text)

    status_sota(listening=True)


def task_tran_and_stock(rec_file_wav):

    print("Transcription.")
    tran_result = transcription(rec_file_wav)
    msg = f"{rec_file_wav}->{tran_result}"
    print(msg)
    print("Transcription end.")

    # exit()
    print("Process dialog.")
    question_text = process_dialog(tran_result)
    print("Process dialog end.")

    print("Send.")
    send_talker(question_text)


# 画像をshiftピクセル左に移動する関数
# 画像の右端に空白を追加
def shift_x(img, dx, dy):

    # 画像サイズ
    height = img.shape[0]  # 高さ
    width = img.shape[1]  # 幅

    # 平行移動の変換行列を作成
    affine_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    # アファイン変換適用
    affined_img = cv2.warpAffine(img, affine_matrix, (width, height))

    return affined_img


def visualize_amplitude(
    img,
    amplitude_min,
    amplitude_max,
    width,
    height,
    force_muted,
    robot_available,
    robot_online,
    audio_active,
    is_recording,
    frames_in_buffer_count,
    rate=1.0,
):

    stepwise = 2

    # openCVの画像を1ピクセル左に移動する
    img = shift_x(img, -1 * stepwise, 0)

    # 画像レベルの0点を中央に設定
    center_y = height // 2

    wave_height_max = center_y - 10  # 波形の最大高さ

    hakei_color = (160, 160, 160)
    robot_color = (100, 255, 100)
    audio_label = "AUDIO : OFF"
    robot_label = "ROBOT : AVAILABLE"

    if robot_available == False:
        robot_color = (255, 100, 255)
        robot_label = "ROBOT : BUSY"

    if robot_online == False:
        robot_color = (0, 0, 255)
        robot_label = "ROBOT : OFFLINE"

    if audio_active:
        hakei_color = (255, 255, 255)
        audio_label = "AUDIO : ACTIVE"

    if is_recording:
        hakei_color = (100, 255, 100)
        audio_label = "AUDIO : IN SEGMENT"

    if force_muted:
        hakei_color = (100, 100, 255)
        audio_label = "AUDIO : MUTED"

    # ラベル表示用エリアの下地を黒く塗りつぶし
    cv2.rectangle(img, (0, 0), (400, 70), (0, 0, 0), -1)

    # 画面左端に状態表示
    cv2.putText(
        img,
        f"{audio_label}:{frames_in_buffer_count:03}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        hakei_color,
        1,
    )
    cv2.putText(
        img, robot_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, robot_color, 1
    )

    cv2.putText(
        img,
        f"AUDIO_GAIN:{AUDIO_GAIN:.3f}",
        (220, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        hakei_color,
        1,
    )

    # 0点の線を引く
    cv2.line(img, (0, center_y), (width - 1, center_y), (255, 255, 255), 1)

    # 録音開始レベルの線を引く
    y_startline_upper = center_y - int(BEGIN_SEGMENT_THRESH * wave_height_max * rate)
    y_startline_lower = center_y + int(BEGIN_SEGMENT_THRESH * wave_height_max * rate)

    cv2.line(
        img, (0, y_startline_upper), (width - 1, y_startline_upper), (50, 50, 200), 1
    )
    cv2.line(
        img, (0, y_startline_lower), (width - 1, y_startline_lower), (50, 50, 200), 1
    )

    # 録音終了レベルの線を引く
    y_endline_upper = center_y - int(END_SEGMENT_THRESH * wave_height_max * rate)
    y_endline_lower = center_y + int(END_SEGMENT_THRESH * wave_height_max * rate)
    cv2.line(img, (0, y_endline_upper), (width - 1, y_endline_upper), (200, 50, 50), 1)
    cv2.line(img, (0, y_endline_lower), (width - 1, y_endline_lower), (200, 50, 50), 1)

    # print(f"Amplitude: {amplitude}")
    # 音声レベルの波形を描画
    # 振幅を正規化して高さを計算
    norm_amplitude_min = int((amplitude_min) * (wave_height_max) * rate)
    norm_amplitude_max = int((amplitude_max) * (wave_height_max) * rate)
    # print(f"norm_amplitude:{norm_amplitude}")

    # 振幅の線を描画
    cv2.line(
        img,
        (width - 1, center_y),
        (width - 1, center_y - norm_amplitude_min),
        hakei_color,
        stepwise,
    )
    cv2.line(
        img,
        (width - 1, center_y),
        (width - 1, center_y - norm_amplitude_max),
        hakei_color,
        stepwise,
    )

    return img


def message_greetings():

    greeting_text = "こんにちは。今回の課題である、金山駅前の再開発に関して、調べてきたことについてお話を聞かせてください。よろしくお願いします。"

    status_sota(in_progress=True)

    append_dialog_log("システム", greeting_text)
    send_talker(greeting_text)

    status_sota(listening=True)


def dump_dialog_log():

    global dialog_log

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dialog_log_filename = f"dialog_log_{timestamp}.json"

    dialog_log_dict = {"datetime": timestamp, "dialog_log": dialog_log}

    with open(dialog_log_filename, "w", encoding="utf-8") as f:
        json.dump(dialog_log_dict, f, ensure_ascii=False, indent=4)
    print(f"Dialog log dumped to {dialog_log_filename}")


def clear_dialog_log():

    global dialog_log
    dialog_log = []
    print("Dialog log cleared.")


def main():
    global recording, frames, silence_buf_time, active_buf_time, force_muted, is_sota_is_saying, sota_is_online, audio_active, AUDIO_GAIN

    # Sotaの発話状態を監視するスレッドを起動
    thread_sota_state = threading.Thread(
        target=update_state_sota_saying, args=()
    )  # 処理を割り当てる
    thread_sota_state.daemon = True
    thread_sota_state.start()

    frames_in_buffer_count = 0

    print("Start recording...  [Ctrl+C]->Quit")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    ):

        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        try:
            while True:
                data = audio_queue.get()

                # Sotaが話している最中は無音扱いにする
                if force_muted or is_sota_is_saying:
                    data = np.zeros_like(data)

                # if not is_sota_is_saying:
                #    status_sota( listening = True)
                # else:
                #    status_sota( speaking = True)

                # 入力音声にゲインをかける
                data = data * AUDIO_GAIN

                # print(f"is_sota_is_saying:{is_sota_is_saying}, force_muted:{force_muted}")

                amplitude = np.max(np.abs(data))

                amplitude_max = np.max(data)
                amplitude_min = np.min(data)

                if recording:  # 録音しているとき
                    frames.append(data)
                    if amplitude < END_SEGMENT_THRESH:

                        # 無音が連続している時間
                        silence_buf_time += CHUNK_DURATION

                        # 無音が規定時間を超えたらセグメント終了
                        if silence_buf_time >= END_SEGMENT_TIME_LIMIT:

                            print("Segment end.")

                            frames_to_save = copy.deepcopy(frames)
                            wav = np.concatenate(frames_to_save, axis=0)
                            savepath, filename = process_segment(wav)

                            thread_ts = threading.Thread(
                                target=task_tran_and_send, args=([savepath])
                            )  # 処理を割り当てる
                            thread_ts.start()

                            # 状態リセット
                            recording = False
                            frames = []
                            silence_buf_time = 0.0
                    else:
                        silence_buf_time = 0.0

                else:  # 録音していないとき

                    # print(f"active_buf_time:{active_buf_time}")
                    # 音声入力判定
                    if AUDIO_ACTIVATE_AMPLITUDE_THRESH <= amplitude:

                        audio_active = True
                        # とりあえず録音バッファに追加
                        frames.append(data)
                        # 音が連続して入っている時間
                        active_buf_time += CHUNK_DURATION

                        # セグメント開始レベルを超えていて
                        if BEGIN_SEGMENT_THRESH <= amplitude:
                            # 規定時間を超えたらセグメント開始
                            if BEGIN_SEGMENT_TIME_LIMIT <= active_buf_time:
                                recording = True
                                print("Segment start.")

                    else:
                        active_buf_time = 0.0
                        frames = []
                        audio_active = False

                frames_in_buffer_count = len(frames)

                # 波形ビジュアライズ
                robot_is_available = not is_sota_is_saying
                img = visualize_amplitude(
                    img,
                    amplitude_min,
                    amplitude_max,
                    img_width,
                    img_height,
                    force_muted,
                    robot_is_available,
                    sota_is_online,
                    audio_active,
                    recording,
                    frames_in_buffer_count,
                )

                cv2.imshow("Waveform", img)
                key = cv2.waitKey(1)
                # print("key:", key)
                if key == 109:  #'mキー'
                    force_muted = not force_muted
                    print(f"force_muted: {force_muted}")
                if key == 27:  #'ESCキー'
                    break

                if key == ord("w"):  #'[up]キー'
                    AUDIO_GAIN += 0.2

                if key == ord("q"):  #'[down]キー'
                    AUDIO_GAIN -= 0.2

                if key == ord("g"):  #'gキー'
                    message_greetings()

                if key == ord("d"):  #'dキー'
                    dump_dialog_log()

                if key == ord("c"):  #'cキー'
                    clear_dialog_log()

        except KeyboardInterrupt:
            print("\nStop recording.")


if __name__ == "__main__":

    device_list = sd.query_devices()
    # print(device_list)
    # exit()

    main()
