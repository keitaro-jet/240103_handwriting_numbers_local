# 必要なライブラリをインポート
import os  # OSモジュールのインポート
import requests
import streamlit as st  # Streamlitライブラリのインポート
import pandas as pd  # Pandasライブラリのインポート
from io import BytesIO  # BytesIOクラスのインポート

# サーバーから受け取った結果ファイルのフォーマットを行う関数
def format_results(results):
    return pd.DataFrame(results)  # 受け取った結果をPandasのDataFrameに変換して返す

# バックエンドのホストを設定
BACKEND_HOST = '127.0.0.1:8000' 

# Streamlitを使用して画像ファイルをアップロード
image_files = st.file_uploader('Target image file', type=['png', 'jpg'], accept_multiple_files=True)  # ユーザーに画像ファイルをアップロードするUIを表示

# 画像ファイルが1つ以上あり、Submitボタンが押された場合に処理を実行
if image_files and st.button('Submit'):  # 画像ファイルが存在し、Submitボタンが押されたかどうかを確認
    # 送信するファイルのリストを作成（ファイル名、ファイルの中身、ファイルのMIMEタイプ）
    files = [('files', (file.name, BytesIO(file.read()), file.type)) for file in image_files]
    # request POSTリクエストを使ってサーバーにファイルを送信
    r = requests.post(f'http://{BACKEND_HOST}/predict', files=files)  
    # 結果を取得して表示
    df_results = format_results(r.json())  # サーバーからの結果をDataFrameに変換
    st.write(df_results)  # 結果をStreamlitアプリに表示

# Refreshボタンが押された場合の処理
if st.button('Refresh'):  # Refreshボタンが押されたかどうかを確認
    st.success('Refreshed')  # 成功メッセージを表示