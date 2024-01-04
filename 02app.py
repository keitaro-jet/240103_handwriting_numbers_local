# 必要なライブラリをインポート
import os
import httpx
import streamlit as st
import pandas as pd
from io import BytesIO

# サーバーから受け取った結果ファイルのフォーマットを行う関数
def format_results(results):
    return pd.DataFrame(results)

# バックエンドのホストを設定
BACKEND_HOST = '127.0.0.1:8000'

# Streamlitを使用して画像ファイルをアップロード
image_files = st.file_uploader('Target image file', type=['png', 'jpg'], accept_multiple_files=True)

# 画像ファイルが1つ以上あり、Submitボタンが押された場合に処理を実行
if image_files and st.button('Submit'):
    files = [('files', (file.name, BytesIO(file.read()), file.type)) for file in image_files]
    # HTTP POSTリクエストを使ってサーバーにファイルを送信
    r = httpx.post(f'http://{BACKEND_HOST}/predict', files=files)
    # 結果を取得して表示
    df_results = format_results(r.json())
    st.write(df_results)

# Refreshボタンが押された場合の処理
if st.button('Refresh'):
    st.success('Refreshed')