import requests
import sys
import io
from path import Path


def call_coze_api(text_path):
    personal_access_token = "pat_4G6Q8Y1j3SMlaQdt7kPeGTDHlGc95ViheCLrdPch3PlJ92Vv5E2Pn6ZZfaTPudKe"
    bot_id = "7498933389769129993"
    headers = {
        "Authorization": f'Bearer {personal_access_token}',
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Connection": "keep-alive"
    }
    payload = {
        "bot_id": bot_id,
        "user": "1",
        "query": text_path,
        "stream": False
    }
    response = requests.post(
        "https://api.coze.cn/open_api/v2/chat",
        headers=headers,
        json=payload
    )
    if response.ok:
        print("请求成功，返回数据：")
        response_data = response.json()
        for message in response_data['messages']:
            if message.get("type") == "answer":
                result = message['content']
        return result
    else:
        print("请求失败，状态码：", response.status_code)
        print("错误信息：", response.text)
        return None


def LLM_text_classification(text, prompt):
    old_stdout = sys.stdout
    response = None
    try:
        sys.stdout = io.StringIO()
        call_coze_api(prompt)
        response = call_coze_api(text)
        return response
    finally:
        sys.stdout = old_stdout

if __name__ == '__main__':
    print(LLM_text_classification('我爱你', '开心，愤怒，中性，伤心'))