import os
import sys
import io
from coze import Coze


def LLM_text_classification(text, prompt):
    os.environ['COZE_API_TOKEN'] = 'pat_4G6Q8Y1j3SMlaQdt7kPeGTDHlGc95ViheCLrdPch3PlJ92Vv5E2Pn6ZZfaTPudKe'
    os.environ['COZE_BOT_ID'] = '7498933389769129993'
    response = None
    old_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        chat = Coze(
            api_token=os.environ['COZE_API_TOKEN'],
            bot_id=os.environ['COZE_BOT_ID'],
            max_chat_rounds=20,
            stream=True
        )
        chat(prompt)
        response = chat(text)
    finally:
        sys.stdout = old_stdout
    return response

if __name__ == '__main__':
    LLM_text_classification('我爱你', '开心，愤怒，中性，伤心')