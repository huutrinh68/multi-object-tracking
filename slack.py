import requests
import json

#URL = 'https://hooks.slack.com/services/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'
URL = 'https://slack.com/api/files.upload'

BOT_TOKEN = 'xoxb-2551747356581-2610518419925-MDKrBNvV1ci19mXywMDfUj7z' #FREE_PARAMで指定
CHANNEL_ID = 'C02GCTFCQEQ'                                              #FREE_PARAMで指定

files = {
        'file': open('img.jpeg','rb')
        }
param = {
        'token': BOT_TOKEN, 
        'channels': CHANNEL_ID,
        'filename': "image.jpeg",
        'initial_comment': "アルゴリズムにより異常が検知されました。",
        'title': "画像"
        }

response = requests.post(URL, params=param, files=files)
