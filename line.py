import requests

URL = 'https://notify-api.line.me/api/notify'
TOKEN = '1diSoy8QSo2329ISadwlU4DyBPuKebpdzQ5OYrRVo6P' #FREE_PARAMで指定

headers = { 'Authorization': f'Bearer {TOKEN}' }
data = { 'message': '通知されましたよ？' }
files = {
        'imageFile': open('img.jpeg','rb') #png,jpgのみ
        }

response = requests.post(URL, headers=headers, data=data, files=files)
print(response.headers)
#{
#    'Server': 'nginx'
#    'Date': 'Sun, 17 Oct 2021 08:37:27 GMT'
#    'Content-Type': 'application/json;charset=UTF-8'
#    'Transfer-Encoding': 'chunked'
#    'Connection': 'keep-alive'
#    'Keep-Alive': 'timeout=3'
#    'X-RateLimit-Limit': '1000'
#    'X-RateLimit-ImageLimit': '50'
#    'X-RateLimit-Remaining': '994'
#    'X-RateLimit-ImageRemaining': '48'
#    'X-RateLimit-Reset': '1634462878'
#}
