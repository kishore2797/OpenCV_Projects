import requests

api_key = 'acc_f8b0699171b4727'
api_secret = '3773dfebe12efadd78929ad1f9297dd1'
image_url = 'https://media.wired.com/photos/5d09594a62bcb0c9752779d9/master/w_2560%2Cc_limit/Transpo_G70_TA-518126.jpg0'

response = requests.get('https://api.imagga.com/v2/tags?image_url=%s' % image_url, auth=(api_key, api_secret))

print(response.json())