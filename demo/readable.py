import os
import re
import json,logging
from geopy.geocoders import  Nominatim
'''
geolocator = Nominatim()
location = geolocator.geocode('北京')
print(location)




'''
path = r'C:\Users\Administrator\Desktop\smartzhongtong.sql'
word=[]
with open(path,'r',encoding='utf-8')as f:
    con = f.read()
    #c=re.match(r'[\u4e00-\u9fa5]+',con)
    pattern = r'[\u4e00-\u9fa5]+'
    all_words = re.findall(pattern,con)
    print(['\n'.join(set(all_words))])
