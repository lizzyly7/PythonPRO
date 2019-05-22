# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:26:22 2018

@author: Adam-Liusq

@E-mail： 470581985@qq.com
"""
from tqdm import tqdm
import re
from collections import Counter
import sqlite3
import os


file_path = 'chat.conv'
with open(file_path, 'rb') as fp:
    b = fp.read()
content = b.decode('utf8', 'ignore')
lines = []

for line in tqdm(content.split('\n')):
    try:
        line = line.replace('\n', '').strip()
        if line.startswith('E'):
            lines.append('')
        elif line.startswith('M '):
            chars = line[2:].split('/')
            while len(chars) and chars[len(chars) - 1] == '.':
                chars.pop()
            if chars:
                sentence = ''.join(chars)
                sentence = re.sub('\s+', '，', sentence)
                lines.append(sentence)
    except:
        print(line)
        lines.append('')

QADict={}
for i in range(0,len(lines),3):
    QADict[lines[i+1]] = lines[i+2]
    

def contain_chinese(s):
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False

def valid(a, max_len=0):
    if len(a) > 0 and contain_chinese(a):
        if max_len <= 0:
            return True
        elif len(a) <= max_len:
            return True
    return False

def insert(a, b, cur):
    cur.execute("""
    INSERT INTO conversation (ask, answer) VALUES
    ('{}', '{}')
    """.format(a.replace("'", "''"), b.replace("'", "''")))

def insert_if(question, answer, cur, input_len=500, output_len=500):
    if valid(question, input_len) and valid(answer, output_len):
        insert(question, answer, cur)
        return 1
    return 0

db = 'db/conversation.db'
if os.path.exists(db):
    os.remove(db)
conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS conversation
    (ask text, answer text);
    """)
conn.commit()

words = Counter()
a = ''
b = ''
inserted = 0

for q, a in tqdm(QADict.items(), total=len(QADict)):
    words.update(Counter(q+a))
    ask = q
    answer = a
    inserted += insert_if(ask, answer, cur)
    # 批量提交
    if inserted != 0 and inserted % 5000 == 0:
        conn.commit()
conn.commit()    