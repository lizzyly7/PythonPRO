import json
import re
import time

path='D:\data\data.json'
data_path ='D:\data\data_custumer.txt'
pattern ='content":".*？"'
data =[]
s_time = time.time()
with open(path,'r',encoding='utf-8') as file:
    lines = file.readlines()
    print(lines[0])
    for line in lines:
        line = json.loads(line)
        try:
            for lin in line['records']:
                if lin['content'] is not '' and lin['speaker'] == 0 and lin['content'] not in data:
                    data.append(lin['content'])
        except KeyError:
            print(lin)

        elapsed_time = time.time() - s_time
        print('processing time',elapsed_time)

with open(data_path,'w',encoding ='utf-8') as file:

    file.write('\n'.join(data))
    file.close()


        #print(line['records'][1])

