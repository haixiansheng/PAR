import json
from shutil import copyfile
import os
import cv2
import threadpool

def creat_dataset(y):
    # y = annos["annotations"]

    # for y in annos["annotations"]:
    # print(annos['annotations'][i])
    if "1-smoking" in y:
        fi = open(f"labels.txt", 'a+')
        fi.write(f"{y} 0 1\r\n")
        fi.close()
    elif "2-calling" in y:
        fi = open(f"labels.txt", 'a+')
        fi.write(f"{y} 1 0\r\n")
        fi.close()
    elif "3-calling&smoking" in y:
        fi = open(f"labels.txt", 'a+')
        fi.write(f"{y} 1 1\r\n")
        fi.close()
    elif "4-others" in y:
        fi = open(f"labels.txt", 'a+')
        fi.write(f"{y} 0 0\r\n")
        fi.close()
    else:
        pass
# print(os.walk(r'data/'))
more_filepath = []
for filepath,dirnames,filenames in os.walk(r'data'):
    for filename in filenames:
        more_filepath.append(os.path.join(filepath, filename))
    # more_filepath.append(filepath+'/'+filenames)
pool = threadpool.ThreadPool(30)
requests = threadpool.makeRequests(creat_dataset, more_filepath)
[pool.putRequest(req) for req in requests]
pool.wait()
