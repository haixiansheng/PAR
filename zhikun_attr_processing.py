#encoding=utf-8
#人："\u4eba"
#车：\u8f66

'''
0:airplane
1:person
2:car

'''
import json
from shutil import copyfile
import os
import cv2
import threadpool 
import numpy as np

# def draw_react(img,bbox,cls,save_path):
#     img = cv2.imread(img_path)
#     cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]),(0,255,0)),2)
#     cv2.imwrite(f"{save_path}/{img_path})

# ac,pc,cc = 0,0,0
def data_process(y):
    
    try:
        type_label = np.zeros((2,), dtype=np.int)
        #类型标签设置[摩托车,SUV,卡车,公交车,面包车,小轿车,其他]
        if "2calling" in y:#摩托车
            type_label[0] = 1
        elif "1smoking" in y:
            type_label[1] = 1
        elif "3smoking_calling" in y:#卡车
            type_label[0] = 1
            type_label[1] = 1
        elif "4others" in y:#卡车
            type_label[0] = 0
            type_label[1] = 0
        else:
            print("rthbrthrt")
            pass

        #颜色标签设置[黑,棕,蓝,灰,红,白,黄,紫,未知]
        
        #写入label文件
        fi = open(f"zhikun_attr/labels.txt",'a+')
        fi.write(f"./{y} "+" ".join([str(x) for x in type_label])+"\r\n")
        fi.close()
        # print(type_label)
    except:
        print(filename)


file_path = []
for dirpath,dirnames,filenames in os.walk(r'zhikun_attr'):
    for filename in filenames:
        file_path.append(os.path.join(dirpath,filename))
print("file nums:",len(file_path))
pool = threadpool.ThreadPool(40)
requests = threadpool.makeRequests(data_process, file_path) 
[pool.putRequest(req) for req in requests] 
pool.wait() 