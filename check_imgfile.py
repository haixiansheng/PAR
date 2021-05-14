import os
from PIL import Image
import threadpool

# for dirpath,dirnames,filenames in os.walk(r'car_type_data'):
#         n = 0
#         for filename in filenames:
#             n += 1
#             if n==1:
#                 print (os.path.join(dirpath,filename))
            
def aaa(img_name):
    try:
        img = Image.open(img_name)
        print("yes ",img_name)
    except:
        os.remove(img_name)
        print("no,I will remove ",img_name)

f = open("data/train.txt",'r')
lines=f.readlines()  #读取整个文件所有行，保存在 list 列表中
img_paths = []
for line in lines:
    img_path = line.split(" ")[0]
    img_paths.append(img_path)

pool = threadpool.ThreadPool(50)
requests = threadpool.makeRequests(aaa, img_paths) 
[pool.putRequest(req) for req in requests] 
pool.wait() 