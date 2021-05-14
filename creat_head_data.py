import os
import threadpool
import cv2

def crop_img(file_path):
    img = cv2.imread(file_path)
    height,width = img.shape[:2]
    print(height,width)
    ratio = width/height
    head_height = height*(ratio/1)
    roi = img[0:int(head_height),0:width]
    cv2.imwrite(file_path,roi)

files_list = []
for root,dirs,files in os.walk("./data_hh"):
    for file in files:
        img_path = os.path.join(root, file)
        files_list.append(img_path)    
# root,dirs,files = os.walk("./data")
pool = threadpool.ThreadPool(30)
requests = threadpool.makeRequests(crop_img, files_list) 
[pool.putRequest(req) for req in requests] 
pool.wait()