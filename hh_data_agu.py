# import os 
# from shutil import copyfile

# for filepath,dirnames,filenames in os.walk(r'Objects365_V2/objects365/images/'):
#     for filename in filenames:
#         print(filename)    
#         copyfile(os.path.join(filepath,filename),f"person-car-airplane-ob/images/{filename}")



import os
import os.path
import cv2
import threadpool
from shutil import copyfile
import numpy as np

def rotate_bound(image, angle):
    '''
    图像旋转不截断方法
    :param image:
    :param angle:111111111111111111
    :return:
    '''
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def aaa(filename):
    # print(filename)
    global images_path,cout
    img = cv2.imread(f"{images_path}{filename}")
    
    for x in range(-10,10,3):
        if x !=0:
            img = rotate_bound(img,x)
            cv2.imwrite(f"{images_path}{x}-{filename}",img)
    # copyfile(f"person-car-airplane/images/{filename}",f"person-car-airplane/data/images/{filename}")
    # copyfile(f"person-car-airplane/labels/{filename}.txt",f"person-car-airplane/data/labels/{file_idx}.txt")
    # print(filename)
    # img = cv2.imread(f"person-car-airplane/ob365/images/{file_idx}.jpg")

def bbb(filename):
    # print(filename)
    global images_path,cout
    img = cv2.imread(f"{images_path}{filename}")
    img = cv2.flip(img,1)
    cv2.imwrite(f"{images_path}flip{filename}",img)
# cout = 0
images_path="data_hh/3-calling&smoking/"
# labels_path="person-car-airplane/labels/"
files=os.listdir(images_path)  #得到文件夹下所有文件名称
pool = threadpool.ThreadPool(30)
requests = threadpool.makeRequests(aaa, files) 
[pool.putRequest(req) for req in requests] 
pool.wait()