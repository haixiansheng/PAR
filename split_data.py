import os
import os.path
import cv2



path = "labels.txt"
# files = os.listdir(path)  # 得到文件夹下所有文件名称


inFile = open(f"{path}", 'r')  # 以只读方式打开标签文件文件
print(len(inFile))
# 读每一行的bbox的数据，并画框
for line in inFile:
    trainingSet = line.split(' ')  # 对于每一行，按','把数据分开，这里是分成两部分
    cls, x, y, w, h = int(trainingSet[0]), float(trainingSet[1]), float(trainingSet[2]), float(
        trainingSet[3]), float(trainingSet[4])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (int(width * x) - int(width * w) // 2, int(height * y) - int(height * h) // 2),
                  (int(width * x) + int(width * w) // 2, int(height * y) + int(height * h) // 2), (0, 255, 0), 2)
    cv2.putText(img, f'{cls}', (int(width * x), int(height * y)), font, 1, (0, 0, 255), 1)
cv2.imwrite(f"reac/{file_idx}.jpg", img)
