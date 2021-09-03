import cv2
import os
import numpy as np
root_path = "C:\\Users\\52572\Desktop\\tmp"
imgs = os.listdir("C:\\Users\\52572\Desktop\\tmp")
# print(imgs)
img1 = cv2.imread(os.path.join(root_path, imgs[0]))

img1 = cv2.resize(img1, (700, 700))[200:500, 200:500]

empty = np.zeros((300, 10, 3),dtype="uint8")
empty = empty+255

empty2= np.zeros((10, 920, 3),dtype="uint8")
empty2 = empty2+255


img2 = cv2.imread(os.path.join(root_path, imgs[1]))
img2 = cv2.resize(img2, (700, 700))[200:500, 200:500]

img3 = cv2.imread(os.path.join(root_path, imgs[2]))
img3 = cv2.resize(img3, (700, 700))[200:500, 200:500]

img4 = cv2.imread(os.path.join(root_path, imgs[3]))
img4 = cv2.resize(img4, (700, 700))[200:500, 200:500]

img5 = cv2.imread(os.path.join(root_path, imgs[4]))
img5 = cv2.resize(img5, (700, 700))[200:500, 200:500]

img6 = cv2.imread(os.path.join(root_path, imgs[5]))
img6 = cv2.resize(img6, (700, 700))[200:500, 200:500]

result1 = np.hstack((img1,empty, img2,empty, img3))
result2 = np.hstack((img4,empty, img5,empty, img6))
result = np.vstack((result1, empty2, result2))
# result = np.hstack((img1,empty ))
# print(result.shape)

#
cv2.imshow("img",result)
cv2.waitKey()
cv2.destroyAllWindows()