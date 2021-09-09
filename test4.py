import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# img1 = cv2.imread("D:\download_dataset\FSS1000\\fewshot_data\\fewshot_data\\ab_wheel\\1.png", 0)
img1 = Image.open("D:\download_dataset\FSS1000\\fewshot_data\\fewshot_data\\ab_wheel\\1.png")
array = np.array(img1, dtype=int)
img1 = array[:, :, 0]
img1[img1 != 0] = 1
print(img1)

# for i in range(img1.shape[0]):
#     for j in range(img1.shape[1]):
#         if img1[i, j] != 0:
#             img1[i, j] = 1
# img1 = np.zeros((200, 200))
Image.fromarray(np.uint16(img1)).save(
    "D:\download_dataset\FSS1000\\fewshot_data\\fewshot_data\\abacus\\1mask.png")
# img2.save("D:\download_dataset\FSS1000\\fewshot_data\\fewshot_data\\abacus\\1mask.png")
