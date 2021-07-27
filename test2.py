import cv2

from xml.dom.minidom import parse
import xml.dom.minidom


def label_acqure(label_path):
    DOMTree = xml.dom.minidom.parse(label_path)
    collection = DOMTree.documentElement
    imgs = collection.getElementsByTagName("image")
    imgs_label = {}
    classes = []
    for img in imgs:
        imgName = img.getElementsByTagName("imageName")[0].childNodes[0].data.split("/")[1]
        address = img.getElementsByTagName("address")[0].childNodes[0].data
        taggedRectangles = img.getElementsByTagName("taggedRectangle")
        imgs_label[imgName] = []
        # imgs_label[imgName].append(address)
        for rec in taggedRectangles:
            tag = rec.getElementsByTagName("tag")[0].childNodes[0].data
            height = rec.getAttributeNode("height").childNodes[0].data
            width = rec.getAttributeNode("width").childNodes[0].data
            x = rec.getAttributeNode("x").childNodes[0].data
            y = rec.getAttributeNode("y").childNodes[0].data
            imgs_label[imgName].append([x, y, width, height, tag])
            if tag not in classes:
                classes.append(tag)
    return classes, imgs_label


classes, imgs_label = label_acqure(
    "G:\download_dataset\StreetViewText\svt\svt1\\test.xml"
)

a = imgs_label["00_05.jpg"]

print(a)
img = cv2.imread("G:\\download_dataset\\StreetViewText\\svt\\svt1\\img\\00_05.jpg")
cv2.rectangle(img, (img[0][0], img[0][1]), (img[0][0] + img[0][2], img[0][1] + img[0][3]), (255, 255, 0))
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
