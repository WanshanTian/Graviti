from xml.dom.minidom import parse
import xml.dom.minidom


def acquire_label_xml(img_path) -> []:
    DOMTree = xml.dom.minidom.parse(img_path)
    collection = DOMTree.documentElement
    boundingbox = collection.getElementsByTagName("object")
    img_lable = []
    for i in boundingbox:
        tmp = []
        category = i.getElementsByTagName("name")[0].childNodes[0].data
        tmp.append(float(
            [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmin")][0]))
        tmp.append(float(
            [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymin")][0]))
        tmp.append(float(
            [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmax")][0]))
        tmp.append(float(
            [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymax")][0]))
        tmp.append(category)
        img_lable.append(tmp)
    return img_lable
