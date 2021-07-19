from tensorbay import GAS
from tensorbay.dataset import Dataset, Data
from config import *
from category import catalog
from tensorbay.label import LabeledBox2D
import os
from common.label_acquire import acquire_label_xml
from common.dataset_initial import initial
import os
import csv
import json

root_path = "G:\\shannon\\deeplearning_dataset\\openlogo\\openlogo\\openlogo~\\openlogo"
labels = os.listdir(os.path.join(root_path, "Annotations"))
imgs = os.listdir(os.path.join(root_path, "JPEGImages"))

labels_no = [i.split(".")[0] for i in labels]
imgs_no = [i.split(".")[0] for i in imgs]

dataset_name = "Openlogo"
a = ['guinness', 'dhl', 'stellaartois', 'fosters', 'becks', 'nvidia', 'shell', 'corona', 'ford', 'google', 'aldi',
     'pepsi', 'apple', 'fedex', 'starbucks', 'singha', 'heineken', 'carlsberg', 'paulaner', 'ferrari', 'ups',
     'cocacola', 'erdinger', 'tsingtao', 'texaco', 'esso', 'chimay', 'bmw', 'adidas', 'rittersport', 'hp', 'milka',
     'adidas_text', '3m', 'heineken_text', 'abus', 'accenture', 'adidas1', 'airhawk', 'honda_text', 'aldi_text',
     'target_text', 'walmart_text', 'subway', 'amazon', 'alfaromeo', 'allett', 'allianz', 'allianz_text', 't-mobile',
     'aluratek', 'aluratek_text', 'playstation', 'amcrest', 'amcrest_text', 'americanexpress', 'americanexpress_text',
     'mastercard', 'visa', 'android', 'anz', 'anz_text', 'apc', 'apecase', 'aquapac_text', 'aral', 'audi', 'mcdonalds',
     'mercedesbenz', 'lotto', 'volkswagen', 'honda', 'ec', 'armani', 'armitron', 'aspirin', 'bayer', 'asus', 'athalon',
     'audi_text', 'michelin', 'bosch_text', 'mercedesbenz_text', 'lexus', 'axa', 'bacardi', 'bankofamerica',
     'bankofamerica_text', 'kodak', 'samsung', 'barbie', 'barclays', 'basf', 'batman', 'bbc', 'bbva', 'tnt', 'umbro',
     'standard_liege', 'gucci', 'dexia', 'puma', 'puma_text', 'nike', 'base', 'citroen_text', 'total', 'bfgoodrich',
     'kia', 'citroen', 'quick', 'carglass', 'airness', 'bik', 'reebok1', 'bridgestone_text', 'us_president',
     'bridgestone', 'bellataylor', 'bellodigital', 'bellodigital_text', 'bem', 'twitter', 'youtube', 'benrus',
     'bershka', 'bionade', 'blackmores', 'blizzardentertainment', 'volvo', 'mini', 'toyota', 'facebook', 'intel',
     'teslamotors', 'boeing', 'boeing_text', 'unicef', 'bosch', 'porsche', 'porsche_text', 'opel', 'siemens',
     'panasonic', 'bottegaveneta', 'budweiser', 'budweiser_text', 'corona_text', 'bulgari', 'burgerking',
     'burgerking_text', 'pizzahut', 'costa', 'mcdonalds_text', 'calvinklein', 'canon', 'sony', 'carters', 'cartier',
     'caterpillar', 'chanel', 'chanel_text', 'prada', 'chevrolet', 'chevrolet_text', 'subaru', 'chevron', 'chickfila',
     'chiquita', 'cisco', 'rbc', 'citi', 'coach', 'coke', 'sprite', 'gap', 'hyundai', 'colgate', 'comedycentral',
     'converse', 'costco', 'homedepot_text', 'cpa_australia', 'cvs', 'cvspharmacy', 'danone', 'disney', 'drpepper',
     'dunkindonuts', 'ebay', 'espn', 'esso_text', 'mobil', 'tigerwash', 'spar', 'spar_text', 'shell_text', 'renault',
     'evernote', 'nissan', 'firefox', 'nbc', 'redbull', 'vodafone', 'yahoo', 'fly_emirates', 'fritolay', 'fritos',
     'cheetos', 'doritos', 'lays', 'ruffles', 'sunchips', 'tostitos', 'pepsi_text', 'lg', 'generalelectric', 'gildan',
     'gillette', 'venus', 'walmart', 'goodyear', 'hanes', 'head', 'head_text', 'heraldsun', 'hermes', 'hersheys',
     'kitkat', 'reeses', 'hh', 'hisense', 'hm', 'jcrew', 'homedepot', 'hsbc', 'hsbc_text', 'rolex', 'huawei_text',
     'huawei', 'sap', 'hyundai_text', 'ikea', 'ibm', 'at_and_t', 'windows', 'internetexplorer', 'jackinthebox',
     'jacobscreek', 'jagermeister', 'johnnywalker', 'jurlique', 'kelloggs', 'lego', 'kfc', 'suzuki', 'kraft', 'jello',
     'maxwellhouse', 'miraclewhip', 'philadelphia', 'planters', 'velveeta', 'lacoste', 'lacoste_text', 'lamborghini',
     'levis', 'lexus_text', 'londonunderground', 'loreal', 'lv', 'luxottica', 'marlboro', 'marlboro_text',
     'marlboro_fig', 'maserati', 'maxxis', 'mccafe', 'philips', 'medibank', 'microsoft', 'millerhighlife', 'mitsubishi',
     'mk', 'motorola', 'mtv', 'nasa', 'nb', 'nescafe', 'netflix', 'nike_text', 'nintendo', 'nissan_text', 'infiniti',
     'infiniti_text', 'nivea', 'northface', 'obey', 'olympics', 'optus', 'optus_yes', 'oracle', 'pampers',
     'pepsi_text1', 'pizzahut_hut', 'poloralphlauren', 'recycling', 'redbull_text', 'firelli', 'santander_text',
     'reebok_text', 'rolex_text', 'reebok', 'tommyhilfiger', 'republican', 'zara', 'santander', 'schwinn', 'sega',
     'shell_text1', 'select', 'skechers', 'vaio', 'soundcloud', 'soundrop', 'spiderman', 'superman', 'supreme',
     'tacobell', 'target', 'toyota_text', 'thomsonreuters', 'timberland', 'tissot', 'verizon', 'scion_text',
     'underarmour', 'uniqlo', 'uniqlo1', 'unitednations', 'verizon_text', 'volkswagen_text', 'warnerbros', 'wellsfargo',
     'wellsfargo_text', 'wii', 'williamhill', 'wordpress', 'xbox', 'yamaha', 'yonex_text', 'yonex']
# print(len(a))
initial = initial(root_path, dataset_name, "box2d", a)
gas, dataset = initial.generate_catalog()

segment = dataset.create_segment("train & val & test")
for img in imgs:
    path = os.path.join(os.path.join(root_path, "JPEGImages"), img)
    if img.split(".")[0] in labels_no:
        img_label = acquire_label_xml(os.path.join(os.path.join(root_path, "Annotations"), img.split(".")[0] + ".xml"))
    else:
         img_label =[]
    data = Data(path)
    if len(img_label) != 0:
         data.label.box2d = []
         for i in range(len(img_label)):
              xmin = img_label[i][0]
              ymin = img_label[i][1]
              xmax = img_label[i][2]
              ymax = img_label[i][3]
              data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                             category=img_label[i][4],
                                                             # attributes={"occluded": box["occluded"]}))
                                                             ))
    segment.append(data)

dataset_client = gas.upload_dataset(dataset,jobs=8)
dataset_client.commit("Initial commit")



