from common.dataset_initial import INITIAL
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json
import csv
from xml.dom.minidom import parse
import xml.dom.minidom

classes = ['SURGERY', 'SIXTH', 'THREE', 'KYRO', 'HOBBY', 'PERFORMING', 'RAMADA', 'ALLEN', 'MAN', 'BREXTON', 'MISSION',
           'HOMEWOOD', 'LION', 'BRASSERIE', 'IRVING', 'BALTIMORE', 'WASHINGTON', 'PLAZA', 'VIETNAMESE', 'DAY', 'JEROME',
           'CHECK', 'GOMBEI', 'MOOSE', 'STATION', 'VILLAGE', 'SAIGON', 'YMCA', 'TIMKEN', 'FOR', 'LINCOLN', 'ROOM',
           'HESS', 'PINE', 'MARLBORO', 'QUIZNOS', 'MAGIC', 'INSTRUMENTS', 'HARBOR', 'TOWN', 'ECONO', 'GARAGE', 'PARK',
           'REDWOOD', 'JAS', 'BOOT', 'ZESCO', 'HILL', 'FUJI', 'TOWING', 'SHAKE', 'QUALITY', 'STORE', 'MUSIC', 'LARRY',
           'ORPHEUM', 'MUSEUM', 'FRANKLY', 'FRANCIS', 'COLONIAL', 'PARC', 'ORLEANS', 'LEON', 'HALL', 'GARNY', 'DONALD',
           'CINERAMA', 'UNIVERSITY', 'MIRAIDO', 'AIRWAYS', 'LULA', 'LIQUID', 'FITTING', 'SOTTO', 'MOSSER', 'FRANCISCO',
           'EVEREST', 'SHACK', 'SERVICES', 'ANNO', 'MUZEO', 'TRAVELODGE', 'TAQUERIA', 'INN', 'RADIO', 'SIGNS', 'NAILS',
           'APARTMENTS', 'MCKINNEY', 'PHOTO', 'PUFF', 'NEUMOS', 'TRIBUNALI', 'JEWELRY', 'DUNBAR', 'VISTA', 'THREADS',
           'DING', 'TEA', 'WHEELER', 'CHINESE', 'HOSPITAL', 'PICKLES', 'PLACE', 'DONG', 'CASTLE', 'SPEED', 'CHINATOWN',
           'RESTAURANT', 'EXPRESS', 'HARD', 'CHANTERELLE', 'SOL', 'DUNKIN', 'SAINTE', 'CAR', 'PUB', 'THAI', 'JOE',
           'IMPORTS', 'LOYAL', 'JONES', 'PAUL', 'CONEY', 'BILLIARDS', 'EYE', 'MACYS', 'RED', 'CROWNE', 'HERITAGE',
           'AVE', 'SAN', 'GAMESTOP', 'CENTRE', 'BACAR', 'SPAGHETTI', 'BROTHERS', 'LABRADOR', 'FACTORY', 'INDUSTRIAL',
           'BUDGET', 'MELANYA', 'DAILY', 'PASADA', 'WORK', 'FORUM', 'SAKS', 'ACCOUNTANCY', 'CIRCLE', 'POLISH', 'TIMES',
           'CAPOGIRO', 'SYMPHONY', 'BAR', 'SCORES', 'SPA', 'EYES', 'GLENOAK', 'LOUNGE', 'SUSHI', 'BODY', 'ANODIZING',
           'BELVEDERE', 'CASBAH', 'SOLEIL', 'TILE', 'BLUE', 'FOODS', 'ERNST', 'EAGLE', 'PARAMOUNT', 'GRILL', 'STORAGE',
           'SALLE', 'LOUIS', 'PARLIAMENT', 'MARBLE', 'ASTORIA', 'VERNON', 'SANDWICH', 'BARBER', 'WATERCOURSE', 'FORD',
           'LINE', 'DEPARTMENT', 'MOTIF', 'ARMY', 'JIMMY', 'ZULA', 'HOLIDAY', 'ICEBOX', 'RESORT', 'BREWERY', 'TWO',
           'CAPITOL', 'MAX', 'JOINT', 'MOUNTAIN', 'TARGET', 'CHIROPRACTIC', 'THE', 'TAPES', 'TAILORS', 'WARFIELD',
           'GRAND', 'MART', 'BICYCLE', 'JOHN', 'MEDICAL', 'JAPANESE', 'DETROIT', 'SOPRA', 'ROCK', 'BOB', 'TINY',
           'FOSSIL', 'THEATERS', 'BARNSLEY', 'FOX', 'GARIBYAN', 'VEGETARIAN', 'MILLS', 'REI', 'TRIPLE', 'MARU',
           'WHITCOMB', 'PANCAKE', 'JEWEL', 'HAMBURGERS', 'MAGNOLIA', 'PACIFIC', 'HAHM', 'EMBASSY', 'SENTINEL', 'MASSA',
           'WINDSOR', 'CAFE', 'LOLA', 'PHIL', 'HOLLYWOOD', 'PRODUCTS', 'WARWICK', 'JUST', 'WHOLE', 'VINE', 'DOLL',
           'SPORTIQUE', 'JUKE', 'ANAHEIM', 'TIJUANA', 'BURBANK', 'DALLAS', 'BENDERS', 'SEMINARY', 'WILLIAMS', 'SAVE',
           'GARDEN', 'EXECUTIVE', 'SUITES', 'TAVERN', 'SAKANA', 'BRAZIL', 'BOX', 'STREET', 'ORION', 'INSURANCE', 'WALL',
           'DIEGO', 'QUINN', 'SHOE', 'JOBSITE', 'HYATT', 'ANGELO', 'PLAYS', 'PHARMACY', 'MODIFIED', 'PIZZA', 'COLORADO',
           'FLORAL', 'CREST', 'DODGE', 'SING', 'MASSAGE', 'CITY', 'LIGHTS', 'GHIRARDELLI', 'AGENCY', 'METHODIST',
           'PIKE', 'SPRUCE', 'DAYS', 'GIULIO', 'MEMORIAL', 'TEXTILES', 'COURTYARD', 'YARD', 'STEAK', 'BOHEMIAN', 'JEAN',
           'COLLEGE', 'RENT', 'SUPPLY', 'SIMON', 'CARLTON', 'ANTIQUE', 'AMERICANIA', 'BROWN', 'AUTO', 'CARROLL',
           'FORTUNE', 'MACY', 'PORTAL', 'MOON', 'VIDEO', 'BANGKOK', 'STAR', 'UPTOWN', 'TABU', 'AVENUE', 'WINES',
           'AMOEBA', 'AUTOMOTIVE', 'ROCKY', 'SUPER', 'GARDENS', 'GREAT', 'HEALTH', 'GENIES', 'CRAFTS', 'THAIRISH',
           'TOWER', 'COFFEE', 'TIMBERLINE', 'TEN', 'GIFTS', 'WHITE', 'MOORE', 'ISLAND', 'PALACE', 'HORTON', 'ART',
           'OSCO', 'VALUE', 'MARKET', 'FIVE', 'CALIFORNIA', 'GOODWILL', 'EVERYDAY', 'CONVENTION', 'DOLLAR', 'SIX',
           'OLIN', 'ENTERPRISE', 'INC', 'PAYLESS', 'TARA', 'SHELL', 'BIERSCH', 'PUBLIC', 'DAN', 'QUILTS', 'BAYSIDE',
           'STUFF', 'LENSCRAFTERS', 'FUDDRUCKERS', 'BREW', 'MOBILE', 'SHOGUN', 'BEST', 'ZOU', 'CANTO', 'WALNUT',
           'CHEUVRONT', 'DESIGN', 'CITYPLACE', 'MEAT', 'CINEMA', 'CIRCLES', 'STEWART', 'NICK', 'HOUSE', 'DOMINO',
           'PORTLAND', 'MALL', 'AVANTE', 'THEOLOGICAL', 'HIGGINS', 'SUN', 'PLAYERS', 'NOB', 'COMFORT', 'GALLERY',
           'BAND', 'RECORDS', 'MOUNTS', 'UNITED', 'MODERN', 'ZERO', 'ALDEN', 'KYLE', 'MASTER', 'AMERICA', 'THEATRE',
           'FIFTH', 'GLENOAKS', 'AND', 'MICHAEL', 'BELLA', 'LODGE', 'WORLD', 'COLLAR', 'MINT', 'MICHOACANA', 'MAI',
           'SUBWAY', 'JOES', 'CLUB', 'DELI', 'FOUR', 'GORDON', 'KABOB', 'BOTTLE', 'PHYSICIANS', 'MET', 'LUCKY', 'FOOD',
           'BANK', 'SCOOTERS', 'GROCERY', 'GARFIELD', 'BLACK', 'TRADER', 'HOUSTON', 'NIGHT', 'CVS', 'TIRE', 'DAVID',
           'LAURENCE', 'LOCKSMITH', 'STATES', 'CLEANERS', 'JAMES', 'HUT', 'CHERRY', 'JOSE', 'STANDARD', 'KITCHEN',
           'DAKAO', 'DELICIOUS', 'BLOCKBUSTER', 'ANTHROPOLOGIE', 'FISHMAN', 'PAK', 'FABRICS', 'SUBS', 'FLATS',
           'CENTRAL', 'SUB', 'LITTLE', 'COMPLEXIONS', 'HIMALAYAS', 'HOTEL', 'SHOP', 'SALADS', 'DONUTS', 'OYSTER',
           'PHELPS', 'TRAX', 'EAGER', 'WYNDHAM', 'EXCURSION', 'VIA', 'SCIENTOLOGY', 'BURMA', 'LABYRINTH', 'SPACE',
           'CITYARTS', 'KOTAYK', 'CROCE', 'WAY', 'YASDA', 'WESTERN', 'PGE', 'HOULIHAN', 'BIKE', 'MURPHY', 'BEADS',
           'AIR', 'SCHOOL', 'INDIANA', 'FEET', 'BOOKS', 'POLICE', 'GRANDSTAND', 'ZOLL', 'BAKERY', 'MOTORSPORTS',
           'OUTDOOR', 'HATSU', 'DOOR', 'GRANT', 'CHINA', 'ARTS', 'ASU', 'THEATER', 'GASLAMP', 'SOURCE', 'CRUSH', 'AVIS',
           'KFC', 'STARBUCKS', 'WINBRO', 'CUSTOM', 'CHASE', 'FIREPLACE', 'ORTLIEB', 'PARKING', 'HERTZ', 'REPERTORY',
           'SAINT', 'WAX', 'TRIDENT', 'FAHRENHEIT', 'YOUNG', 'YANK', 'HAPPINESS', 'CLEANING', 'CARR', 'CORTEZ',
           'SHERATON', 'CLAIRE', 'CHURCH', 'COMMON', 'CORNER', 'LIVING', 'MOBIL', 'ORLANDO', 'POMPEI', 'FIRST', 'BENTO',
           'ROOTS', 'WAREHOUSE', 'CENTER', 'FLORIST', 'CAMDEN', 'ORIGINAL', 'GIGI', 'SOUTH', 'BOOKSTORE', 'UNDER',
           'DOLAN', 'WINE', 'SUPERSTAR', 'MARRIOTT', 'SEASONS', 'ARCADE', 'ZONE', 'DOMINI', 'CHIAPPARELLI', 'SQUARE',
           'FITNESS', 'MARTIN', 'PIONEER', 'CHOCOLATE', 'HIGH', 'THERAPY', 'CELLARS', 'NAPASORN', 'SHEA', 'SATYRICON',
           'POSTERS']

root_path = "G:\\download_dataset\\StreetViewText\\svt\\svt1"
train_label_path = os.path.join(root_path, "train.xml")
test_label_path = os.path.join(root_path, "test.xml")
config.timeout = 100
config.max_retries = 10
dataset_name = "StreetViewText"


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


initial = INITIAL(root_path, dataset_name, ["BOX2D"], classes)
gas, dataset = initial.generate_catalog()

for split in ["train", "test"]:
    segment = dataset.create_segment(split)
    classes, imgs_label = label_acqure(os.path.join(root_path, split + ".xml"))
    imgs = imgs_label.keys()
    for img in imgs:
        img_path = os.path.join(root_path, "img\\"+img)
        data = Data(img_path)
        img_labels = imgs_label[img]
        data.label.box2d = []
        for labels in img_labels:
            xmin = float(labels[0])
            ymin = float(labels[1])
            xmax = float(labels[2])+xmin
            ymax = float(labels[3])+ymin
            data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                 category=labels[4]))
        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
