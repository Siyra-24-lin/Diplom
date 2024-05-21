import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_hub as hub
import json
import time
import base64
import requests
import threading
import cv2
from googletrans import Translator
import numpy as np
import math
import sys
import random
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import (QLineEdit, QLabel, QComboBox, QRadioButton, QFileDialog, QColorDialog,
                               QSpinBox, QListWidget)
from PySide6.QtGui import QPixmap
from pathlib import Path
import pandas as pd
import zipfile
import shutil

# TODO: progressBars?????????????????
# TODO: убрать блики/тени????????????? (отнесено к дальнейшим планам разработки)
# TODO: сделать сравнение результатов DeepLab до и после дообучения


def translate_text(text):
    translator = Translator()
    translated_ita = translator.translate(text, src='ru', dest='en')
    tr_obj = translated_ita.text
    return tr_obj


# ------------------------------------ГЕНЕРАЦИЯ---------------------------------------------------
class Text2ImageAPI:
    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_model(self):
        response = requests.get(self.URL + 'key/api/v1/models', headers=self.AUTH_HEADERS)
        data = response.json()
        return data[0]['id']

    def generate(self, prompt, model, images=1, width=512, height=512):
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "style": "UHD",
            "negativePromptUnclip": "неравномерное освещение, тени, блики, блеск, капли, неравномерная яркость,\
                                            неравномерный задний план, объём",
            "generateParams": {
                "query": f"{prompt}"
            }
        }

        data = {
            'model_id': (None, model),
            'params': (None, json.dumps(params), 'application/json')
        }
        response = requests.post(self.URL + 'key/api/v1/text2image/run', headers=self.AUTH_HEADERS, files=data)
        data = response.json()
        return data['uuid']

    def check_generation(self, request_id, attempts=50, delay=50):
        while attempts > 0:
            response = requests.get(self.URL + 'key/api/v1/text2image/status/' + request_id, headers=self.AUTH_HEADERS)
            data = response.json()
            if data['status'] == 'DONE':
                return data['images']

            attempts -= 1
            time.sleep(delay)


def generate_images(api, promt, i, model_id, folder, imsize):
    uuid = api.generate(promt, model_id, width=imsize, height=imsize)
    images = api.check_generation(uuid)
    img_b = base64.b64decode(images[0])
    # запись в папку
    with open(f'{folder}/image{i}.png', 'wb') as f:
        f.write(img_b)

    image = cv2.imread(f'{folder}/image{i}.png')
    res = rotate_img(image)
    cv2.imwrite(f'{folder}/image{i}.png', res)

    print(f'картинка с номером {i} была сгенерирована.')


def _generate_(obj, img_count, promts, imsize):
    api = Text2ImageAPI('https://api-key.fusionbrain.ai/', '8EDD04A0A3339391D3DA04FC947801CB',
                        'F29E43D7A9EB9C60C34C1B45ABFF2104')
    model_id = api.get_model()

    # создаём папку, если такой нет
    tr_obj = translate_text(obj)

    path_orig = f'images/{tr_obj}/original'
    if not (os.path.exists(path_orig)):
        os.makedirs(path_orig)

    if img_count < 600:
        n = 5
    else:
        n = 200  # число разбиения потоков

    count = math.trunc(img_count / n)
    mod = img_count % n

    num = -1

    if img_count >= n:
        for j in range(n):
            for i in range(count):
                num = i + j * count
                print(f"image{num} started to generate: {promts[num]}")
                thread = threading.Thread(target=generate_images,
                                          args=(api, promts[num], num, model_id, path_orig, imsize))
                thread.start()
            time.sleep(25)

    if mod != 0:
        for i in range(num + 1, num + mod + 1):
            print(f"image{i} started to generate: {promts[i]}")
            thread = threading.Thread(target=generate_images, args=(api, promts[i], i, model_id, path_orig, imsize))
            thread.start()
    # -----------------------------------------------------------------------------------------------------------------


def generate_promts(obj, img_count):
    promts = []
    colors = {
        # 'бордовый': [-1],
        # 'тёмно-зелёный': [-2],
        # 'тёмно-синий': [-3, -6],
        # 'чёрный': [-5, -8]
        'чёрный': [-1]
    }
    back_color = {
        # 'бежевый': [6],
        # 'белый': [1, 2, 3, 5],
        # 'светло-жёлтый': [8],
        # 'ярко-белый': [1, 2],

        'светло-розовый': [1],
        'белый': [1],
        'ярко-белый': [1],
        'нежно-розовый': [1]
    }
    for i in range(img_count):
        color = random.choice(list(colors.keys()))
        ind = random.choice(colors[color])
        # print(f'{color}: {ind}')
        ncolors = []
        # print(colors.items())
        for key, val in back_color.items():
            for v in val:
                if v == -ind:
                    ncolors.append(key)
        back = random.choice(ncolors)
        # promt = f"матовый {color} {obj} на матовом {back} фоне."
        promt = f"матовый {back} {obj} на матовом {color} фоне."
        promts.append(promt)
    return promts
# ------------------------------------------------------------------------------------------------


# --------------------------------------АЛГОРИТМЫ-------------------------------------------
def write_img(path, img_name, func_name, image):
    cv2.imwrite(f'{path}_{func_name}\\{img_name}.png', image)


def grey_img(image):
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_grey


def otsu_threshold(image, mcolor):
    grey_image = grey_img(image)
    ret, otsu = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    mask = image.copy()
    mask[0:image.shape[0], 0:image.shape[0]] = [0, 0, 0]

    for i in range(otsu.shape[0]):
        for j in range(otsu.shape[1]):
            if otsu[i, j] != 0:
                mask[i, j] = list(mcolor)
    return mask


def watershed_img(image, mcolor):
    src = image
    img = src.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Устранение дырок
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=15)
    # Расширение
    sure_bg = cv2.dilate(closing, kernel, iterations=5)
    # Преобразование расстояния
    sure_fg = cv2.erode(closing, kernel, iterations=5)
    # Получить неизвестную область
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    # Отметка
    ret, markers1 = cv2.connectedComponents(sure_fg)
    # Убедитесь, что фон равен 1, а не 0
    markers2 = markers1 + 1
    # Неизвестная область отмечена как 0
    markers2[unknown == 255] = 0
    # cont = np.zeros(image.shape)
    markers = cv2.watershed(img, markers2)
    labels = np.unique(markers)

    coins = []
    for label in labels[2:]:
        target = np.where(markers == label, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        coins.append(contours[0])
    cont = img
    cont[0:image.shape[0], 0:image.shape[0]] = [0, 0, 0]
    cv2.drawContours(cont, coins, -1, color=mcolor, thickness=-2)
    return cont


def kmeans_img(image, mcolor):
    z = image.reshape((-1, 3))
    # convert to np.float32
    z = np.float32(z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(image.shape)

    pix = list(res2[10, 10])
    img = res2

    for i in range(res2.shape[0]):
        for j in range(res2.shape[1]):
            if (list(res2[i, j])[0] > pix[0]) and (list(res2[i, j])[1] > pix[1]) and (list(res2[i, j])[2] > pix[2]):
                pix = list(res2[i, j])
                break

    for i in range(res2.shape[0]):
        for j in range(res2.shape[1]):
            if list(res2[i, j]) == pix:
                img[i, j] = [0, 0, 0]
            else:
                img[i, j] = list(mcolor)
    return img


def bound_box_otsu(image, mcolor):
    thresh_image = otsu_threshold(image, (255, 255, 255))
    thresh_image = grey_img(thresh_image)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxl = 0
    sel_contour = None
    for contour in contours:
        if contour.shape[0] > maxl:
            sel_contour = contour
            maxl = contour.shape[0]

    x, y, w, h = cv2.boundingRect(sel_contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), mcolor, 8)
    return image, [x, y, w, h]


def bound_box_kmeans(image, mcolor):
    kmeans = kmeans_img(image, (255, 255, 255))

    contours, hierarchy = cv2.findContours(grey_img(kmeans), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxl = 0
    sel_countour = None
    for countour in contours:
        if countour.shape[0] > maxl:
            sel_countour = countour
            maxl = countour.shape[0]
    x, y, w, h = cv2.boundingRect(sel_countour)
    cv2.rectangle(image, (x, y), (x + w, y + h), mcolor, 8)
    return image, [x, y, w, h]


def bound_box_watershed(image, mcolor):
    wsh_mask = watershed_img(image, (255, 255, 255))

    thresh_image = grey_img(wsh_mask)
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxl = 0
    sel_countour = None
    for countour in contours:
        if countour.shape[0] > maxl:
            sel_countour = countour
            maxl = countour.shape[0]
    x, y, w, h = cv2.boundingRect(sel_countour)
    cv2.rectangle(image, (x, y), (x + w, y + h), mcolor, 8)
    return image, [x, y, w, h]


def rotate_img(image):
    ind = random.choice([0, 1, 2, 3])
    if ind == 0:
        res = image
    elif ind == 1:
        res = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif ind == 2:
        res = cv2.rotate(image, cv2.ROTATE_180)
    else:
        res = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return res


def th_otsu(path, img_name, image, mcolor):
    otsu = otsu_threshold(image, mcolor)
    write_img(path, img_name, 'otsu', otsu)


def th_wsh(path, img_name, image, mcolor):
    wsh_mask = watershed_img(image, mcolor)
    write_img(path, img_name, 'wsh_mask', wsh_mask)


def th_kmn(path, img_name, image, mcolor):
    kmeans = kmeans_img(image, mcolor)
    write_img(path, img_name, 'kmeans', kmeans)


def th_box_otsu(path, img_name, image, mcolor):
    box_otsu, _coor = bound_box_otsu(image, mcolor)
    write_img(path, img_name, 'box_otsu', box_otsu)
    with open(f"{path}/_box_otsu/coordinates/{img_name}_coordinates.txt", "w") as file:
        file.write(f"x:{_coor[0]}, y:{_coor[1]}, w:{_coor[2]}, h:{_coor[3]}")


def th_box_kmeans(path, img_name, image, mcolor):
    box_kmeans, _coor = bound_box_kmeans(image, mcolor)
    write_img(path, img_name, 'box_kmeans', box_kmeans)
    with open(f"{path}/_box_kmeans/coordinates/{img_name}_coordinates.txt", "w") as file:
        file.write(f"x:{_coor[0]}, y:{_coor[1]}, w:{_coor[2]}, h:{_coor[3]}")


def th_box_watershed(path, img_name, image, mcolor):
    box_watershed, _coor = bound_box_watershed(image, mcolor)
    write_img(path, img_name, 'box_watershed', box_watershed)
    with open(f"{path}/_box_watershed/coordinates/{img_name}_coordinates.txt", "w") as file:
        file.write(f"x:{_coor[0]}, y:{_coor[1]}, w:{_coor[2]}, h:{_coor[3]}")


def th_rotate(path, img_name, image):
    rotate = rotate_img(image)
    write_img(path, img_name, 'rotate', rotate)


# ------------------------------------------------------------------------------------------
# -------------------------------------СТИЛИЗАЦИЯ-------------------------------------------
def img_scaler(image, max_dim):
    original_shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    scale_ratio = max_dim / max(original_shape)
    new_shape = tf.cast(original_shape * scale_ratio, tf.int32)
    return tf.image.resize(image, new_shape)


def load_img(path_to_img, imsize):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img_scaler(img, imsize)
    return img[tf.newaxis, :]


def stylized_img(i, style_image, hub_module, obj, imsize):
    content_path = f'images/{obj}/original/image{i}.png'
    content_image = load_img(content_path, imsize)
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    tf.keras.preprocessing.image.save_img(f'images/{obj}/stylized/image{i}.png', stylized_image[0])
    print(f'image{i} was stylized.')
# ------------------------------------------------------------------------------------------


# -------------------------------ОБЪЕДИНЕНИЕ КЛАССОВ----------------------------------------
def cut_paste_box(paste_img, back, mcolor, mask_alg):
    # размер уменьшения объекта
    sizes = [128, 256, 512]
    size = random.choice(sizes)

    # смешение объекта верт.+гориз.
    if size == 512:
        k = 0
        n = 0
    else:
        k = random.choice(np.arange(size))
        n = random.choice(np.arange(size))

    # меняем размер объекта
    paste_img = cv2.resize(paste_img, [size, size], cv2.INTER_AREA)

    rows, cols, channels = paste_img.shape
    roi = back[k:rows + k, n:cols + n]

    paste_img_copy = paste_img.copy()

    if mask_alg == "otsu":
        mask = otsu_threshold(paste_img, [255, 255, 255])
        color_mask, bound_box = bound_box_otsu(paste_img, mcolor)
    elif mask_alg == "kmeans":
        mask = kmeans_img(paste_img, [255, 255, 255])
        color_mask, bound_box = bound_box_kmeans(paste_img, mcolor)
    elif mask_alg == "watershed":
        mask = watershed_img(paste_img, [255, 255, 255])
        color_mask, bound_box = bound_box_watershed(paste_img, mcolor)

    mask = grey_img(mask)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(paste_img_copy, paste_img_copy, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    back[k:rows + k, n:cols + n] = dst

    x1 = bound_box[0] + n
    y1 = bound_box[1] + k

    x2 = x1 + bound_box[2]
    y2 = y1 + bound_box[3]

    return back, x1, y1, x2, y2


def box_cut_paste_th(img_count, classes, colormap, i, prev, mask_alg, class_list):
    bounding_boxes = []
    for num in range(len(classes)):
        # "C:/Users/iraor/PycharmProjects/FusionBrainGen/images/{translate_text(classes[num])}/original/"
        img_num = random.choice(np.arange(img_count))
        img = cv2.imread(
            f"C:/Users/iraor/PycharmProjects/FusionBrainGen/images/{classes[num]}/original/image{img_num}.png")

        one_class, x1, y1, x2, y2 = cut_paste_box(img, prev, colormap[num], mask_alg)
        prev = one_class
        bounding_boxes.append([x1, y1, x2, y2])

    prev_mask = prev.copy()
    for num in range(len(classes)):
        x1, y1, x2, y2 = bounding_boxes[num]
        prev_mask = cv2.rectangle(prev_mask, (x1, y1), (x2, y2), colormap[num], 2)
        with open(f"join_classes{class_list}/bbox/coordinates/image{i}_coordinates.txt", "a+") as file:
            file.write(f"class:{classes[num]} x:{x1}, y:{y1}, w:{x2-x1}, h:{y2-y1}\n")

    cv2.imwrite(f"join_classes{class_list}/original/image{i}.png", prev)
    cv2.imwrite(f"join_classes{class_list}/bbox/image{i}.png", prev_mask)


def cut_paste(paste_img, back, back_mask, mcolor, mask_alg, color_mask):
    # размер уменьшения объекта
    sizes = [128, 256]  # 521]
    size = random.choice(sizes)

    # смешение объекта верт.+гориз.
    if size == 512:
        k = 0
        n = 0
    else:
        k = random.choice(np.arange(size))
        n = random.choice(np.arange(size))

    # меняем размер объекта
    paste_img = cv2.resize(paste_img, [size, size], cv2.INTER_AREA)

    rows, cols, channels = paste_img.shape
    roi = back[k:rows + k, n:cols + n]

    # if mask_alg == "otsu":
    #     mask = otsu_threshold(paste_img, [255, 255, 255])
    #     color_mask = otsu_threshold(paste_img, mcolor)
    # elif mask_alg == "kmeans":
    #     mask = kmeans_img(paste_img, [255, 255, 255])
    #     color_mask = kmeans_img(paste_img, mcolor)
    # elif mask_alg == "watershed":
    #     mask = watershed_img(paste_img, [255, 255, 255])
    #     color_mask = watershed_img(paste_img, mcolor)
    nmask = cv2.resize(color_mask, [size, size], cv2.INTER_AREA)
    mask = nmask.copy()

    for i in range(nmask.shape[0]):
        for j in range(nmask.shape[1]):
            if list(nmask[i, j]) == list(mcolor):
                mask[i, j] = [255, 255, 255]

    mask = grey_img(mask)

    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(paste_img, paste_img, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    back[k:rows + k, n:cols + n] = dst

    roi = back_mask[k:rows + k, n:cols + n]
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(nmask, nmask, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    back_mask[k:rows + k, n:cols + n] = dst
    return back, back_mask


def cut_paste_th(img_count, classes, colormap, i, prev, prev_mask, mask_alg, class_list):
    for num in range(len(classes)):
        # "C:/Users/iraor/PycharmProjects/FusionBrainGen/images/{translate_text(classes[num])}/original/"
        some_num = random.choice(np.arange(img_count))
        img = cv2.imread(
            f"C:/Users/iraor/PycharmProjects/FusionBrainGen/images/{classes[num]}/original/image{some_num}.png")
        color_mask = cv2.imread(
            f"C:/Users/iraor/PycharmProjects/FusionBrainGen/images/{classes[num]}/_{mask_alg}/image{some_num}.png")
        one_class, one_class_mask = cut_paste(img, prev, prev_mask, colormap[num], mask_alg, color_mask)
        prev = one_class
        prev_mask = one_class_mask

    cv2.imwrite(f"join_classes{class_list}/original/image{i}.png", prev)
    cv2.imwrite(f"join_classes{class_list}/mask/image{i}.png", prev_mask)
# ------------------------------------------------------------------------------------------


# ----------------------------------ИНТЕРФЕЙС---------------------------------------
class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.obj = ''
        self.img_count = 0
        self.mcolor = (255, 255, 255)

        self.buttonGen = QtWidgets.QPushButton("Начать генерацию изображений")
        self.buttonGen.setStyleSheet('background: rgb(176, 200, 214);')

        self.lblSize = QLabel("Выберите размер изображения (от 256x256 до 1024x1024):")
        self.imsize = QSpinBox()
        self.imsize.setMinimum(256)
        self.imsize.setMaximum(1024)

        self.filename = QLineEdit()

        self.buttonJoin = QtWidgets.QPushButton("Объединить классы")
        self.buttonJoin.setStyleSheet('background: rgb(176, 200, 214);')
        self.labelBack = QLabel("Выберите тип фона: ")
        self.comboBack = QComboBox(self)
        self.comboBack.setFixedWidth(500)
        self.comboBack.addItems(["белый", "чёрный", "сгенерированный набор", "пользовательский набор"])

        self.labelClasses = QLabel("Выберите классы: ")
        self.list_widget = QListWidget(self)

        images_dir = "images/"
        content = os.listdir(images_dir)
        for file in content:
            if os.path.isdir(os.path.join(images_dir, file)):
                self.list_widget.addItem(file)

        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        self.buttonSeg = QtWidgets.QPushButton("Запустить алгоритм сегментации")
        self.buttonSeg.setStyleSheet('background: rgb(176, 200, 214);')

        self.buttonStyle = QtWidgets.QPushButton("Стилизовать изображения")
        self.buttonStyle.setStyleSheet('background: rgb(176, 200, 214);')

        self.buttonBrowse = QtWidgets.QPushButton("Загрузить")
        self.buttonBrowse.setStyleSheet('background: rgb(176, 200, 214);')

        self.labelFile = QLabel("Загрузите изображение для стилизации: ")

        self.labelObject = QLabel("Введите объект генерации: ")
        self.textObject = QLineEdit("")

        self.labelCount = QLabel("Введите количество изображений: ")
        self.textCount = QLineEdit("")

        self.labelStyle = QLabel("Выберите стиль: ")
        self.comboStyle = QComboBox(self)
        self.comboStyle.addItems(["style1", "style2", "style3", "style4", "style5",
                                  "style6", "style7", "style8", "style9", "style10", "user style"])
        self.comboStyle.currentTextChanged.connect(self.on_changed)

        self.pic = QLabel()
        self.pic.setFixedWidth(200)
        self.pic.setFixedHeight(200)
        self.pic.setScaledContents(True)
        self.pic.setPixmap(QPixmap("sources/style1.png"))

        self.labelAlg = QLabel("Выберите алгоритм сегментации: ")
        self.combo = QComboBox(self)
        self.combo.addItems(["Пороговая сегментация Отсу (Otsu thresholding)",
                             "Алгоритм водораздела (WaterShed)",
                             "Алгоритм k-средние (k-means)"])

        self.labelSelect = QLabel("Выберите способ выделения: ")
        self.rb1 = QRadioButton("Pixel Mask")
        self.rb1.setChecked(True)
        self.rb2 = QRadioButton("Bounding box")

        self.buttonColor = QtWidgets.QPushButton("Выбрать цвет")
        self.buttonColor.setStyleSheet('background: rgb(176, 200, 214);')

        self.labelBrowseBack = QLabel("Выберите папку с пользовательским набором фонов: ")
        self.buttonBrowseBack = QtWidgets.QPushButton("Выбрать папку")
        self.buttonBrowseBack.setStyleSheet('background: rgb(176, 200, 214);')
        self.dirname = QLineEdit()

        self.btnDataset = QtWidgets.QPushButton("Сохранить выборку")
        self.btnDataset.setStyleSheet('background: rgb(176, 200, 214);')
        self.lblFormat = QLabel("Выберите формат аннотаций: ")
        self.cbFormat = QComboBox(self)
        self.cbFormat.setFixedWidth(300)
        self.cbFormat.addItems(["CSV", "JSON", "YOLO"])
        self.lblSave = QLabel("Выберите папку для сохранения выборки: ")
        self.btnSave = QtWidgets.QPushButton("Выбрать папку сохранения")
        self.btnSave.setStyleSheet('background: rgb(176, 200, 214);')
        self.dirSave = QLineEdit()

        self.layout1 = QtWidgets.QVBoxLayout(self)
        self.layout2 = QtWidgets.QVBoxLayout(self)
        self.layout3 = QtWidgets.QVBoxLayout(self)
        self.layout4 = QtWidgets.QHBoxLayout(self)
        self.layout2.addWidget(self.labelStyle)
        self.layout2.addWidget(self.comboStyle)
        self.layout2.addWidget(self.pic)
        self.layout3.addWidget(self.labelFile)
        self.layout3.addWidget(self.filename)
        self.layout3.addWidget(self.buttonBrowse)
        self.layout3.addWidget(self.buttonStyle)
        self.layout3.addStretch()
        self.layout4.addLayout(self.layout2)
        self.layout4.addLayout(self.layout3)

        self.layout1.addWidget(self.labelObject)
        self.layout1.addWidget(self.textObject)

        self.layout1.addWidget(self.labelCount)
        self.layout1.addWidget(self.textCount)

        self.layout1.addWidget(self.lblSize)
        self.layout1.addWidget(self.imsize)

        self.layout1.addWidget(self.buttonGen)

        self.layout1.addLayout(self.layout4)

        self.layout1.addWidget(self.labelAlg)
        self.layout1.addWidget(self.combo)

        self.layout5 = QtWidgets.QVBoxLayout(self)
        self.layout6 = QtWidgets.QVBoxLayout(self)
        self.layout7 = QtWidgets.QHBoxLayout(self)

        self.layout5.addWidget(self.labelSelect)
        self.layout5.addWidget(self.rb1)
        self.layout5.addWidget(self.rb2)
        self.layout5.addStretch()

        self.labelColor = QLabel("Выберите цвет маски/рамки:")
        self.layout6.addWidget(self.labelColor)
        self.layout6.addWidget(self.buttonColor)
        self.textColor = QLineEdit("")
        self.layout6.addWidget(self.textColor)
        self.layout6.addStretch()

        self.layout7.addLayout(self.layout5)
        self.layout7.addLayout(self.layout6)
        self.layout1.addLayout(self.layout7)

        self.layout1.addWidget(self.buttonSeg)

        self.layout8 = QtWidgets.QVBoxLayout(self)
        self.layout9 = QtWidgets.QVBoxLayout(self)
        self.layout10 = QtWidgets.QHBoxLayout(self)

        self.layout8.addWidget(self.labelBack)
        self.layout8.addWidget(self.comboBack)
        self.layout8.addStretch()

        self.layout9.addWidget(self.labelBrowseBack)
        self.layout9.addWidget(self.dirname)
        self.layout9.addWidget(self.buttonBrowseBack)
        self.layout9.addStretch()

        self.layout10.addLayout(self.layout8)
        self.layout10.addLayout(self.layout9)
        self.layout1.addLayout(self.layout10)

        self.layout1.addWidget(self.labelClasses)
        self.layout1.addWidget(self.list_widget)

        self.layout1.addWidget(self.buttonJoin)

        self.layout11 = QtWidgets.QVBoxLayout(self)
        self.layout12 = QtWidgets.QVBoxLayout(self)
        self.layout13 = QtWidgets.QHBoxLayout(self)

        self.layout11.addWidget(self.lblFormat)
        self.layout11.addWidget(self.cbFormat)
        self.layout11.addStretch()

        self.layout12.addWidget(self.lblSave)
        self.layout12.addWidget(self.dirSave)
        self.layout12.addWidget(self.btnSave)
        self.layout12.addStretch()

        self.layout13.addLayout(self.layout11)
        self.layout13.addLayout(self.layout12)
        self.layout1.addLayout(self.layout13)

        self.layout1.addWidget(self.btnDataset)

        self.buttonGen.clicked.connect(self.start_image_generating)
        self.buttonSeg.clicked.connect(self.start_image_segmentation)
        self.buttonStyle.clicked.connect(self.start_image_stylization)

        self.buttonBrowse.clicked.connect(self.open_file_dialog)
        self.buttonColor.clicked.connect(self.color_dialog)

        self.buttonJoin.clicked.connect(self.join_classes)

        self.buttonBrowseBack.clicked.connect(self.get_directory)

        self.btnSave.clicked.connect(self.get_save_directory)
        self.btnDataset.clicked.connect(self.save_dataset)

    @QtCore.Slot()
    def on_changed(self, text):
        self.pic.setPixmap(QPixmap(f"sources/{text}.png"))

        if text != 'user style':
            self.filename.setText('')

    def open_file_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            "D:\\icons\\avatar\\",
            "Images (*.png)"
        )
        if filename:
            path = Path(filename)
            self.filename.setText(str(path))
            # self.comboStyle.addItem('user style')
            self.comboStyle.setCurrentText('user style')

            text_path = self.filename.text().replace('\\', '/')
            self.pic.setPixmap(QPixmap(f"{text_path}"))

    @QtCore.Slot()
    def start_image_generating(self):
        # --------------------------- исходные параметры ---------------------------------
        self.obj = self.textObject.text()
        self.img_count = int(self.textCount.text())
        imsize = self.imsize.value()

        # --------------------------------------------------------------------------------
        # --------------------------- генерация промтов ----------------------------------
        promts = generate_promts(self.obj, self.img_count)
        # print(promts)
        # --------------------------------------------------------------------------------
        # ------------------------ генерация изображений --------------------------------
        _generate_(self.obj, self.img_count, promts, imsize)
        # -------------------------------------------------------------------------------

    @QtCore.Slot()
    def start_image_stylization(self):
        self.obj = self.textObject.text()
        self.img_count = int(self.textCount.text())
        imsize = self.imsize.value()

        if self.comboStyle.currentText() == 'user style':
            style_path = self.filename.text().replace('\\', '/')
        else:
            style_path = f'sources/{self.comboStyle.currentText()}.png'

        style_image = load_img(style_path, imsize)
        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        if not (os.path.exists(f'images/{translate_text(self.obj)}/stylized')):
            os.makedirs(f'images/{translate_text(self.obj)}/stylized')

        for i in range(self.img_count):
            thread = threading.Thread(target=stylized_img,
                                      args=(i, style_image, hub_module, translate_text(self.obj), imsize))
            thread.start()
            # stylized_img(i, style_image, hub_module, translate_text(self.obj))

    @QtCore.Slot()
    def start_image_segmentation(self):
        # --------------------------- исходные параметры ---------------------------------
        self.obj = self.textObject.text()
        self.img_count = int(self.textCount.text())

        mcolor = self.mcolor
        # --------------------------------------------------------------------------------

        tr_obj = translate_text(self.obj)
        path = f'images\\{tr_obj}\\'

        file_colormap = open("colormap.txt", "a+")
        file_colormap.write(f"{tr_obj}: {mcolor}\n")
        file_colormap.close()

        # ----------------------------------------- МАСКА ------------------------------------------------------------
        if (self.combo.currentText() == "Пороговая сегментация Отсу (Otsu thresholding)") and (self.rb1.isChecked()):
            if not (os.path.exists(f'{path}_otsu')):
                os.makedirs(f'{path}_otsu')

            for i in range(self.img_count):
                img_name = f'image{i}'
                image = cv2.imread(f'{path}\\original\\{img_name}.png')
                # --------------------------------------------------------------------
                thread = threading.Thread(target=th_otsu, args=(path, img_name, image, mcolor))
                thread.start()
                # --------------------------------------------------------------------

        if (self.combo.currentText() == "Алгоритм k-средние (k-means)") and (self.rb1.isChecked()):
            if not (os.path.exists(f'{path}_kmeans')):
                os.makedirs(f'{path}_kmeans')
            # mcolor = (200, 60, 170)

            for i in range(self.img_count):
                img_name = f'image{i}'
                image = cv2.imread(f'{path}\\original\\{img_name}.png')
                # --------------------------------------------------------------------
                thread = threading.Thread(target=th_kmn, args=(path, img_name, image, mcolor))
                thread.start()
                # --------------------------------------------------------------------

        if (self.combo.currentText() == "Алгоритм водораздела (WaterShed)") and (self.rb1.isChecked()):
            if not (os.path.exists(f'{path}_wsh_mask')):
                os.makedirs(f'{path}_wsh_mask')
            # if not (os.path.exists(f'{path}_maskCont')):
            #     os.makedirs(f'{path}_maskCont')
            # mcolor = (200, 60, 170)

            for i in range(self.img_count):
                img_name = f'image{i}'
                image = cv2.imread(f'{path}\\original\\{img_name}.png')
                # --------------------------------------------------------------------
                thread = threading.Thread(target=th_wsh, args=(path, img_name, image, mcolor))
                thread.start()
                # --------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------

        # ---------------------------------------BOUNDING BOXES-------------------------------------------------
        if (self.combo.currentText() == "Пороговая сегментация Отсу (Otsu thresholding)") and (self.rb2.isChecked()):
            if not (os.path.exists(f'{path}_box_otsu')):
                os.makedirs(f'{path}_box_otsu')

            for i in range(self.img_count):
                img_name = f'image{i}'
                image = cv2.imread(f'{path}\\original\\{img_name}.png')
                # --------------------------------------------------------------------
                if not (os.path.exists(f'{path}/_box_otsu/coordinates')):
                    os.makedirs(f'{path}/_box_otsu/coordinates')
                thread = threading.Thread(target=th_box_otsu, args=(path, img_name, image, mcolor))
                thread.start()
                # --------------------------------------------------------------------

        if (self.combo.currentText() == "Алгоритм k-средние (k-means)") and (self.rb2.isChecked()):
            if not (os.path.exists(f'{path}_box_kmeans')):
                os.makedirs(f'{path}_box_kmeans')
            # mcolor = (255, 0, 0)

            for i in range(self.img_count):
                img_name = f'image{i}'
                image = cv2.imread(f'{path}\\original\\{img_name}.png')
                # --------------------------------------------------------------------
                if not (os.path.exists(f'{path}/_box_kmeans/coordinates')):
                    os.makedirs(f'{path}/_box_kmeans/coordinates')
                thread = threading.Thread(target=th_box_kmeans, args=(path, img_name, image, mcolor))
                thread.start()
                # --------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------
        if (self.combo.currentText() == "Алгоритм водораздела (WaterShed)") and (self.rb2.isChecked()):
            if not (os.path.exists(f'{path}_box_watershed')):
                os.makedirs(f'{path}_box_watershed')

            # mcolor = (200, 60, 170)

            for i in range(self.img_count):
                img_name = f'image{i}'
                image = cv2.imread(f'{path}\\original\\{img_name}.png')
                # --------------------------------------------------------------------
                if not (os.path.exists(f'{path}/_box_watershed/coordinates')):
                    os.makedirs(f'{path}/_box_watershed/coordinates')
                thread = threading.Thread(target=th_box_watershed, args=(path, img_name, image, mcolor))
                thread.start()
                # --------------------------------------------------------------------

    @QtCore.Slot()
    def color_dialog(self):
        color = QColorDialog.getColor()
        rgb = str(color.getRgb()).replace('(', '').replace(')', '').replace(' ', '').split(',')
        rgb_list = [int(rgb[2]), int(rgb[1]), int(rgb[0])]
        self.mcolor = tuple(rgb_list)

        if color.isValid():
            self.textColor.setStyleSheet("QWidget { background-color: %s }" % color.name())

    def get_directory(self):
        dirlist = QFileDialog.getExistingDirectory(self, "Выбрать папку", ".")
        self.comboBack.setCurrentText("пользовательский набор")
        self.dirname.setText(dirlist)

    def get_save_directory(self):
        dir_for_save = QFileDialog.getExistingDirectory(self, "Выбрать папку", ".")
        self.dirSave.setText(dir_for_save)

    @QtCore.Slot()
    def join_classes(self):
        self.img_count = int(self.textCount.text())
        classes = [item.text() for item in self.list_widget.selectedItems()]
        # classes = ['flower', 'apple']
        class_list = ""
        for i in range(len(classes)):
            class_list += f"_{classes[i]}"

        join_classes_path = [f"join_classes{class_list}/original", f"join_classes{class_list}/mask"]
        for path in join_classes_path:
            if not (os.path.exists(path)):
                os.makedirs(path)

        shutil.move("colormap.txt", f"join_classes{class_list}/colormap.txt")

        file = open(f"join_classes{class_list}/colormap.txt", "r")
        lines = file.readlines()
        colormap = []
        for line in lines:
            numbs = list(line.split(":")[1]
                         .replace("\n", "").replace(" ", "")
                         .replace(")", "").replace("(", "").split(","))

            numbs = [int(numb) for numb in numbs]
            colormap.append(numbs)

        if self.combo.currentText() == "Пороговая сегментация Отсу (Otsu thresholding)":
            mask_alg = "otsu"
        elif self.combo.currentText() == "Алгоритм k-средние (k-means)":
            mask_alg = "kmeans"
        else:
            mask_alg = "watershed"

        for i in range(self.img_count):
            back_mask = cv2.imread("sources/back_mask.png")
            if self.comboBack.currentText() == "белый":
                back = cv2.imread("sources/back.png")
            elif self.comboBack.currentText() == "чёрный":
                back = cv2.imread("sources/back_mask.png")
            elif self.comboBack.currentText() == "сгенерированный набор":
                num = random.choice(np.arange(3700))
                back = cv2.imread(f"backgrounds/image{num}.png")
            else:
                files_count = len(os.listdir(self.dirname.text()))
                num = random.choice(np.arange(files_count))
                back = cv2.imread(f"{self.dirname.text()}/image{num}.png")

            prev = back
            prev_mask = back_mask

            if self.rb1.isChecked():
                thread = threading.Thread(target=cut_paste_th,
                                          args=(self.img_count, classes, colormap, i, prev, prev_mask,
                                                mask_alg, class_list))
            elif self.rb2.isChecked():
                if not (os.path.exists(f"join_classes{class_list}/bbox")):
                    os.rename(f"join_classes{class_list}/mask", f"join_classes{class_list}/bbox")
                if not (os.path.exists(f"join_classes{class_list}/bbox/coordinates")):
                    os.makedirs(f"join_classes{class_list}/bbox/coordinates")
                thread = threading.Thread(target=box_cut_paste_th,
                                          args=(self.img_count, classes, colormap, i, prev, mask_alg, class_list))
            thread.start()

    @QtCore.Slot()
    def save_dataset(self):
        classes = [item.text() for item in self.list_widget.selectedItems()]
        if len(classes) == 1:  # условие на кол-во классов
            images_from_path = f"images/{classes[0]}/original"
            if ((self.combo.currentText() == "Пороговая сегментация Отсу (Otsu thresholding)")
                    and (self.rb1.isChecked())):
                labeles_from_path = f"images/{classes[0]}/_otsu"
            elif (self.combo.currentText() == "Алгоритм k-средние (k-means)") and (self.rb1.isChecked()):
                labeles_from_path = f"images/{classes[0]}/_kmeans"
            elif (self.combo.currentText() == "Алгоритм водораздела (WaterShed)") and (self.rb1.isChecked()):
                labeles_from_path = f"images/{classes[0]}/_wsh_mask"
            elif ((self.combo.currentText() == "Пороговая сегментация Отсу (Otsu thresholding)")
                    and (self.rb2.isChecked())):
                labeles_from_path = f"images/{classes[0]}/_box_otsu"
            elif (self.combo.currentText() == "Алгоритм k-средние (k-means)") and (self.rb2.isChecked()):
                labeles_from_path = f"images/{classes[0]}/_box_kmeans"
            else:
                labeles_from_path = f"images/{classes[0]}/_box_watershed"

            if not (os.path.exists(f"{self.dirSave.text()}/dataset/images")):
                os.makedirs(f"{self.dirSave.text()}/dataset/images")
            if not (os.path.exists(f"{self.dirSave.text()}/dataset/labels")):
                os.makedirs(f"{self.dirSave.text()}/dataset/labels")

            if self.cbFormat.currentText() == "CSV":
                for i in range(len(os.listdir(images_from_path))):
                    shutil.copy2(f'{images_from_path}/image{i}.png',
                                 f'{self.dirSave.text()}/dataset/images/image{i}.png')
                    shutil.copy2(f'{labeles_from_path}/image{i}.png',
                                 f'{self.dirSave.text()}/dataset/labels/image{i}.png')

                # ------------------------ csv изображение-метка -----------------------------------------
                columns = ['image', 'label']
                data = []
                for i in range(len(os.listdir(images_from_path))):
                    data.append([f"dataset/images/image{i}.png",
                                 f"dataset/labels/image{i}.png"])

                df = pd.DataFrame(data, columns=columns)
                df.to_csv(f'{self.dirSave.text()}/dataset/image-label.csv', index=False)
                # ----------------------------------------------------------------------------------------
                # ------------------------------ csv изображение-координаты bBox -----------------------------------
                if labeles_from_path.find("box") != -1:
                    data = []
                    for i in range(len(os.listdir(f"{self.dirSave.text()}/dataset/images"))):
                        name = f"{labeles_from_path}/coordinates/image{i}_coordinates.txt"
                        file = open(name, "r")
                        data.append([f"dataset/images/image{i}.png", f"{file.readline()}"])

                    df = pd.DataFrame(data, columns=columns)
                    df.to_csv(f'{self.dirSave.text()}/dataset/bounding_boxes.csv', index=False)
                # ---------------------------------------------------------------------------------------------------

            elif self.cbFormat.currentText() == "JSON":
                for i in range(len(os.listdir(images_from_path))):
                    shutil.copy2(f'{images_from_path}/image{i}.png',
                                 f'{self.dirSave.text()}/dataset/images/image{i}.png')
                    shutil.copy2(f'{labeles_from_path}/image{i}.png',
                                 f'{self.dirSave.text()}/dataset/labels/image{i}.png')

                # ------------------------ json изображение-метка-bbox -----------------------------------------
                images_array = []
                labels_array = []

                for i in range(len(os.listdir(images_from_path))):
                    images_array.append({"id": i, "file_name": f"dataset/images/image{i}.png"})
                    if labeles_from_path.find("box") != -1:
                        name = f"{labeles_from_path}/coordinates/image{i}_coordinates.txt"
                        file = open(name, "r")

                        row = file.readline().split(',')
                        x = row[0].split(':')[1].replace(',', "")
                        y = row[1].split(':')[1].replace(',', "")
                        w = row[2].split(':')[1].replace(',', "")
                        h = row[3].split(':')[1]
                        labels_array.append({"id": i, "file_name": f"dataset/labels/image{i}.png",
                                             "bbox": {"x": float(x),
                                                      "y": float(y),
                                                      "width": float(w),
                                                      "high": float(h)}
                                             })
                    else:
                        labels_array.append({"id": i, "file_name": f"dataset/labels/image{i}.png"})

                dict_for_json = {"images": images_array,
                                 "annotations": labels_array}

                with open(f"{self.dirSave.text()}/dataset/bounding_boxes.json", "w") as f:
                    json.dump(dict_for_json, f)
                # ------------------------------------------------------------------------------------------------
            else:
                for i in range(len(os.listdir(images_from_path))):
                    shutil.copy2(f'{images_from_path}/image{i}.png',
                                 f'{self.dirSave.text()}/dataset/images/image{i}.png')
                    shutil.copy2(f'{labeles_from_path}/coordinates/image{i}_coordinates.txt',
                                 f'{self.dirSave.text()}/dataset/labels/label{i}.txt')
        else:  # несколько классов
            class_list = ""
            for i in range(len(classes)):
                class_list += f"_{classes[i]}"

            images_from_path = f"join_classes{class_list}/original"
            if not (os.path.exists(f"join_classes{class_list}/bbox")):
                labeles_from_path = f"join_classes{class_list}/mask"
            else:
                labeles_from_path = f"join_classes{class_list}/bbox"

            if not (os.path.exists(f"{self.dirSave.text()}/dataset/images")):
                os.makedirs(f"{self.dirSave.text()}/dataset/images")
            if not (os.path.exists(f"{self.dirSave.text()}/dataset/labels")):
                os.makedirs(f"{self.dirSave.text()}/dataset/labels")

            if self.cbFormat.currentText() == "CSV":
                for i in range(len(os.listdir(images_from_path))):
                    shutil.copy2(f'{images_from_path}/image{i}.png',
                                 f'{self.dirSave.text()}/dataset/images/image{i}.png')
                    shutil.copy2(f'{labeles_from_path}/image{i}.png',
                                 f'{self.dirSave.text()}/dataset/labels/image{i}.png')

                # ------------------------ csv изображение-метка -----------------------------------------
                columns = ['image', 'label']
                data = []
                for i in range(len(os.listdir(images_from_path))):
                    data.append([f"dataset/images/image{i}.png",
                                 f"dataset/labels/image{i}.png"])

                df = pd.DataFrame(data, columns=columns)
                df.to_csv(f'{self.dirSave.text()}/dataset/image-label.csv', index=False)
                # ----------------------------------------------------------------------------------------
                # ------------------- csv изображение-координаты bBox/mask_color -------------------------
                if labeles_from_path.find("box") != -1:
                    data = []
                    for i in range(len(os.listdir(f"{self.dirSave.text()}/dataset/images"))):
                        name = f"{labeles_from_path}/coordinates/image{i}_coordinates.txt"
                        file = open(name, "r")
                        lines = file.readlines()
                        data.append([f"dataset/images/image{i}.png", f"{lines[0]}{lines[1]}"])

                    df = pd.DataFrame(data, columns=columns)
                    df.to_csv(f'{self.dirSave.text()}/dataset/bounding_boxes.csv', index=False)
                else:
                    columns = ['class', 'r', 'g', 'b']
                    data = []
                    colormap_file = open(f'join_classes{class_list}/colormap.txt', "r")
                    for cl in classes:
                        bgr = (colormap_file.readline().split(":")[1].replace('(', "")
                               .replace(')', "").replace(",", "").split())
                        data.append([cl, bgr[2], bgr[1], bgr[0]])

                    df = pd.DataFrame(data, columns=columns)
                    df.to_csv(f'{self.dirSave.text()}/dataset/mask_color.csv', index=False)
                # -----------------------------------------------------------------------------------------

            elif self.cbFormat.currentText() == "JSON":
                for i in range(len(os.listdir(images_from_path))):
                    shutil.copy2(f'{images_from_path}/image{i}.png',
                                 f'{self.dirSave.text()}/dataset/images/image{i}.png')
                    shutil.copy2(f'{labeles_from_path}/image{i}.png',
                                 f'{self.dirSave.text()}/dataset/labels/image{i}.png')

                # ------------------------ json изображение-метка-bbox -----------------------------------------
                images_array = []
                labels_array = []

                for i in range(len(os.listdir(images_from_path))):
                    images_array.append({"id": i, "file_name": f"dataset/images/image{i}.png"})
                    if labeles_from_path.find("box") != -1:
                        name = f"{labeles_from_path}/coordinates/image{i}_coordinates.txt"
                        file = open(name, "r")
                        bbox = []

                        for cl in classes:
                            row = file.readline().split()
                            x = row[1].split(':')[1].replace(',', "")
                            y = row[2].split(':')[1].replace(',', "")
                            w = row[3].split(':')[1].replace(',', "")
                            h = row[4].split(':')[1].replace('\n', "")
                            bbox.append({"x": float(x),
                                         "y": float(y),
                                         "width": float(w),
                                         "high": float(h),
                                         "class": cl})

                        labels_array.append({"id": i, "file_name": f"dataset/labels/image{i}.png",
                                             "bbox": bbox
                                             })
                    else:
                        labels_array.append({"id": i, "file_name": f"dataset/labels/image{i}.png"})

                dict_for_json = {"images": images_array,
                                 "annotations": labels_array}

                with open(f"{self.dirSave.text()}/dataset/bounding_boxes.json", "w") as f:
                    json.dump(dict_for_json, f)
                # ------------------------------------------------------------------------------------------------
            else:
                if labeles_from_path.find("box") != -1:
                    for i in range(len(os.listdir(images_from_path))):
                        shutil.copy2(f'{images_from_path}/image{i}.png',
                                     f'{self.dirSave.text()}/dataset/images/image{i}.png')
                        shutil.copy2(f'{labeles_from_path}/coordinates/image{i}_coordinates.txt',
                                     f'{self.dirSave.text()}/dataset/labels/label{i}.txt')


# ---------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.setGeometry(600, 50, 500, 250)
    widget.show()

    sys.exit(app.exec())
