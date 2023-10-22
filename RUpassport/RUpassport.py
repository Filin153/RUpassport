import os
from ultralytics import YOLO
import easyocr
import shutil
import cv2

class Pasport:
    def __init__(self):
        self.__reader = easyocr.Reader(['ru'])
        self.__model = YOLO("RUpassport/pasport_model.pt")

    def recognize_pasport(self, img: str, id: str):
        """
        :param img: path to passport img
        :param id: folder id (random value)
        :return: info from passport
        """
        self.__model(img, retina_masks=True, save_crop=True, project="file", name=id)

        info = {}
        for dir in os.listdir(f"file/123/crops"):
            if dir != "pass-ort":
                info[dir] = f"file/123/crops/{dir}/" + os.listdir(f"file/123/crops/{dir}/")[0]

        img = cv2.imread(info["num"])
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(info["num"], img)
        img = cv2.imread(info["ser"])
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(info["ser"], img)

        for i in info.keys():
            info[i] = "".join(self.__reader.readtext(info[i], detail=0)).lower().capitalize()

        shutil.rmtree(f"file/{id}")
        return info