import os
from ultralytics import YOLO
import easyocr
import shutil
import cv2
import re

class Pasport:
    def __init__(self):
        self.__reader = easyocr.Reader(['ru'])
        module_dir = os.path.dirname(__file__)
        model_file_path = os.path.join(module_dir, 'pasport_model.pt')
        self.__model = YOLO(model_file_path)

    def recognize_pasport(self, img, main_folder, folder_id: str) -> dict:
        """
        :param img: path to passport img
        :param main_folder: An intermediate folder for storing recognized photos, after returning the data is deleted
        :param folder_id: id of the folder inside main_folder where the photo recognition is stored (random value)
        :return: info from passport
        """
        self.__model(img, retina_masks=True, save_crop=True, project=main_folder, name=folder_id)

        info = {}
        for dir in os.listdir(f"{main_folder}/{folder_id}/crops"):
            if dir != "pass-ort":
                info[dir] = f"{main_folder}/{folder_id}/crops/{dir}/" + os.listdir(f"{main_folder}/{folder_id}/crops/{dir}/")[0]

        img = cv2.imread(info["num"])
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(info["num"], img)
        img = cv2.imread(info["ser"])
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(info["ser"], img)


        for i in info.keys():
            info[i] = "".join(self.__reader.readtext(info[i], detail=0)).lower().capitalize()

        info['date'] = self.made_date(info['data'])
        info['out_date'] = self.made_date(info['out_data'])
        info.pop('data')
        info.pop('out_data')


        shutil.rmtree(f"{main_folder}/{folder_id}")
        return info

    @staticmethod
    def made_date(line: str) -> str:
        line = re.sub(r"[\D]", "", line)
        return f"{line[:2]}.{line[2:4]}.{line[4:]}"
