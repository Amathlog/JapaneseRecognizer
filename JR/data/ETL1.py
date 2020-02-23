from pathlib import Path
import os
import struct
import zipfile

import numpy as np
from skimage import filters
import wget
from PIL import Image

from JR.data_temp.urls import urls

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
DATA_DIR = CURRENT_DIR / ".." / "data_temp"
RECORD_SIZE = 2052
IMG_SIZE_PACKED = 2016
IMG_DIM = (64, 63)


def progress_bar(p, max_len=30):
    size = int(p * max_len)

    print("[" + ("#" * size) + ("-" * (max_len - size)) + f"] {int(p*100)}%", end='\r', flush=True)


def check_zip_file(filename, url):
    if not filename.exists():
        wget.download(url, str(filename))


def read_int(buffer, p, nb_bytes):
    res = int.from_bytes(buffer[p:p+nb_bytes], "big")
    return res, p + nb_bytes


class Record:
    def __init__(self, buffer):
        unpaked_buffer = struct.unpack('>H2sH6BI4H4B4x2016s4x', buffer)
        self.data_index = unpaked_buffer[0]
        self.ascii = unpaked_buffer[1]
        self.sheet_index = unpaked_buffer[2]
        self.jis_code = unpaked_buffer[3]
        self.ebcdic_code = unpaked_buffer[4]
        self.quality_image = unpaked_buffer[5]
        self.quality_group = unpaked_buffer[6]
        self.writer_gender = unpaked_buffer[7]
        self.writer_age = unpaked_buffer[8]
        self.serial_data_index = unpaked_buffer[9]
        # Useless 4 values
        self.y_pos = unpaked_buffer[14]
        self.x_pos = unpaked_buffer[15]
        self.min_intensity = unpaked_buffer[16]
        self.max_intensity = unpaked_buffer[17]
        self.img = np.array(list(Image.frombuffer('F', IMG_DIM, unpaked_buffer[18], 'bit', 4).getdata()))\
            .reshape((63, 64))
        threshold = filters.threshold_otsu(self.img, 16)
        self.img = np.cast[np.uint8](self.img > threshold)


class ETL1:
    def __init__(self):
        self.data = []
        self.load_data()

    def load_data(self):
        filename = DATA_DIR / "ETL1.zip"
        check_zip_file(filename, urls["ETL-1"])

        zip = zipfile.ZipFile(filename, 'r')
        for name in zip.namelist():
            splitName = name.split('/')
            if len(splitName[1]) == 0 or splitName[0] == "ETL1INFO" or splitName[1] != "ETL1C_07":
                continue
            temp_data = zip.read(name)
            assert len(temp_data) % RECORD_SIZE == 0
            print(f"Processing file {splitName[1]}...")
            nb_entries = len(temp_data) // RECORD_SIZE
            print(f"Found {nb_entries} entries")
            p = 0
            while p < len(temp_data):
                progress_bar(p / len(temp_data))
                self.data.append(Record(temp_data[p:p+RECORD_SIZE]))
                p += RECORD_SIZE
            break


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    x_ = ETL1()
    plt.imshow(x_.data[0].img)
    plt.show()
