import struct
import zipfile

import numpy as np
from skimage import filters
import scipy.signal
import wget
from PIL import Image

from JR.data_temp.urls import urls
from JR.data.config import DATA_DIR

ZIP_FILE = DATA_DIR / "ETL1.zip"
RECORD_SIZE = 2052
IMG_SIZE_PACKED = 2016
ORIGINAL_IMG_DIM = (64, 63)
IMG_DIM = (64, 64)


def progress_bar(p, max_len=30):
    size = int(p * max_len)

    print("[" + ("#" * size) + ("-" * (max_len - size)) + f"] {int(p*100)}%", end='\r', flush=True)


def check_zip_file(filename, url):
    if not filename.exists():
        print(f"{filename.name} doesn't exist... Downloading it")
        wget.download(url, str(filename))
        print()


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
        self.img = np.array(list(Image.frombuffer('F', ORIGINAL_IMG_DIM, unpaked_buffer[18], 'bit', 4).getdata()))\
            .reshape((ORIGINAL_IMG_DIM[1], ORIGINAL_IMG_DIM[0]))
        self.img = np.concatenate([np.zeros((1, 64)), self.img], axis=0)
        self.valid = False
        # For some reason some images have all the same pixel value.
        # Discard those ones
        if np.mean(self.img) == self.img[0][0]:
            return
        threshold = filters.threshold_otsu(self.img, 16)
        self.img = list(np.cast[np.uint8](self.img > threshold))
        # Compute the mean. If it is above 0.2, there is definitely a problem (lots of noise)
        mean = np.mean(self.img)
        if mean > 0.2:
            return
        self.valid = True


class ETL1:
    def __init__(self, files_to_unpack=None, clean_up=False):
        self.data = []
        self.load_data(files_to_unpack)
        if clean_up:
            ZIP_FILE.unlink()

    def load_data(self, files_to_unpack):
        check_zip_file(ZIP_FILE, urls["ETL-1"])

        zip = zipfile.ZipFile(ZIP_FILE, 'r')
        if files_to_unpack is None:
            files_to_unpack = zip.namelist()
        for name in files_to_unpack:
            splitName = name.split('/')
            if len(splitName[1]) == 0 or splitName[1] == "ETL1INFO":
                continue
            temp_data = zip.read(name)
            assert len(temp_data) % RECORD_SIZE == 0
            print(f"Processing file {splitName[1]}...")
            nb_entries = len(temp_data) // RECORD_SIZE
            print(f"Found {nb_entries} entries")
            p = 0
            while p < len(temp_data):
                progress_bar(p / len(temp_data))
                record = Record(temp_data[p:p+RECORD_SIZE])
                if record.valid:
                    self.data.append(record)
                p += RECORD_SIZE
            print("\nDone!")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    x_ = ETL1()
    plt.imshow(x_.data[0].img)
    plt.show()
