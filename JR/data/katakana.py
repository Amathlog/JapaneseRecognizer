from collections import defaultdict
import copy
import pickle
import zlib

import scipy.ndimage

from JR.data.config import DATA_DIR
from JR.data.ETL1 import ETL1

FILENAME = DATA_DIR / "Katakana.data"

SEED = 152664
TRAINING_RATIO = 0.8

class Katakana:
    def __init__(self):
        self.all_data = defaultdict(list)
        self.training_data = []
        self.training_classes = []
        self.evaluation_data = []
        self.evaluation_classes = []
        self.load_data()

    def load_data(self):
        print("Loading data...")
        if not FILENAME.exists():
            print("Preprocessed data doesn't exists. Create it...")
            self.preprocess_data()
            print("Preprocess done!")

        with FILENAME.open('rb') as f:
            data = pickle.loads(zlib.decompress(f.read()))

        for d in data:
            self.all_data[d.ascii].append(d)
        self.generate_dataset()
        #self.data_augmentation()

    def generate_dataset(self):
        classes = {key: i for i, (key, _) in enumerate(self.all_data.items())}
        flatten = lambda x: [np.reshape(d.img, (-1,)) for d in x]
        np.random.seed(SEED)
        for key, values in self.all_data.items():
            limit = int(len(values) * TRAINING_RATIO)
            indexes = np.arange(len(values))
            np.random.shuffle(indexes)
            values = np.array(values)

            self.training_data.extend(flatten(values[indexes[:limit]]))
            self.training_classes.extend([classes[key]] * limit)

            self.evaluation_data.extend(flatten(values[indexes[limit:]]))
            self.evaluation_classes.extend([classes[key]] * (len(values) - limit))

    def preprocess_data(self):
        files_to_unpack = [
            "ETL1/ETL1C_07",
            "ETL1/ETL1C_08",
            "ETL1/ETL1C_09",
            "ETL1/ETL1C_10",
            "ETL1/ETL1C_11",
            "ETL1/ETL1C_12",
            "ETL1/ETL1C_13",
        ]

        etl = ETL1(files_to_unpack, False)

        print(f"Data loaded from zip. {len(etl.data)} entries")
        print("Now compress it")
        data = pickle.dumps(etl.data)
        with FILENAME.open('wb') as f:
            f.write(zlib.compress(data))

    def data_augmentation(self):
        # We will 3 random rotation of the image
        # between -20 and 20 deg
        print("Generating data augmentation")
        initial_data_size = len(self.all_data)
        for i in range(initial_data_size):
            for _ in range(3):
                tmp = copy.deepcopy(self.all_data[i])
                angle = np.random.uniform(-20, 20)
                tmp.img = scipy.ndimage.rotate(tmp.img, angle)
                self.all_data.append(tmp)

    def __repr__(self):
        res = "KATANA DATASET:\n"
        for key, value in self.all_data.items():
            res += f"{key}: {len(value)} entries\n"
        return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    x_ = Katakana()
    print(x_)
    # Visualize some data
    for i in range(10):
        values = np.random.choice(x_.all_data, 36)
        shape = list(np.shape(values[0].img))
        big_shape = (shape[0] * 6, shape[1] * 6)
        total = np.zeros(big_shape)
        for i in range(6):
            for j in range(6):
                total[i*shape[0]:(i+1)*shape[0],j*shape[1]:(j+1)*shape[1]] = values[i*6+j].img
        plt.imshow(total)
        plt.show()
        plt.close()