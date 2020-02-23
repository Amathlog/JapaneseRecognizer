import pickle
import zlib

from JR.data.config import DATA_DIR
from JR.data.ETL1 import ETL1

FILENAME = DATA_DIR / "Katakana.data"


class Katakana:
    def __init__(self):
        self.data = None
        self.load_data()

    def load_data(self):
        if not FILENAME.exists():
            self.preprocess_data()

        with FILENAME.open('rb') as f:
            self.data = pickle.loads(zlib.decompress(f.read()))

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

        etl = ETL1(files_to_unpack, True)

        data = pickle.dumps(etl.data)
        with FILENAME.open('wb') as f:
            f.write(zlib.compress(data))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    x_ = Katakana()
    plt.imshow(x_.data[0].img)
    plt.show()