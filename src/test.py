# import pandas as pd

# csv_path = r"D:\diabetic-retinopathy-project\Dataset\trainLabels.csv"
# df = pd.read_csv(csv_path)
# print(df.head())

import os

folder = r"D:\diabetic-retinopathy-project\Dataset\resized_train"
files = os.listdir(folder)
print("First 10 files:", files[:10])
