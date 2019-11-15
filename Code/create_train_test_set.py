from collections import defaultdict
import os
from shutil import copyfile


def copy_data(directory, files):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file in files:
        dst = directory + file.split("/")[-1]
        copyfile(file, dst)
    print(file, dst)


fake = defaultdict(list)
real = []
paths = defaultdict(dict)
    
source_dir = '/kaggle/input/real-and-fake-face-detection/real_and_fake_face_detection/real_and_fake_face'
for dirname, _, filenames in os.walk(source_dir):
    for filename in filenames:
        if "real" in filename:
            num = int(filename.split("_")[1].split(".")[0])
            real.append(num)
            paths["real"][num] = os.path.join(dirname, filename)
        else:
            difficulty = filename.split("_")[0]
            num = int(filename.split("_")[1] + filename.split("_")[2].split(".")[0])
            fake[difficulty].append(num)
            paths[difficulty][num] = os.path.join(dirname, filename)

train_fake = []
test_fake = []
train_real = []
test_real = []
for i in fake.keys():
    files = sorted(fake[i])
    split = int(len(files)*0.2)
    test_fake.extend([paths[i][j] for j in files[:split]])
    train_fake.extend([paths[i][j] for j in files[:split]])

files = sorted(real)
split = int(len(files)*0.2)
test_real.extend([paths["real"][j] for j in files[:split]])
train_real.extend([paths["real"][j] for j in files[:split]])

copy_data("test/real/",test_real)
copy_data("test/fake/",test_fake)
copy_data("train/real/",train_real)
copy_data("train/fake/",train_fake)
