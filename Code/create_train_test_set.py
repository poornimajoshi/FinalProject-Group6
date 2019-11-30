from collections import defaultdict
import os
from shutil import copyfile
from mtcnn import MTCNN
import gc, cv2
from tqdm import tqdm
errors = []
detector = MTCNN()
def get_face(img_path, save_path):
    im = cv2.imread(img_path)
    save_path = save_path.replace("jpg", "png")
    detected = detector.detect_faces(im)
    try:
        x, y, w, h = detected[0]["box"]
        im2 = im[x:x + w + 15, y:y + h]  # +15 on width to include the chin
        cv2.imwrite(save_path, im2)
    except:
        # !wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_clip = im[y:y + h, x:x + w]  # cropping the face in image
            cv2.imwrite(save_path, cv2.resize(face_clip, (414, 511)))
        # cv2.imwrite(save_path, im)
        # errors.append([detected, img_path])
    # return im2

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

source_dir = '/home/ubuntu/data/project_data/real_and_fake_face_detection/real_and_fake_face'

for dirname, _, filenames in os.walk(source_dir):
    # print(filenames[0])
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
    split = int(len(files) * 0.2)
    test_fake.extend([paths[i][j] for j in files[:split]])
    train_fake.extend([paths[i][j] for j in files[split:]])

files = sorted(real)
split = int(len(files) * 0.2)

test_real.extend([paths["real"][j] for j in files[:split]])
train_real.extend([paths["real"][j] for j in files[split:]])

data_path = {"test/real/":test_real,
             "test/fake/": test_fake,
             "train/real/": train_real,
             "train/fake/":train_fake}
copy_data_path = "/home/ubuntu/data/project_data/format/"

# for i in data_path.keys():
#     copy_data(copy_data_path+i, data_path[i])

# copy_data("/home/ubuntu/data/project_data/format/test/real/", test_real)
# copy_data("/home/ubuntu/data/project_data/format/test/fake/", test_fake)
# copy_data("/home/ubuntu/data/project_data/format/train/real/", train_real)
# copy_data("/home/ubuntu/data/project_data/format/train/fake/", train_fake)

cropped_data_path = "/home/ubuntu/data/project_data/cropped2/"
for i in data_path.keys():
    for j in tqdm(data_path[i]):
        flname = j.split("/")[-1]
        if not os.path.exists(cropped_data_path+i):
            os.makedirs(cropped_data_path+i)
        get_face(j, cropped_data_path+i+flname)
        collected = gc.collect()
print(j)
print("DONE")
print("ERRORS:",errors)
