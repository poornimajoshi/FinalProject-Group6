from os import listdir
from os.path import isfile, join
import pickle
from fastai.vision import *

def predict(img_path):
        img = open_image(img_path)
        p = learn.predict(img)
        lbl = categorical[img_path.split("/")[-2]]
        return [categorical[str(p[0])], p[2], lbl]

path = "/home/ubuntu/data/project_data/cropped2/train/"
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

print(data.classes, data.c, len(data.train_ds), len(data.valid_ds))

learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.model_dir='/home/ubuntu/'
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(10, max_lr=slice(3e-6,3e-4))


learn = cnn_learner(data, models.densenet101, metrics=accuracy)
learn.model_dir='/home/ubuntu/'
learn.fit_one_cycle(14, max_lr=slice(3e-6,3e-4))

mypath = "/home/ubuntu/data/project_data/cropped2/train/real/"
onlyfiles = []
onlyfiles.extend([mypath + f for f in listdir(mypath) if isfile(join(mypath, f))])
mypath = "/home/ubuntu/data/project_data/cropped2/train/fake/"
onlyfiles.extend([mypath + f for f in listdir(mypath) if isfile(join(mypath, f))])
onlyfiles = sorted(onlyfiles)
prediction = []
prediction_prob = []
categorical = {"real":1,
               "fake":0}
acc = []
result = []
for i in onlyfiles:
    ans = predict(i)
    if ans[0]==categorical[i.split("/")[-2]]:
            acc.append(1)
    else:
            acc.append(0)
    result.append(ans)
df = pd.DataFrame(result, columns=["Prediction", "Logits", "TrueLabels"])
df.to_csv("densenet121_train_chirag.csv", index=False)
