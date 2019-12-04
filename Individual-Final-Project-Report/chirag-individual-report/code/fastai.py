# cutout -
# contrast +
# brightness +
# dihedral +
# jitter
# perspective_warp
# rgb_randomize

from fastai.vision import *
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

learn = cnn_learner(data, models.densenet121, metrics=accuracy)
learn.model_dir='/home/ubuntu/'
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(6, max_lr=slice(3e-6,3e-4))

learn = cnn_learner(data, models.vgg16_bn, metrics=accuracy)
learn.model_dir='/home/ubuntu/'
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(10, max_lr=slice(3e-6,3e-4))

# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_top_losses(9, figsize=(15,11))
from os import listdir
from os.path import isfile, join
import pickle

def predict(img_path):
        img = open_image(img_path)
        p = learn.predict(img)
        return p[0], p[2]

mypath = "/home/ubuntu/data/project_data/cropped2/test/real/"
onlyfiles = []
onlyfiles.extend([mypath + f for f in listdir(mypath) if isfile(join(mypath, f))])
mypath = "/home/ubuntu/data/project_data/cropped2/test/fake/"
onlyfiles.extend([mypath + f for f in listdir(mypath) if isfile(join(mypath, f))])
onlyfiles = sorted(onlyfiles)
prediction = []
prediction_prob = []
categorical = {"real":1,
               "fake":0}

acc = []
for i in onlyfiles:
        ans = predict(i)
        if categorical[str(ans[0])]==categorical[i.split("/")[-2]]:
                acc.append(1)
        else:
                acc.append(0)
print("Final test set accuracy:",sum(acc)/len(acc))


for i in onlyfiles:
        ans = predict(i)
        prediction.append(categorical[str(ans[0])])
        prediction_prob.append([float(ans[1][0]), float(ans[1][1])])
        # prediction_prob_1.append()


with open('dense101_chirag_p.pickle', 'wb') as handle:
    pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('dense101_chirag_L.pickle', 'wb') as handle:
    pickle.dump(prediction_prob, handle, protocol=pickle.HIGHEST_PROTOCOL)
