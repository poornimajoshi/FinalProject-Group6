import pickle
import numpy as np
import torch
import os


def read_soft(paths):
    """
    :param paths:  should be a list of paths to the lists of predictions
    """
    if 'resnet18' in paths[0]:
        first_arr = np.array( list(reversed(pickle.load(open(paths[0], 'rb')))) ).reshape(1,381,2)
    else:
        first_arr = np.array(pickle.load(open(paths[0], 'rb'))).reshape(1, 381, 2)
    #print(first_arr.shape)
    # this loads all of the predictions into a list of lists.
    for i in paths[1:]:
        #print(i)
       # print(first_arr.shape)
        #print(np.array(pickle.load(open(i, 'rb'))).shape)
        if 'resnet18' in i:
            first_arr = np.vstack((first_arr, np.array( list(reversed(pickle.load(open(i, 'rb'))))  ).reshape(1, 381, 2)))

        first_arr = np.vstack((first_arr, np.array(pickle.load(open(i, 'rb'))).reshape(1,381,2)))
    # it = iter(blank)
    # it won't work if the lists aren't all the same the length, so I raise this error.
    # the_len = len(next(it))
    # if not all(len(l) == the_len for l in it):
    #    raise ValueError('not all lists have same length!')
    return first_arr
# you want to replace the list here with the list of paths directed to your predictions (pickled python lists)
source_dir = '/home/ubuntu/Deep-Learning/Final_Project/pickles'
pot_paths2 = os.listdir('/home/ubuntu/Deep-Learning/Final_Project/pickles')
my_paths2 = []

for i in pot_paths2:
    if '_p' not in i and '_L' in i and 'drn' not in i:
        my_paths2.append(source_dir + '/'+i)
poors = []
for i in pot_paths2:
    if 'poor' in i and '_L' in i:
        poors.append(i)
no_poors = [i for i in my_paths2 if 'poornimajoshi' not in i]


# read in the model's predictions
predz2 = read_soft(no_poors)

if len(poors) ==0:
    pass
elif len(poors) ==1:
    predz2 = np.vstack((np.array(pickle.load(open(source_dir + '/' + poors[0], 'rb')).reshape(1, 381, 2), predz2)))
    #np.vstack(predz2,pickle.load(poors[0],'rb'))
elif len(poors)>1:
    for i in poors:
        print(predz2.shape)
        if 'resnet18' in i:
            print('res')
            predz2 = np.vstack((np.array(list(reversed(pickle.load(open(source_dir + '/' + i, 'rb'))))).reshape(1, 381, 2), predz2))

        else:
            predz2 = np.vstack((np.array(pickle.load(open(source_dir + '/' + i, 'rb'))).reshape(1, 381, 2), predz2))
print(len(poors))
# this is what actually gets us the indicators for the majority decision
maj2 = np.mean(predz2,axis=0)

for i in maj2:
    if i[0] == i[1]:
        print('whoo')

# read in the gound truth from list form
y_test_path = '/home/ubuntu/Deep-Learning/Final_Project/pickles/gt'
y_test = np.array(pickle.load(open(y_test_path,'rb')))

# now turn those indicators into actual predicted values..
soft_avg  = [np.argmax(i)for i in maj2]

maj_vote = sum(soft_avg ==y_test) / len(y_test)
print('the accuracy of all classifiers by soft voting is: {}%'.format(round(maj_vote*100,2)))

##############################

print("it is time for the classifier part ")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
penalties = ['l1','l2']
Cs = [.01,.1,.5,1,10,50,100]
fit_intercept = ['True','False']
rs = [0]

hyperparameters = {'C':Cs, 'penalty':penalties,'random_state':rs,'fit_intercept':fit_intercept}

skL = [np.concatenate((predz2[0][i],predz2[1][i],predz2[2][i],predz2[3][i])) for i in range(predz2.shape[1])]

start = skL[0]

for i in skL[1:]:
    start = np.vstack((start,i))

GS = GridSearchCV(LogisticRegression(),hyperparameters,cv = 3)

gwid = GS.fit(start, y_test)

print(gwid.best_estimator_)

best_Log = gwid.best_estimator_.fit(start,y_test)

blPreds = best_Log.predict(start)

ne = [30,50,100,250,1000,10000]
from sklearn.ensemble import AdaBoostClassifier
hyperparameters2 = {'n_estimators':ne,'random_state':rs}

GS2 = GridSearchCV(AdaBoostClassifier(),hyperparameters2,cv = 3)

ada = GS2.fit(start, y_test)

ada_best = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=30, random_state=0).fit(start,y_test)

adP = ada_best.predict(start)

# hyperparameters3={
# 'criterion': ['gini','entropy'],
# 'n_jobs': [-1],
# 'random_state':[0],
# 'n_estimators':[51,101,501,1001]}
#
#
# from sklearn.ensemble import RandomForestClassifier
# GS3 = GridSearchCV(RandomForestClassifier(),hyperparameters3,cv = 3)
#
# rF = GS3.fit(start, y_test)
#
# best_rF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=None, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=501,
#                        n_jobs=-1, oob_score=False, random_state=0, verbose=0,
#                        warm_start=False).fit(start,y_test)
