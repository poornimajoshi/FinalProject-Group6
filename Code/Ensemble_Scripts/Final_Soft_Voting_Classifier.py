import pickle
import numpy as np
import torch
import os
import pandas as pd

# so the order is  resnet18, mobilenet, resnet50, vgg16
my_paths3 =['mobilenetv2_train_poornimajoshi.csv', 'resnet18_train_poornimjoshi.csv', 'densenet121_train_chirag.csv','resnet50_train_chirag.csv', 'vgg16_train_chirag.csv']
# you want to replace the list here with the list of paths directed to your predictions (pickled python lists)
source_dir = '/home/ubuntu/Deep-Learning/Final_Project/pickles'
pot_paths3 = os.listdir('/home/ubuntu/Deep-Learning/Final_Project/pickles')
#my_paths3 = [i for i in pot_paths3 if '.csv' in i ]

my_frames = [pd.read_csv(source_dir + '/' + i) for i in my_paths3]
import re
pat1 = re.compile('\.[0-9]+')
pat2 = re.compile('\.[0-9]+|0\.|1\.')

logitz = []
for x in my_frames:
    blanks = []
    for j in range(x.shape[0]):
        temp = np.array([ float(i)  for i in pat1.findall(x.Logits[j])])
        if len(temp) ==0:
            print('flamin cheetos')
            blanks.append(np.array([ float(i)  for i in pat2.findall(x.Logits[j])]))
        else:
            blanks.append(np.array([ float(i)  for i in pat1.findall(x.Logits[j])]))


    logitz.append(np.array(blanks))
# read in the model's predictions
st = logitz[0][0]
# we currently exclude #4 , or well 5, python indexing because chirag didn't have the correct model there.

chirag_logs = np.hstack((logitz[0],logitz[1],logitz[2],logitz[3],logitz[4]))
y_train = my_frames[1].TrueLabels



# read in the gound truth from list form
y_test_path = '/home/ubuntu/Deep-Learning/Final_Project/pickles/gt'
y_test = np.array(pickle.load(open(y_test_path,'rb')))

# now turn those indicators into actual predicted values..

####### THIS IS WHERE I LOAD IN THE TEST DATA WE NEED
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
no_poors = sorted(no_poors)
predz2 = read_soft(no_poors)
print(no_poors)
poors = sorted(poors)
print(poors)
# dang it
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
##############################
skL = [np.concatenate((predz2[0][i],predz2[1][i],predz2[2][i],predz2[3][i],predz2[4][i])) for i in range(predz2.shape[1])]
start = skL[0]
for i in skL[1:]:
    start = np.vstack((start,i))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
penalties = ['l1','l2']
Cs = [.01,.1,.5,1,10,50,100]
fit_intercept = ['True','False']
rs = [0]

hyperparameters = {'C':Cs, 'penalty':penalties,'random_state':rs,'fit_intercept':fit_intercept}

#skL = [np.concatenate((predz2[0][i],predz2[1][i],predz2[2][i],predz2[3][i])) for i in range(predz2.shape[1])]
#
# start = skL[0]
#
# for i in skL[1:]:
#     start = np.vstack((start,i))

GS = GridSearchCV(LogisticRegression(),hyperparameters,cv = 3)

gwid = GS.fit(chirag_logs, y_train)

print(gwid.best_estimator_)

best_Log = gwid.best_estimator_.fit(chirag_logs,y_train)

blPreds = best_Log.predict(chirag_logs)

legit_preds = best_Log.predict(start)



ne = [30,50,100,250,1000,10000]
from sklearn.ensemble import AdaBoostClassifier
hyperparameters2 = {'n_estimators':ne,'random_state':rs}

GS2 = GridSearchCV(AdaBoostClassifier(),hyperparameters2,cv = 3)

ada = GS2.fit(chirag_logs, y_train)

ada_best = ada.best_estimator_.fit(chirag_logs,y_train)

adP = ada_best.predict(start)


hyperparameters3={
'criterion': ['gini','entropy'],
'n_jobs': [-1],
'random_state':[0],
'n_estimators':[51,101,501,1001]}


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
GS3 = GridSearchCV(RandomForestClassifier(),hyperparameters3,cv = 3)

rF = GS3.fit(chirag_logs, y_train)

best_rF = rF.best_estimator_.fit(chirag_logs,y_train)

rfP = best_rF.predict(start)



eclf1 = VotingClassifier(estimators=[('lr', best_Log), ('rf', best_rF)], voting='soft')

eclf1.fit(chirag_logs,y_train)

big_predz = eclf1.predict(start)
sum(big_predz==y_test)