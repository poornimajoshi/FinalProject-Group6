import pickle
import numpy as np
import torch


def read_fun(paths):
    """
    :param paths:  should be a list of paths to the lists of predictions
    """
    first_arr = np.array(pickle.load(open(paths[0], 'rb')))

    # this loads all of the predictions into a list of lists.
    for i in paths[1:]:
        print(pickle.load(open(i, 'rb')))
        first_arr = np.vstack((first_arr, np.array(pickle.load(open(i, 'rb')))))
    # it = iter(blank)
    # it won't work if the lists aren't all the same the length, so I raise this error.
    # the_len = len(next(it))
    # if not all(len(l) == the_len for l in it):
    #    raise ValueError('not all lists have same length!')
    return first_arr
# you want to replace the list here with the list of paths directed to your predictions (pickled python lists)
my_paths = ['','','']
# read in the model's predictions
predz = read_fun(my_paths)
# this is what actually gets us the indicators for the majority decision
maj = np.mean(predz,axis=0)

# read in the gound truth from list form
y_test_path = ''
y_test = np.array(pickle.load(open(y_test_path,'rb')))

# now turn those indicators into actual predicted values..
blank = []
for i in maj:
    if i > .5:
        blank.append(1)
    elif i < .5:
        blank.append(0)
    elif i == .5:
        blank.append(int(np.random.randint(low=0, high=2, size=1)))

maj_vote = np.array(blank)
maj_acc = np.sum(y_test==maj_vote)/len(y_test)
print('the accuracy of all classifiers by hard voting is: {}'.format(round(maj_acc*100,2)))

for i in range(predz.shape[0]):
    print("The accuracy for your {}th classifier was: {}%".format(i+1,np.sum(y_test==predz[i])/len(y_test)))