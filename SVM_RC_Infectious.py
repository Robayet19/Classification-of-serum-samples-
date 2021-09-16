from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import ttest_ind
import time
tic = time.time()
# set up the parameters
sample_trainfraction = 0.2  # percent of samples to be used in training the model
pep_num = 5  # number of peptides in training set : default - 500
num_rep = 1000  # number of iterations

file1 = 'sequence_data_NIBIB_Dengue_ML_CTSSeraCare_noMed_CV317-Jul-2020-00-10.csv'
file2 = 'sequence_data_NIBIB_Normal_ML_mod_CV315-Jul-2020-23-50.csv'

# read the data and sequence for two cases, or a case and the control
data_seq_1 = pd.read_csv(file1, header=None)
sequence = data_seq_1.iloc[:, 1]
total_seq = len(sequence)
data1 = data_seq_1.iloc[:, 1:]
num_sam_data1 = data1.shape[1]
data_seq_2 = pd.read_csv(file2, header=None)
data2 = data_seq_2.iloc[:, 1:]
num_sam_data2 = data2.shape[1]

data_1vs2 = np.concatenate((data1, data2), axis=1)
data_1vs2_save = data_1vs2
total_sample = data_1vs2.shape[1]

# label the samples with binary numbers
data_1vs2_label = np.zeros(total_sample, dtype=int)
for i in range(total_sample):
    if i < num_sam_data1:
        data_1vs2_label[i] = 1
data_1vs2_label_save = data_1vs2_label

# shuffle the data and their labels
ind_w_label = list(enumerate(data_1vs2_label))  # create indices for the labels and store them with labels
np.random.shuffle(ind_w_label)  # shuffle the labels along with their indices
raw_indices, data_1vs2_label = zip(*ind_w_label)  # unzipping original or raw indices of the shuffled labels
data_1vs2_label = np.asarray(data_1vs2_label)
data_1vs2 = data_1vs2[:, raw_indices]
new_indices = np.arange(0, total_sample)

numpick = np.zeros((total_sample, 1), dtype=int)
totscore = np.zeros((total_sample, 1), dtype=int)

ntrain = int(np.round(total_sample * sample_trainfraction))  # number of samples to be used in the training set

Case_ind = []
Control_ind = []
for i, j in enumerate(data_1vs2_label):
    if j == 1:
        Case_ind.append(i)
    else:
        Control_ind.append(i)
        
Peptrain_freq = np.zeros((total_seq, 1))  # keep track of indices of peptides that are being selected for training
print('classifying two datasets from array using SVM')

for i in range(num_rep):
    print('running iteration no:', i)
    # select samples from case and control at random
    CaseTrainindex = random.sample(Case_ind, int(np.round(ntrain / 2)))
    CaseTrainindex.sort()
    ControlTrainindex = random.sample(Control_ind, int(np.round(ntrain / 2)))
    ControlTrainindex.sort()
    # Merge the indices of the train samples
    Trainindex = CaseTrainindex + ControlTrainindex
    Trainindex.sort()
    # Choose the sample data that will be used in the test set
    Testindex = np.setdiff1d(new_indices, Trainindex)
    Test_data = data_1vs2[:, Testindex]
    Test_label = data_1vs2_label[Testindex]
    # Calculate p-values and sort them in ascending order
    [_, p_value] = ttest_ind(data_1vs2[:, CaseTrainindex], data_1vs2[:, ControlTrainindex], axis=1, equal_var=False)
    # p_value_w_ind = list(enumerate(p_value)) # store the p-values with indices
    ind_sorted_pval = np.argsort(p_value)
    p_value.sort()

    Peptrain_freq[ind_sorted_pval[0:pep_num]] = Peptrain_freq[ind_sorted_pval[0:pep_num]] + 1

    # Now apply a SVM algorithm
    model = svm.SVC(kernel='linear', probability=True)
    temp_traintestdata = data_1vs2[ind_sorted_pval[0:pep_num], :]  # select the train data for the selected peptides
    temp_trainlabel = list(data_1vs2_label[Trainindex])
    model.fit(np.transpose(temp_traintestdata[:, Trainindex]), data_1vs2_label[Trainindex])
    predicted_label = model.predict(np.transpose(temp_traintestdata[:, Testindex]))
    # predicted_score = model.predict_proba(np.transpose(temp_traintestdata[:, Testindex]))
    predicted_score = model.decision_function(
        np.transpose(temp_traintestdata[:, Testindex]))  # distance from the decision boundary

    ## plot roc curve for a replicate or an iteration
    # fpr, tpr, threshorld = roc_curve(data_1vs2_label[Testindex], decison_score)
    # print(f'AUC is {roc_auc_score(data_1vs2_label[Testindex], decison_score):0.3f}')
    # # plot roc curve
    # plt.plot(fpr, tpr, "b", label='Linear SVM')
    # plt.plot([0, 1], [0, 1], "k--", label='Random Guess')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.legend(loc="best")
    # plt.title("ROC Curve")
    # plt.show()

    numpick[Testindex] = numpick[Testindex] + 1
    totscore[Testindex] = totscore[Testindex] + predicted_score[:, np.newaxis]

# Calculate sensitivity and specificity
finscore = np.divide(totscore, numpick)
minfinscore = min(finscore)
maxfinscore = max(finscore)
fraction_disease = np.zeros((101, 1))
fraction_control = np.zeros((101, 1))
Accuracy = np.zeros((101, 1))

k = 0
for i in np.arange(minfinscore, maxfinscore, (maxfinscore - minfinscore) / 100):
    # use np.arange to loop through decimals
    threshold = i
    # number of disease samples classified correctly (sensitivity
    fraction_disease[k] = sum(np.logical_and((data_1vs2_label == 1)[:, np.newaxis], (finscore > threshold)))/num_sam_data1
    # number of control samples classified correctly (specificity)
    fraction_control[k] = sum(np.logical_and((data_1vs2_label == 0)[:, np.newaxis], (finscore < threshold)))/num_sam_data2
    Accuracy[k] = sum(np.logical_and((data_1vs2_label == 1)[:, np.newaxis], (finscore > threshold))) \
                  + sum(np.logical_and((data_1vs2_label == 0)[:, np.newaxis],
                                       (finscore < threshold)))/ total_sample
    k = k + 1


sensitivity = fraction_disease
specificity = fraction_control
AUC= sum((0.5*(sensitivity[1:]+sensitivity[0:-1])) * (specificity[1:]-specificity[0:-1]))
AUC_str = str('AUC = ') + str(round(float(AUC), 3))
# plot the ROC Curve
fig1, ax = plt.subplots()
ax.plot(fraction_control, fraction_disease, marker='o', color='b', label='Linear SVM')
ax.plot([1, 0], [0, 1], "k--", label='Random Guess') # add a random line
ax.invert_xaxis()
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.75, 0.1, AUC_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=props)
ax.set_xlabel("Specificity")
ax.set_ylabel("Sensitivity")
ax.legend(loc="best")
ax.set_title("ROC Curve")
plt.show()

toc = time.time()

print(f'Time to run the SVM is = {toc - tic} minutes')


