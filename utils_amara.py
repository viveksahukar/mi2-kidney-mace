import pickle as pkl
import numpy as np
from random import randint, sample
from sklearn.metrics import classification_report, roc_auc_score
def confidence_interval(dct):
    ##confidence interval
    """
    dct contains labels, preds, probs
    """


    #dct = pkl.load(open('../bucket/Amara/Osteo/Results/baseline_w_diameter_binned_test_6k_ann_mrn.pkl', 'rb'))
    y_test = dct['labels']
    preds = dct['preds']
    probs = dct['probs']


    target_test = y_test

    avg_precision = []
    avg_recall = []
    avg_fscore = []
    aucroc = []

    test_set_size = len(target_test)
    good_set = 0
    for i in range(1000):
        # randomly pick size of the test set
        i_size = randint(round(0.8*test_set_size), test_set_size)

        i_test_idx = np.random.choice([ii for ii in range(test_set_size)], i_size, replace=True) #sample
        i_test_idx.sort()




        i_y_test = target_test[i_test_idx]

        i_y_pred = preds[i_test_idx]
        i_y_prob = probs[i_test_idx]
#         try:
        dct = classification_report(i_y_test, i_y_pred, output_dict=True, zero_division=0)
        avg_precision.append(dct['macro avg']['precision'])
        avg_recall.append(dct['macro avg']['recall'])
        avg_fscore.append(dct['macro avg']['f1-score'])

        aucroc.append(roc_auc_score(i_y_test, i_y_prob))
        good_set+=1
#         except:
#             print('too small sample')
        if i%100==0:
            print('Iteration:\t'+str(i), good_set)

    # confidence intervals
    alpha = 0.95
    print('%.1f confidence interval ' % (alpha*100))


    p = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(avg_precision, p, axis= 0)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(avg_precision, p, axis = 0)
    print('Precision')
    print('['+str(np.round(lower*100, decimals=1))+'-'+str(np.round(upper*100, decimals=1))+']')

    p = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(avg_recall, p, axis= 0)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(avg_recall, p, axis = 0)
    print('Recall')
    print('['+str(np.round(lower*100, decimals=1))+'-'+str(np.round(upper*100, decimals=1))+']')

    p = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(avg_fscore, p, axis= 0)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(avg_fscore, p, axis = 0)
    print('F-score')
    print('['+str(np.round(lower*100, decimals=1))+'-'+str(np.round(upper*100, decimals=1))+']')

    p = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(aucroc, p, axis= 0)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(aucroc, p, axis = 0)
    print('AUC-ROC')
    print('['+str(np.round(lower*100, decimals=1))+'-'+str(np.round(upper*100, decimals=1))+']')