'''
 This example shows how to load the dataset using python and how to evaluate a method
 For each video in the test set it
  - randomly scores frames
  - computes the average precision and nMSD
'''
__author__ = 'michaelgygli'
import pandas as pd
import numpy as np
import sys


# Import v2g_evaluation
# Needs to be done from the root of the repository
# or the package is installed via python setup.py install
import v2g_evaluation


# Read csv file using pandas
#For more info on pandas check http://pandas.pydata.org/pandas-docs/stable/10min.html
dataset=pd.read_csv('metadata.txt',sep=';\t',engine='python')

# Read test IDs
with open('testset.txt','r') as f:
    test_ids=[l.rstrip('\n') for l in f]


def evaluate_random():
    '''
     This function shows how a method can be evaluated
    '''
    all_ap=np.zeros(len(test_ids))
    all_msd=np.zeros(len(test_ids))
    for idx,youtube_id in enumerate(test_ids):
        y_gt=v2g_evaluation.get_gt_score(youtube_id, dataset)
        y_predicted=np.random.rand(len(y_gt[0]))

        all_ap[idx] = v2g_evaluation.get_ap(np.array(y_gt).max(axis=0), y_predicted)
        all_msd[idx] = v2g_evaluation.meaningful_summary_duration(y_gt, y_predicted)
    print('AP=%.2f%%; MSD=%.2f%%' % (100*np.mean(all_ap),100*np.mean(all_msd)))


if __name__=='__main__':
    sys.stdout.write('Evaluate random performance\n')
    sys.stdout.flush()
    evaluate_random()