__author__ = 'michaelgygli'

import pandas as pd
import evaluation
import numpy as np

# Read csv file using pandas
# For more info on pandas check http://pandas.pydata.org/pandas-docs/stable/10min.html
dataset=pd.read_csv('../metadata.txt',sep=';\t')

# Read test IDs
with open('../testset.txt','r') as f:
    test_ids=[l.rstrip('\n') for l in f]


def evaluate_random():
    '''
     This function shows how a method can be evaluated
    '''
    all_ap=np.zeros(len(test_ids))
    all_msd=np.zeros(len(test_ids))
    for idx,youtube_id in enumerate(test_ids):
        y_gt=evaluation.get_gt_score(youtube_id,dataset)
        y_predicted=np.random.rand(len(y_gt[0]))

        all_ap[idx] = evaluation.get_ap(np.array(y_gt).max(axis=0), y_predicted, interpolate=True, point_11=True)
        all_msd[idx] = evaluation.meaningful_summary_duration(y_gt,y_predicted)

    print('Random performance: AP=%.2f%%; MSD=%.2f%%' % (100*np.mean(all_ap),100*np.mean(all_msd)))


if __name__=='__main__':
    print('Evaluate random performance')
    evaluate_random()