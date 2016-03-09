'''
This module contains the evaluation metrics used in
 Michael Gygli, Yale Song, Liangliang Cao
    "Video2GIF: Automatic Generation of Animated GIFs from Video," IEEE CVPR 2016

 The metrics are
  - Average Precision (AP)
  - normalized Meaningful Summary Duration (nMSD)

  Furthermore it has the helper function get_gt_score, which builds the ground truth vector using
  the dataset (pandas.DataFrame object)
'''
__author__ = 'michaelgygli'
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd


def get_gt_score(youtube_id,dataset):
    '''
     get the ground truth vector for all the gifs of that video
    :param youtube_id: a youtube_id in from the dataset
    :return: ground_truth_selection: a list of label vectors
    '''
    assert isinstance(dataset,pd.DataFrame), "dataset needs to be a pandas DataFrame"

    # Get the rows for this Youtube ID
    rows=dataset[dataset.youtube_id == youtube_id]

    # Get the position of the aligned GIFs
    all_start_end_frames=rows[['gif_start_frame','gif_end_frame']].get_values()

    # For each GIF that was created from this video, create an indicator vector
    ground_truth_selection=[]
    for start_frame,end_frame in all_start_end_frames:
        y_gt=np.zeros(int(rows['video_frame_count'].get_values()[0]))
        y_gt[int(start_frame):int(end_frame)]=1
        ground_truth_selection.append(y_gt)

    return ground_truth_selection


def get_ap(y_true, y_predict, interpolate=True,point_11=False):
    '''
    Average precision in different formats: (non-) interpolated and/or 11-point approximated
    point_11=True and interpolate=True corresponds to the 11-point interpolated AP used in
    the PASCAL VOC challenge up to the 2008 edition and has been verfied against the vlfeat implementation
    The exact average precision (interpolate=False, point_11=False) corresponds to the one of vl_feat

    :param y_true: list/ numpy vector of true labels in {0,1} for each element
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation to average precision?
    :return: average precision
    '''

    # Check inputs
    assert len(y_true)==len(y_predict), "Prediction and ground truth need to be of the same length"
    if len(set(y_true))==1:
        if y_true[0]==0:
            raise ValueError('True labels cannot all be zero')
        else:
            return 1
    else:
        assert sorted(set(y_true))==[0,1], "Ground truth can only contain elements {0,1}"

    # Compute precision and recall
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    recall=recall.astype(np.float32)

    if interpolate: # Compute the interpolated precision
        for i in range(1,len(precision)):
            precision[i]=max(precision[i-1],precision[i])

    if point_11: # Compute the 11-point approximated AP
         precision_11=[precision[np.where(recall>=t)[0][-1]] for t in np.arange(0,1.01,0.1)]
         return np.mean(precision_11)
    else: # Compute the AP using precision at every additionally recalled sample
        indices=np.where(np.diff(recall))
        return np.mean(precision[indices])




def meaningful_summary_duration(gif_ground_truths,y_predict, min_inclusion=0.5,mode='mean'):
    '''
    Compute the meaningful summary duration, inspired by "Category-specific video summarization"
    We however normalize by the length of the gif and the length of the video
    :param gif_ground_truths: a list of label vectors, one for each GIF of a particular video
    :param y_predict: the predicted score for each frame of the video
    :param min_inclusion: portion of the ground truth GIF that eeds to be inluded
    :param mode: Can be 'mean' (average nMSD) or 'best' (best/minimal nMSD when there are multiple GIFs for a video)
    :return: msd_score
    '''

    # Check inputs
    assert isinstance(gif_ground_truths, list), "gif_ground_truths needs to be a list"
    for y_true in gif_ground_truths:
        assert len(y_true)==len(y_predict), "Prediction and ground truth need to be of the same length"
        if len(set(y_true))==1:
            raise ValueError('True labels cannot all be the same')
        else:
            assert sorted(set(y_true))==[0,1], "Ground truth can only contain elements {0,1}"


    # Compute MSD w.r.t. each ground truth GIF
    msd_scores=[]
    for y_true in gif_ground_truths:

        gif_scores=y_predict[np.where(y_true>0)[0]]
        min_gt_length=len(gif_scores)*min_inclusion

        # Account for the fact, that some part of the gif might not be included in a segment at all!
        # Only enforce overlap of min_inclusion
        score_threshold=np.percentile(gif_scores,100*(1-min_inclusion))

        # Compute plain MSD
        meaningful_duration=np.sum(y_predict>=score_threshold)

        # Get normalization parameters
        video_duration = float(len(y_predict))

        # How much of the ground truth GIF was actually included
        final_inclusion=np.sum(y_true[y_predict>=score_threshold])
        assert final_inclusion>=min_gt_length, "Less than min_inclusing was in the selection"

        # Compute normalized msd
        norm_msd = (meaningful_duration-min_gt_length) / (video_duration-min_gt_length)

        msd_scores.append(norm_msd)

    # Return the score where we handle multiple gifs according to the specified mode
    if mode=='mean':
        return np.mean(msd_scores)
    elif mode=='best':
        return np.min(msd_scores)
    else:
        raise NotImplementedError()
