from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np
import sys



def get_gt_score(youtube_id,dataset):
    '''
     get the ground truth vector for all the gifs of that video
    :param youtube_id:
    :return: ground_truth_selection
    '''
    rows=dataset[dataset.youtube_id == youtube_id]
    all_start_end_frames=rows[['gif_start_frame','gif_end_frame']].get_values()
    ground_truth_selection=[]
    for start_frame,end_frame in all_start_end_frames:
        y_gt=np.zeros(int(rows['video_frame_count'].get_values()[0]))
        y_gt[int(np.round(float(start_frame))):int(np.round(float(end_frame)))]=1
        ground_truth_selection.append(y_gt)

    return ground_truth_selection


def get_ap(y_true, y_predict, interpolate=True,point_11=True):
    '''
    Average precision in different formats: (non-) interpolated and/or 11-point approximated
    point_11=True and interpolate=True corresponds to the 11-point interpolated AP used in
    the PASCAL VOC challenge up to the 2008 edition and has been verfied against the vlfeat implementation
    Also, the exact average precision corresponds to the one of vl_feat

    :param y_true: list/ numpy vector of true labels in {0,1} for each element
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation to average precision?
    :return: average precision
    '''
    if len(set(y_true))==1:
        if y_true[0]==0:
            raise ValueError('True labels cannot all be zero')
        else:
            return 1
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    recall=recall.astype(np.float32)

    if interpolate:
        for i in range(1,len(precision)):
            precision[i]=max(precision[i-1],precision[i])

    if point_11:
         precision_11=[precision[np.where(recall>=t)[0][-1]] for t in np.arange(0,1.01,0.1)]
         return np.mean(precision_11)
    else:
        indices=np.where(np.diff(recall))
        return np.mean(precision[indices])




def meaningful_summary_duration(gif_ground_truths,scores,min_inclusion=0.5,mode='mean'):
    '''
    Compute the meaningful summary duration, inspired by "Category-specific video summarization"
    We however normalize by the length of the gif and the length of the video
    :param gif_ground_truths:
    :param scores:
    :param min_inclusion: portion of the ground truth GIF that eeds to be inluded
    :param mode: Can be 'mean' (average nMSD) or 'best' (best/minimal nMSD when there are multiple GIFs for a video)
    :return: msd_score
    '''

    # Account for the fact, that some part of the gif might not be included in a segment at all!
    # Only enforce overlap of min_inclusion
    msd_scores=[]
    for y_gt in gif_ground_truths:
        gif_scores=scores[np.where(y_gt>0)[0]]
        score_threshold=np.percentile(gif_scores,100*(1-min_inclusion))
        meaningful_duration=np.sum(scores>=score_threshold)


        gif_duration = float(len(np.nonzero(y_gt)[0]))
        video_duration = float(len(scores))
        final_inclusion=np.sum(y_gt[scores>=score_threshold])/gif_duration

        # Compute normalized msd
        norm_msd = (meaningful_duration-gif_duration*final_inclusion) / (video_duration-gif_duration*final_inclusion)

        msd_scores.append(norm_msd)

    # Return the score where we handle multiple gifs according to the specified mode
    if mode=='mean':
        return np.mean(msd_scores)
    elif mode=='best':
        return np.min(msd_scores)
    else:
        raise NotImplementedError()
