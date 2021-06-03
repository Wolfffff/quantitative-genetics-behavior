"""Classes and functions for initial processing of .slp files
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
import swifter
import shelve
import dill
from numba import jit
from tslearn import *
from itertools import chain
from tslearn.clustering import TimeSeriesKMeans
import tslearn
from tslearn.utils import to_time_series_dataset
from pykalman import KalmanFilter
from hmmlearn import hmm
import scipy
import sklearn
import datetime
import pomegranate as pg


class centroids_and_parts:
    def __init__(self):
        self.centroids = {}
        self.parts = {}


class frame:
    def __init__(self):
        self.frame_id = None
        self.num_instances = None
        self.num_skeletal_points = None
        self.coordinates = centroids_and_parts()


def get_centroid(x):
    return np.array([((x[0] + x[2]) / 2), ((x[1] + x[3]) / 2)])


def assign_bb(centroid, bbs):
    for i in range(0, bbs.shape[0]):
        current_bb = bbs[i, :]
        if centroid[0] >= current_bb[0] and centroid[0] <= (current_bb[0] + current_bb[2]) and \
                centroid[1] >= current_bb[1] and centroid[1] <= (current_bb[1] + current_bb[3]):
            return i


def centroid_mapping(df, instances, pred_points, bbs,start_frame,end_frame):
    frames_arr = [frame() for n in range(start_frame,end_frame)]
    for id in tqdm(range(start_frame,end_frame)):
        curr_ids = df.loc[id]
        curr_instances = instances[range(curr_ids[0], curr_ids[1])]
        frame_number_true = curr_instances[0]['frame_id']
        frame_number = id - start_frame
        frames_arr[frame_number].frame_id = frame_number_true
        frames_arr[frame_number].num_instances = len(range(curr_ids[0], curr_ids[1]))
        for i in range(0, len(range(curr_ids[0], curr_ids[1]))):
            curr_points = pred_points[range(curr_instances[i]['point_id_start'], curr_instances[i]['point_id_end'])][['x', 'y']]
            curr_xyx1y1 = np.array([curr_points[0][0], curr_points[0][1], curr_points[1][0], curr_points[1][1]])  # [x,y,x_1,y_1]
            curr_centroid = get_centroid(curr_xyx1y1)
            bb = assign_bb(curr_centroid, bbs)
            if not bb == None:
                if bb in frames_arr[frame_number].coordinates.centroids.keys():
                    j = 1
                    while True:
                        try:
                            distance_new = np.sqrt(((frames_arr[int(frame_number - j)].coordinates.centroids[bb][0] -
                                                     curr_centroid[0]) ** 2) + ((frames_arr[int(frame_number - j)].coordinates.centroids[bb][1] - curr_centroid[1]) ** 2))
                            distance_old = np.sqrt(((frames_arr[int(frame_number - j)].coordinates.centroids[bb][0] -
                                                     frames_arr[frame_number].coordinates.centroids[bb][0]) ** 2) + ((frames_arr[ int( frame_number - j)].coordinates.centroids[ bb][ 1] - frames_arr[ frame_number].coordinates.centroids[ bb][ 1]) ** 2))
                            break
                        except:
                            j = j + 1

                    if distance_new < distance_old:
                        frames_arr[frame_number].coordinates.centroids[bb] = curr_centroid
                        frames_arr[frame_number].coordinates.parts[bb] = curr_xyx1y1
                    else:
                        frames_arr[frame_number].coordinates.centroids[bb] = \
                            frames_arr[int(frame_number - j)].coordinates.centroids[bb]
                        frames_arr[frame_number].coordinates.parts[bb] = \
                            frames_arr[int(frame_number - j)].coordinates.parts[bb]
                else:
                    frames_arr[frame_number].coordinates.centroids[bb] = curr_centroid
                    frames_arr[frame_number].coordinates.parts[bb] = curr_xyx1y1
    return frames_arr


def remove_zero_entries(centroid_df):
    return centroid_df.ffill().bfill()



def centroid_speed(centroid_df,hz=10):
    centroid_df=centroid_df.swifter.progress_bar(enable=True).apply(speed_generator_centroid)
    return(centroid_df)

def speed_generator_centroid(centroid_col, hz=10):
    curr_x = pd.DataFrame([item[0] for item in centroid_col]).diff()
    curr_y = pd.DataFrame([item[1] for item in centroid_col]).diff()
    speed_df = pd.DataFrame(np.sqrt(curr_x ** 2 + curr_y ** 2))[0] * hz
    return speed_df

def orientation_generator(parts_df):
    orientation_df = pd.DataFrame()
    for r in tqdm(parts_df.keys()):
        curr_head = pd.DataFrame([item[range(2)] for item in parts_df[r]])
        curr_abd = pd.DataFrame([item[range(2,4)] for item in parts_df[r]])
        difference = np.subtract(curr_head,curr_abd)
        orientation_df[r] = pd.DataFrame(np.arctan2(difference[1], difference[0]))[0]
    return orientation_df


def heading_generator(centroid_df):
    heading_df = pd.DataFrame()
    for r in tqdm(centroid_df.keys()):
        curr_x = pd.DataFrame([item[0] for item in centroid_df[r]]).diff()
        curr_y = pd.DataFrame([item[1] for item in centroid_df[r]]).diff()
        heading_df[r] = pd.DataFrame(np.arctan2(curr_y, curr_x))[0]
    return heading_df

def body_length_generator(parts_df):
    bl_df = pd.DataFrame()
    for r in tqdm(parts_df.keys()):
        curr_head = pd.DataFrame([item[range(2)] for item in parts_df[r]])
        curr_abd = pd.DataFrame([item[range(2,4)] for item in parts_df[r]])
        difference = np.subtract(curr_head,curr_abd)
        bl_df[r] = pd.DataFrame(np.sqrt(difference[0] ** 2 + difference[1] ** 2))[0]
    return bl_df

def mean_parts_speed(parts_df, hz=10):
    speed_df=parts_df.swifter.progress_bar(enable=True).apply(speed_generator_parts)
    return speed_df

def speed_generator_parts(x, hz = 10):
    head_x = pd.DataFrame([item[0] for item in x]).diff()
    head_y = pd.DataFrame([item[1] for item in x]).diff()
    abd_x = pd.DataFrame([item[2] for item in x]).diff()
    abd_y = pd.DataFrame([item[3] for item in x]).diff()
    speed_series = pd.DataFrame((np.sqrt(head_x ** 2 + head_y ** 2) + np.sqrt(abd_x ** 2 + abd_y ** 2)) / 2)[0] * hz
    return(speed_series)

def rle(met_df, threshold = 7,hz = 10):
    rle_dict = {}
    for i in tqdm(met_df.keys()):
        temp = pd.DataFrame([np.array((k, sum(1 for i in g))) for k,g in groupby((np.array(met_df[i]) > threshold).astype(int))])
        temp['start'] = np.cumsum(temp[1].shift(1))
        temp.loc[0,'start'] = 0
        temp['start'].astype(int)
        temp['end'] = np.cumsum(temp[1]).astype(int)
        temp['indices'] = [range(r['start'].astype(int),r['end'].astype(int)) for i,r in temp.iterrows()]
        temp.columns = ['state','length','start','end','indices']
        rle_dict[i] = temp

    return rle_dict



def normalize_parts_df(parts_df, hz = 10):
    pdf = parts_df.copy()
    for r in tqdm(pdf.keys()):
        curr_x = np.array([item[[0,2]] for item in pdf[r]])
        curr_y = np.array([item[[1, 3]] for item in pdf[r]])
        x_diff = curr_x.max() - curr_x.min()
        y_diff = curr_y.max() - curr_y.min()

        head_x = 2*((pd.DataFrame([item[0] for item in pdf[r]])-curr_x.min())/x_diff) - 1
        head_y = 2*((pd.DataFrame([item[1] for item in pdf[r]])-curr_y.min())/y_diff) - 1
        abd_x = 2*((pd.DataFrame([item[2] for item in pdf[r]])-curr_x.min())/x_diff) - 1
        abd_y = 2*((pd.DataFrame([item[3] for item in pdf[r]])-curr_y.min())/y_diff) - 1

        pdf[r] = pd.Series(list(np.array([head_x[0][:],head_y[0][:],abd_x[0][:],abd_y[0][:]]).transpose()))

    return(pdf)

def normalize_centroid_df(centroid_df, hz = 10):
    cdf = centroid_df.copy()
    for r in tqdm(cdf.keys()):
        curr_x = np.array([item[0] for item in cdf[r]])
        curr_y = np.array([item[1] for item in cdf[r]])
        x_diff = curr_x.max() - curr_x.min()
        y_diff = curr_y.max() - curr_y.min()

        cent_x = 2*((pd.DataFrame([item[0] for item in cdf[r]])-curr_x.min())/x_diff) - 1
        cent_y = 2*((pd.DataFrame([item[1] for item in cdf[r]])-curr_y.min())/y_diff) - 1
        cdf[r] = pd.Series(list(np.array([cent_x[0][:],cent_y[0][:]]).transpose()))

    return(cdf)


def shelf_current():
    my_shelf = shelve.open("/tmp/shelf.out", 'n')  # 'n' for new
    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()


def shelf_restore(filename="/tmp/shelf.out"):
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        globals()[key] = my_shelf[key]
    my_shelf.close()


def metrics_from_rle(rle_dict,speed_df):
    mean_within_bout_speed_df = {}
    mean_movement_length = {}
    mean_nonmovement_length = {}
    mvnr = {}

    for i in tqdm(rle_dict.keys()):
        ranges = [item for items in rle_dict[i]['indices'][np.where(rle_dict[i][0] == 1)[0]] for item in items]
        length_of_movement_indices = [len(item) for item in rle_dict[i]['indices'][np.where(rle_dict[i][0] == 1)[0]]]
        length_of_nonmovement_indices = [len(item) for item in rle_dict[i]['indices'][np.where(rle_dict[i][0] == 0)[0]]]
        indices = [speed_df[i][j] for j in ranges]
        mean_movement_length[i] = np.mean(length_of_movement_indices)
        mean_nonmovement_length[i] = np.mean(length_of_nonmovement_indices)
        mean_within_bout_speed_df[i] = np.mean(indices)
        mvnr[i] = len(length_of_movement_indices)

    return mean_within_bout_speed_df,mean_movement_length, mean_nonmovement_length, mvnr

def rotate(row,theta):
    r = np.array(((np.cos(theta), -np.sin(theta)),
                  (np.sin(theta), np.cos(theta))))
    return r.dot(row)

def metrics_from_rle_extended(rle_dict,metric_df,metric_df_2,orientation,min_len = 10,max_len=30):
    metric_df_group = {}
    metric_df_group_2 = {}
    indices = {}
    for i in tqdm(rle_dict.keys()):
        in_bout_indices = rle_dict[i]['indices'][np.where(rle_dict[i][0] == 1)[0]]
        ctr = [list(grp) for grp in in_bout_indices]
        metric_df_group[i] = list()
        metric_df_group_2[i] = list()
        indices[i] = list()
        for j in range(len(ctr)):
            met_1 = np.array([metric_df[i][k] for k in ctr[j]])
            met_2 = np.array([metric_df_2[i][k] for k in ctr[j]])

            theta = np.array([orientation[i][k] for k in ctr[j]])[0]

            merged = np.array([met_1,met_2]).transpose()
            merged = np.apply_along_axis(rotate, 1, np.array(merged), theta=-theta + np.pi / 2)

            met_1 = np.array([i[0] for i in merged.tolist()])
            met_1 = met_1 - met_1[0]
            metric_df_group[i].append(met_1.tolist())

            met_2 = np.array([i[1] for i in merged.tolist()])
            met_2 = met_2 - met_2[0]
            metric_df_group_2[i].append(met_2.tolist())
            indices[i].append(ctr[j])

    # unlisted_metrics = [item for k, sublist in metric_df_group.items() for item in sublist]
    # unlisted_metrics_2 = [item for k, sublist in metric_df_group_2.items() for item in sublist]

    # unlist metrics and get keys for comparing to metadata later
    unlisted_metrics = [item for k,sublist in metric_df_group.items() for item in sublist]
    unlisted_metrics_2 = [item for k, sublist in metric_df_group_2.items() for item in sublist]

    keys = [k for k, sublist in metric_df_group.items() for item in sublist]
    indices = [indices[k] for k, sublist in metric_df_group.items()]
    indices = [item for sublist in indices for item in sublist]



    # unlisted_metrics_2 = [item for k, sublist in metric_df_group_2.items() for item in sublist]

    arr = np.array([np.array(x) for x in np.array(unlisted_metrics)[np.where([(len(i) >= min_len and len(i) <= max_len) for i in unlisted_metrics])[0]]])
    arr_2 = np.array([np.array(x) for x in np.array(unlisted_metrics_2)[np.where([(len(i) >= min_len and len(i) <= max_len) for i in unlisted_metrics])[0]]])
    # arr.shape
    # arr_2.shape

    key_arr = np.array([k for k in np.array(keys)[np.where([(len(i) >= min_len and len(i) <= max_len) for i in unlisted_metrics])[0]]])
    indices_arr = np.array([k for k in np.array(indices)[np.where([(len(i) >= min_len and len(i) <= max_len) for i in unlisted_metrics])[0]]])

    tsd = to_time_series_dataset(arr)
    tsd_2 = to_time_series_dataset(arr_2)

    tsd_merged = np.dstack((tsd,tsd_2))

    # tsd_merged = tslearn.preprocessing.TimeSeriesScalerMeanVariance().fit_transform(tsd_merged)
    tsd_merged = tslearn.preprocessing.TimeSeriesResampler(10).fit_transform(tsd_merged)
    print(tsd_merged.shape)
    km = TimeSeriesKMeans(n_clusters=15, metric="softdtw", verbose=True, n_jobs=-1,max_iter=3)
    y_pred = km.fit_predict(tsd_merged)
    return key_arr,indices_arr,tsd_merged,km,y_pred


def generate_phi(orientation_df, heading_df):
    phi_df = pd.DataFrame()
    for i in tqdm(orientation_df.keys()):
        curr_heading = heading_df[i]
        curr_orientation = orientation_df[i]
        # arr = np.array([curr_orientation,curr_heading]).transpose()
        curr_phi = pd.Series(0)
        curr_phi = curr_phi.append(np.arctan2(np.sin(curr_heading[1:] - curr_orientation.shift(1)[1:]), np.cos(curr_heading[1:] - curr_orientation.shift(1)[1:])))

        phi_df[i] = curr_phi
    return phi_df

def extract_0(c):
    return [i[0] for i in c]

def extract_1(c):
    return [i[1] for i in c]

def generate_v_parperp(centroid_df, hz = 10):
    pp = centroid_df.apply(rolling_apply_vparperp_speeditup)
    vpar = pp.swifter.apply(extract_0)
    vperp = pp.swifter.apply(extract_1)
    return vpar, vperp


def rolling_apply_vparperp(c,hz=10,window=1):
    temp = pd.DataFrame(item for item in c)
    temp = pd.Series(v_parperp(temp.iloc[i-1:i + 2]) for i in range(1,len(temp)-1))
    temp = pd.concat([pd.Series([[0,0],[0,0]]),temp],ignore_index=True)
    return temp.transpose()

def rolling_apply_vparperp_speeditup(c,hz=10):
    temp = pd.DataFrame(item for item in c)
    df = pd.concat([temp,temp.shift(1),temp.shift(2)],axis=1).T.reset_index(drop=True).T
    temp_vpar = (((df.iloc[:,2] - df.iloc[:,4]) * (df.iloc[:,0] - df.iloc[:,2])) + ((df.iloc[:,1] - df.iloc[:,3]) * (df.iloc[:,3] - df.iloc[:,5]))) / (np.sqrt((df.iloc[:,2] - df.iloc[:,4]) ** 2 + (df.iloc[:,3] - df.iloc[:,5]) ** 2))*hz
    temp_vperp = (((df.iloc[:,3] - df.iloc[:,5]) * (df.iloc[:,0] - df.iloc[:,2])) - ((df.iloc[:,1] - df.iloc[:,3]) * (df.iloc[:,2] - df.iloc[:,4]))) / (np.sqrt((df.iloc[:,2] - df.iloc[:,4]) ** 2 + (df.iloc[:,3] - df.iloc[:,5]) ** 2))*hz

    # temp_vpar = (((df.iloc[:,4] - df.iloc[:,2]) * (df.iloc[:,2] - df.iloc[:,0])) + ((df.iloc[:,5] - df.iloc[:,3]) * (df.iloc[:,3] - df.iloc[:,1]))) / (np.sqrt((df.iloc[:,2] - df.iloc[:,0]) ** 2 + (df.iloc[:,3] - df.iloc[:,1]) ** 2))*hz
    # temp_vperp = (((df.iloc[:,3] - df.iloc[:,1]) * (df.iloc[:,4] - df.iloc[:,2])) - ((df.iloc[:,5] - df.iloc[:,3]) * (df.iloc[:,2] - df.iloc[:,0]))) / (np.sqrt((df.iloc[:,2] - df.iloc[:,0]) ** 2 + (df.iloc[:,3] - df.iloc[:,1]) ** 2))*hz
    temp_vpar= temp_vpar
    temp_vperp = temp_vperp
    return pd.Series(list(np.array([temp_vpar,temp_vperp]).transpose()))

def v_parperp(d,hz=10):
    try:
        p_0 = d.iloc[0]
        p_1 = d.iloc[1]
        p_2 = d.iloc[2]
        temp_vpar = (((p_2[0] - p_1[0]) * (p_1[0] - p_0[0])) + ((p_2[1] - p_1[1]) * (p_1[1] - p_0[1]))) / (np.sqrt((p_1[0] - p_0[0]) ** 2 + (p_1[1] - p_0[1]) ** 2))
        temp_vperp = (((p_1[1] - p_0[1]) * (p_2[0] - p_1[0])) - ((p_2[1] - p_1[1]) * (p_1[0] - p_0[0]))) / (np.sqrt((p_1[0] - p_0[0]) ** 2 + (p_1[1] - p_0[1]) ** 2))
        return np.array([temp_vpar,temp_vperp])
    except:
        return np.array([np.nan,np.nan])


def apply_fit_1(x,model):
    return [np.nan] + model.predict(np.array(x[1:]).reshape(-1,1)).tolist()

def apply_fit_generic_skip1_append1(x,model):
    pred = model.predict(np.stack(x)[1:])
    return [np.nan] + pred.tolist() + [np.nan]

def apply_fit(x,model):
    return model.predict(np.array(x).reshape(-1,1)).tolist()

def threshold(x, threshold_float = 7):
    return (np.array(x) > threshold_float).astype(int)

def personality_vector(x,uniq):
    pvec = {}
    for i in uniq:
        try:
            pvec[i] = len(np.where(x==i)[0])/len(x)
        except:
            pvec[i] = 0
    return np.array(list(pvec.values()))


def kal_smooth_1_1(c):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(c[1:],n_iter=3)
    smoothed = np.array(kf.smooth(c[1:])[0])
    x = np.insert(smoothed,0,0,axis=0)
    return x

def kal_smooth_coordinates(c,kf,hz=10):
    measurements = np.asarray([tuple(i) for i in c])
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
    kf_df = pd.DataFrame(smoothed_state_means)
    kf_df.iloc[:,1] = kf_df.iloc[:,1]*hz
    kf_df.iloc[:, 3] = kf_df.iloc[:,3]*hz
    return kf_df
def kal_smooth_coordinates_parts(c,kf,hz=10):
    measurements = np.asarray([tuple(i) for i in c])
    kf_temp = KalmanFilter(observation_matrices=kf.observation_matrices, initial_state_mean = [measurements[0, 0], 0, measurements[0, 1], 0, measurements[0, 2],0,measurements[0, 3],0],transition_matrices=kf.transition_matrices,observation_covariance=kf.observation_covariance,initial_state_covariance=kf.initial_state_covariance)
    (smoothed_state_means, smoothed_state_covariances) = kf_temp.smooth(measurements)
    kf_df = pd.DataFrame(smoothed_state_means)
    kf_df.iloc[:,1] = kf_df.iloc[:,1]*hz
    kf_df.iloc[:, 3] = kf_df.iloc[:,3]*hz
    kf_df.iloc[:, 5] = kf_df.iloc[:, 5] * hz
    kf_df.iloc[:, 7] = kf_df.iloc[:, 7] * hz
    return kf_df

def centroid_generator(parts_df, hz=10):
    cent_out = pd.DataFrame()
    for r in tqdm(parts_df.keys()):
        curr_x = pd.DataFrame([item[[0,2]] for item in parts_df[r]]).mean(axis=1)
        curr_y = pd.DataFrame([item[[1,3]] for item in parts_df[r]]).mean(axis=1)
        cent_out[r] = pd.Series(list(np.array([curr_x,curr_y]).transpose()))
    return cent_out

def hmm_movement(metric_df,n_comp,threshold_num):
    movement_bool = metric_df.apply(threshold,threshold_float=threshold_num)

    hmm_lens = np.array([len(movement_bool[k]) for k in movement_bool.keys()])
    hmm_dset= np.array([movement_bool[k] for k in movement_bool.keys()])
    hmm_dset = np.concatenate(hmm_dset)


    model = hmm.MultinomialHMM(n_components=2)
    model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
    model.fit(hmm_dset.reshape(-1, 1),hmm_lens)


    hmm_out_movement_alone = movement_bool.apply((lambda x: apply_fit(x,model)))
    return hmm_out_movement_alone

def hmm_movement_gaussian(metric_df,n_comp):
    hmm_lens = np.array([len(metric_df[k]) for k in metric_df.keys()])
    hmm_dset_1 = np.array([metric_df[k] for k in metric_df.keys()])
    # hmm_dset_2 = np.array([vel_perp_[k][2:] for k in movement_bool.keys()])

    hmm_dset_1[np.where(np.isnan(hmm_dset_1))] = 0
    # hmm_dset_2[np.where(np.isnan(hmm_dset_2))] = 0

    hmm_dset_1 = hmm_dset_1.reshape(-1, 1)
    # hmm_dset_2 = hmm_dset_2.reshape(-1, 1)


    # hmm_dset = np.hstack([np.array(hmm_dset_1),np.array(hmm_dset_2)])
    # np.where(np.isnan(hmm_dset))

    model = hmm.GaussianHMM(n_components=n_comp)
    model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])

    model.fit(hmm_dset_1,hmm_lens)

    hmm_out = metric_df.apply((lambda x: apply_fit_1(x,model)))
    return hmm_out

def hmm_movement_gmmhmm(metric_df,n_comp,n_mix):
    hmm_lens = np.array([len(metric_df[k]) for k in metric_df.keys()])
    hmm_dset_1 = np.array([metric_df[k] for k in metric_df.keys()])

    hmm_dset_1[np.where(np.isnan(hmm_dset_1))] = 0

    hmm_dset_1 = hmm_dset_1.reshape(-1, 1)

    model = hmm.GMMHMM(n_components=n_comp, n_mix=3)
    model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
    model.fit(hmm_dset_1,hmm_lens)

    hmm_out = metric_df.apply((lambda x: apply_fit_1(x,model)))
    return hmm_out

def hmm_movement_gaussian_multiple(metric_list,n_comp):
    metric_count = len(metric_list)
    hmm_dsets = list()
    hmm_scalers = list()
    hmm_lens = np.array([len(metric_list[0][k]) for k in metric_list[0].keys()])
    for metric_df in metric_list:
        hmm_dset = np.array([metric_df[k] for k in metric_df.keys()])
        hmm_dset[np.where(np.isnan(hmm_dset))] = 0
        hmm_dset = hmm_dset.reshape(-1, 1)
        # scaler = sklearn.preprocessing.StandardScaler().fit(hmm_dset)
        # hmm_dset = scaler.transform(hmm_dset)
        # hmm_scalers.append(scaler)
        hmm_dsets.append(hmm_dset)

    hmm_dset_agg = np.hstack([np.asarray(i) for i in hmm_dsets])

    model = hmm.GaussianHMM(n_components=n_comp)
    # model.transmat_ = np.array([[0.99, 0.01], [0.01, 0.99]])

    model.fit(hmm_dset_agg,hmm_lens)

    pred = pd.DataFrame()
    for i in metric_list[0].keys():
        l = list()
        for k in range(len(metric_list)):
            l.append(np.array(metric_list[k][i]).reshape(-1, 1))
        pred[i] = pd.Series(list(np.array(np.hstack(l))))
    hmm_out = pred.apply((lambda x: apply_fit_generic_skip1_append1(x[:-1],model)))
    # hmm_out = pd.DataFrame.from_dict(dict(zip(hmm_out.index, hmm_out.values)))
    return hmm_out, model

def radial_distribution(normalized_cent_df):
    rad_df = pd.DataFrame()
    for k in tqdm(normalized_cent_df.keys()):
        col = normalized_cent_df[k]
        arr = np.vstack(col)
        sq = np.square(arr)
        sm = np.sum(sq,axis=1)
        sqrt = np.sqrt(sm)
        rad_df[k] = pd.Series(sqrt)
    return rad_df

# https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
def get_angular_diff(c):
    return np.concatenate([np.array([0]), np.arctan2(np.sin(c[1:] - c.shift(1)[1:]), np.cos(c[1:] - c.shift(1)[1:]))])

def append_0_apply(c,fun_to_apply):
    d = fun_to_apply(c)
    return [0] + d.tolist()


def pomegranate_hmm(metric_list,distribution=pg.MultivariateGaussianDistribution,n_comp=9,model=None):
    sequences = list()
    if type(metric_list) == pd.core.frame.DataFrame:
        for i in metric_list.keys():
            sequences.append(np.array(metric_list[i]))
        if(model == None):
            model = pg.HiddenMarkovModel.from_samples(distribution, n_components=n_comp, X=sequences, n_jobs=-1,                                             algorithm='viterbi')

        df = pd.DataFrame()
        for i in metric_list.keys():
            df[i] = model.predict(np.array(metric_list[i]))
    else:
        for i in metric_list[0].keys():
            sequences.append(np.stack([np.array(metric[i]) for metric in metric_list]).T)
        if (model == None):
            model = pg.HiddenMarkovModel.from_samples(distribution, n_components=n_comp, X=sequences, n_jobs=-1,                                                  algorithm='viterbi')

        df = pd.DataFrame()
        for i in metric_list[0].keys():
            df[i] = model.predict(np.stack([np.array(metric[i]) for metric in metric_list]).T)
    return df, model



def hmm_movement_gmm_multiple(metric_list,n_comp):
    metric_count = len(metric_list)
    hmm_dsets = list()
    hmm_scalers = list()
    hmm_lens = np.array([len(metric_list[0][k]) for k in metric_list[0].keys()])
    for metric_df in metric_list:
        hmm_dset = np.array([metric_df[k] for k in metric_df.keys()])
        hmm_dset[np.where(np.isnan(hmm_dset))] = 0
        hmm_dset = hmm_dset.reshape(-1, 1)
        # scaler = sklearn.preprocessing.StandardScaler().fit(hmm_dset)
        # hmm_dset = scaler.transform(hmm_dset)
        # hmm_scalers.append(scaler)
        hmm_dsets.append(hmm_dset)

    hmm_dset_agg = np.hstack([np.asarray(i) for i in hmm_dsets])

    model = hmm.GMMHMM(n_components=n_comp)
    # model.transmat_ = np.array([[0.99, 0.01], [0.01, 0.99]])

    model.fit(hmm_dset_agg,hmm_lens)

    pred = pd.DataFrame()
    for i in metric_list[0].keys():
        l = list()
        for k in range(len(metric_list)):
            l.append(np.array(metric_list[k][i]).reshape(-1, 1))
        pred[i] = pd.Series(list(np.array(np.hstack(l))))
    hmm_out = pred.apply((lambda x: apply_fit_generic_skip1_append1(x[:-1],model)))
    # hmm_out = pd.DataFrame.from_dict(dict(zip(hmm_out.index, hmm_out.values)))
    return hmm_out, model


def rle_new(met_df, threshold = 7,hz = 10):
    rle_dict = {}
    for i in tqdm(met_df.keys()):
        temp = pd.DataFrame([np.array((k, sum(1 for i in g))) for k,g in groupby((np.array(met_df[i])).astype(int))])
        temp['start'] = np.cumsum(temp[1].shift(1))
        temp.loc[0,'start'] = 0
        temp['start'].astype(int)
        temp['end'] = np.cumsum(temp[1]).astype(int)
        temp['indices'] = [range(r['start'].astype(int),r['end'].astype(int)) for i,r in temp.iterrows()]
        temp.columns = ['state','length','start','end','indices']
        rle_dict[i] = temp
    return rle_dict

def curr_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")


def extract_head(parts_df):
    head_df = parts_df.swifter.progress_bar(enable=True).apply(extract_01)
    return head_df
def extract_abd(parts_df):
    abd_df = parts_df.swifter.progress_bar(enable=True).apply(extract_23)
    return abd_df

def extract_01(c):
    return [i[0:2] for i in c]
def extract_23(c):
    return [i[2:4] for i in c]


def vpar_perp_orientation_ref(centroid,parts,hz=10):
    vparperp = pd.DataFrame()
    for fly in tqdm(centroid.keys()):
        vparperp[fly] = vparperp_orientation_reference_cols(centroid[fly],parts[fly],hz=hz)
    vpar = vparperp.swifter.apply(extract_0)
    vperp = vparperp.swifter.apply(extract_1)
    return vpar, vperp

def vparperp_orientation_reference_cols(centroid,parts,hz=10):
    temp_centroid = pd.DataFrame(item for item in centroid)
    temp_parts = pd.DataFrame(item for item in parts)
    temp_parts[0] = temp_parts[0] - temp_parts[2]
    temp_parts[1] = temp_parts[1] - temp_parts[3]
    temp_parts = temp_parts[[0,1]]

    df = pd.concat([temp_centroid,temp_centroid.shift(1),temp_parts.shift(1)],axis=1).T.reset_index(drop=True).T
    temp_vpar = (((df.iloc[:,0]-df.iloc[:,2])*(df.iloc[:,4])) + ((df.iloc[:,1] - df.iloc[:,3])*(df.iloc[:,5]))) / (np.sqrt((df.iloc[:,4]) ** 2 + (df.iloc[:,5]) ** 2))*hz
    temp_vperp = (((df.iloc[:,5])*(df.iloc[:,0] - df.iloc[:,2])) - ((df.iloc[:,1] - df.iloc[:,3])*(df.iloc[:,4]))) / (np.sqrt((df.iloc[:,4]) ** 2 + (df.iloc[:,5]) ** 2))*hz
    return pd.Series(list(np.array([temp_vpar,temp_vperp]).transpose()))


def generate_curvature(centroid_df, vel_perp):
    curv = pd.DataFrame()
    for fly in tqdm(centroid_df.keys()):
        curv[fly] = generate_curvature_cols(centroid_df[fly], vel_perp[fly])
    return curv

def generate_curvature_cols(centroids,vperps):
    centroid_x = np.asarray([c[0] for c in centroids])
    centroid_y = np.asarray([c[1] for c in centroids])
    diff_x = np.diff(centroid_x,prepend=centroid_x[0])
    diff_y = np.diff(centroid_y,prepend=centroid_y[0])
    diff_x_shift = pd.Series(diff_x).shift(1).to_numpy()
    diff_y_shift = pd.Series(diff_y).shift(1).to_numpy()
    curv = vperps.to_numpy()/(np.square(diff_x_shift) + np.square(diff_y_shift))
    return curv

