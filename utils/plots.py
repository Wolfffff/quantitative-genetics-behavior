import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
import swifter
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from datetime import datetime

def plot_grid_interpolation(normalized_centroid_df,centroid_speed_df):
    raw_z = centroid_speed_df[:][1:].values.flatten()
    raw_x = [i[0] for i in normalized_centroid_df[1:].values.flatten()]
    raw_y = [i[1] for i in normalized_centroid_df[1:].values.flatten()]

    zi, yi, xi = np.histogram2d(raw_y, raw_x, bins=(20, 20), weights=raw_z, normed=False)
    counts, _, _ = np.histogram2d(raw_y, raw_x, bins=(20, 20))

    zi = zi / counts
    zi = np.ma.masked_invalid(zi)

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xi, yi, zi, edgecolors='black')
    # plt.scatter(raw_x, raw_y, s=1,alpha=(1/256),c='black')
    clb = plt.colorbar()
    clb.ax.set_title('Speed(px/s)')
    plt.title("Speed by Location - Set 1 Plates 123 - 2hr - " + datetime.now().strftime("%Y%m%d-%H%M"))
    plt.ylabel('Y-Coord of Centroid')
    plt.xlabel('X-Coord of Centroid')
    plt.show()

def plot_centroid_density(normalized_centroid_df):
    raw_x = [i[0] for i in normalized_centroid_df[1:].values.flatten()]
    raw_y = [i[1] for i in normalized_centroid_df[1:].values.flatten()]
    plt.figure(figsize=(10, 8))
    plt.hist2d(raw_x,raw_y,bins=20,density=True)
    clb = plt.colorbar()
    clb.ax.set_title('Density')
    plt.title("Density of Location - Set 1 Plates 123 - 2hr - " + datetime.now().strftime("%Y%m%d-%H%M"))
    plt.ylabel('Y-Coord of Centroid')
    plt.xlabel('X-Coord of Centroid')
    plt.show()

def plot_linear_interpolation(centroid_df,speed_df):
    raw_z = speed_df[:][1:].values.flatten()
    raw_x = [i[0] for i in centroid_df[1:].values.flatten()]
    raw_y = [i[1] for i in centroid_df[1:].values.flatten()]

    min_x, max_x, dim_x = (np.min(raw_x), np.max(raw_x), 25)
    min_y, max_y, dim_y = (np.min(raw_y), np.max(raw_y), 25)

    x = np.linspace(min_x, max_x, dim_x)
    y = np.linspace(min_y, max_y, dim_y)

    X, Y = np.meshgrid(x, y)
    Z = scipy.interpolate.griddata((raw_x, raw_y), raw_z, (X, Y), method='linear')
    plt.pcolormesh(X, Y, Z)
    clb = plt.colorbar()
    clb.ax.set_title('Speed(px/s)')
    plt.title("Linear Interpolation of Speed on Normalized Arenas")
    plt.ylabel('Y-Coord of Centroid')
    plt.xlabel('X-Coord of Centroid')
    # plt.savefig("/Users/wolf/git/QGB/qgb_python/plots/" + datetime.today().strftime(
    #     '%Y%m%d-%H%M') + "SpeedDistribution_Zoom.png", dpi=300)
    plt.show()

def plt_dist_speed(orientation_df, metadata):
    f, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(16, 8))
    axs[0].set_title("Control")
    axs[1].set_title("Rotenone")
    axs[2].set_title("High Sugar")
    ctrl_bool = np.where((metadata.Empty == False) & (metadata.Treatment == "C"))[0]
    ctrl = [orientation_df[i] for i in ctrl_bool]
    ctrl = pd.concat(ctrl)
    sns.distplot(ctrl, ax=axs[0], kde=False, hist=True, norm_hist=True, bins=75,
                 hist_kws={"color": "#E5FCC2", "alpha": 1, "range": (0, 200), 'edgecolor': 'black'},
                 kde_kws={"gridsize": 2048, "bw": 5, "shade": False}, label='C')
    r_bool = np.where((metadata.Empty == False) & (metadata.Treatment == "R"))[0]
    r = [orientation_df[i] for i in r_bool]
    r = pd.concat(r)
    sns.distplot(r, ax=axs[1], kde=False, hist=True, norm_hist=True, bins=75,
                 hist_kws={"color": "#45ADA8", "alpha": 1, "range": (0, 200), 'edgecolor': 'black'},
                 kde_kws={"gridsize": 2048, "bw": 5, "shade": False}, label='R')
    hs_bool = np.where((metadata.Empty == False) & (metadata.Treatment == "HS"))[0]
    hs = [orientation_df[i] for i in hs_bool]
    hs = pd.concat(hs)
    sns.distplot(hs, ax=axs[2], kde=False, hist=True, norm_hist=True, bins=75,
                 hist_kws={"color": "#594F4F", "alpha": 1, "range": (0, 200), 'edgecolor': 'black'},
                 kde_kws={"gridsize": 2048, "bw": 5, "shade": False}, label='HS')
    plt.yscale("log")
    plt.xlabel("Speed (px/s)")
    axs[1].set_ylabel("Density")
    f.suptitle("Histograms of Speed Grouped by Treatment - F1s Set 1 Plates 123 - First 8 Hours", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    # plt.savefig("/Users/wolf/git/QGB/qgb_python/plots/" + datetime.today().strftime(
    #     '%Y%m%d-%H%M') + "_f1s1p123_8h-Histogram-Speed-GroupedByTreatment-.png", dpi=300)
    plt.show()

def plot_dist_speed(speed_df):
    plt.figure(figsize=(16, 4))
    sns.distplot(speed_df, bins=np.linspace(0, 800, 1000), kde=False, norm_hist=True, color="orange")
    plt.yscale('log')
    plt.xlabel("Speed (px/s)")
    plt.ylabel('Density')
    plt.title("Histogram of Speed on Mean Part Speed - Kalman Filtered Parts - 36000 Time Points - Set 1 Plates 123 - " + datetime.today().strftime('%Y%m%d-%H%M'))
    # plt.savefig("/Users/wolf/git/QGB/qgb_python/plots/" + datetime.today().strftime('%Y%m%d-%H%M') + "SpeedDistribution_Zoom.png", dpi=300)
    plt.show()

    # flatted_speed_df = speed_df.to_numpy().flat

def plt_dist_speed_single(orientation_df, metadata):

    plt.figure(figsize=(16,6))
    ctrl_bool = np.where((metadata.Empty == False) & (metadata.Treatment == "C"))[0]
    ctrl = [orientation_df[i] for i in ctrl_bool]
    ctrl = pd.concat(ctrl)
    sns.distplot(ctrl, kde=True, hist=False, norm_hist=True, bins=75,
                 hist_kws={"color": "#E5FCC2", "alpha": 1, "range": (0, 200), 'edgecolor': 'black'},
                 kde_kws={"gridsize": 2048, "bw": 5, "shade": False}, label='C')
    r_bool = np.where((metadata.Empty == False) & (metadata.Treatment == "R"))[0]
    r = [orientation_df[i] for i in r_bool]
    r = pd.concat(r)
    sns.distplot(r, kde=True, hist=False, norm_hist=True, bins=75,
                 hist_kws={"color": "#45ADA8", "alpha": 1, "range": (0, 200), 'edgecolor': 'black'},
                 kde_kws={"gridsize": 2048, "bw": 5, "shade": False}, label='R')
    hs_bool = np.where((metadata.Empty == False) & (metadata.Treatment == "HS"))[0]
    hs = [orientation_df[i] for i in hs_bool]
    hs = pd.concat(hs)
    sns.distplot(hs, kde=True, hist=False, norm_hist=True, bins=75,
                 hist_kws={"color": "#594F4F", "alpha": 1, "range": (0, 200), 'edgecolor': 'black'},
                 kde_kws={"gridsize": 2048, "bw": 5, "shade": False}, label='HS')
    plt.yscale("log")
    plt.xlabel("Speed (px/s)")
    plt.ylabel('Density')
    plt.xlim(0,400)
    plt.ylim(1e-9,1)
    plt.title("Density Plot of Speed Grouped by Treatment - Raw -  F1s Set 1 Plates 123 - 1h", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    # plt.savefig("/Users/wolf/git/QGB/qgb_python/plots/" + datetime.today().strftime(
    #     '%Y%m%d-%H%M') + "_f1s1p123_8h-Histogram-Speed-GroupedByTreatment-.png", dpi=300)
    plt.show()

def plot_dist_radial(centroid_speed_df,rad_df):
    centroid_speed_df_masked_up = centroid_speed_df.where(rad_df > 0.9)
    centroid_speed_df_masked_down = centroid_speed_df.where(rad_df < 0.9)
    plt.figure(figsize=(16, 4))
    sns.distplot(centroid_speed_df_masked_up, bins=100, kde=True, hist=False, kde_kws={"bw": 10, "shade": False},
                 norm_hist=True, color="red", label="Edge")
    sns.distplot(centroid_speed_df_masked_down, bins=100, kde=True, hist=False, kde_kws={"bw": 10, "shade": False},
                 norm_hist=True, color="blue", label="Center")
    plt.yscale('log')
    plt.xlabel("Speed (px/s)")
    plt.ylabel('Density')
    plt.xlim(0, 300)
    plt.title(
        "Speed Distribution Edge/Center - 36000 Time Points - Set 1 Plates 123 - " + datetime.today().strftime(
            '%Y%m%d-%H%M'))
    # plt.savefig("/Users/wolf/git/QGB/qgb_python/plots/" + datetime.today().strftime('%Y%m%d-%H%M') + "SpeedDistribution_Zoom.png", dpi=300)
    plt.show()

