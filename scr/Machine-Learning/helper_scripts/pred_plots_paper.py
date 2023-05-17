from training import *
from metrics import *
from dataFunctions import *
import seaborn as sns
from numpy.core.fromnumeric import mean
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from matplotlib import gridspec
from matplotlib import colors

import matplotlib.dates as mdates


# Set seed:
seed_all(SEED)

#from prediction import *



def getPixels(p, preds1, preds2, preds3, true_smb, train_set, labels, target_dataset):
        Ptarg = np.array(true_smb)[:, p["y"], p["x"], 0]
        Ppred1 = np.array(preds1)[:, p["y"], p["x"], 0]
        Ppred2 = np.array(preds2)[:, p["y"], p["x"], 0]
        Ppred3 = np.array(preds3)[:, p["y"], p["x"], 0]
    
        dfPixels = pd.DataFrame(
                data={labels[3]: Ptarg, labels[0]: Ppred1, labels[1]: Ppred2, labels[2]: Ppred3},
                index=target_dataset.time.values[len(train_set) :],
        )
        return dfPixels

def getMetricsPixels(df, labels):
    metrics = {}
    metrics['pearson'] = np.corrcoef(df[labels[0]], df[labels[3]])[0, 1]
    metrics["pearsonGCM"] = np.corrcoef(df[labels[1]], df[labels[3]])[0, 1]
    metrics["pearsonGCMtr"] = np.corrcoef(df[labels[2]], df[labels[3]])[0, 1]
    metrics["rmse"] = mean_squared_error(y_true = df[labels[3]], y_pred = df[labels[0]], squared = False)
    metrics["rmseGCM"] = mean_squared_error(y_true = df[labels[3]], y_pred = df[labels[1]], squared = False)
    metrics["rmseGCMtr"] = mean_squared_error(y_true = df[labels[3]], y_pred = df[labels[2]], squared = False)
    metrics["nrmse"] = metrics["rmse"]/(df[labels[3]].max()- df[labels[3]].min())
    metrics["nrmseGCM"] = metrics["rmseGCM"]/(df[labels[3]].max()- df[labels[3]].min())
    metrics["nrmseGCMtr"] = metrics["rmseGCMtr"]/(df[labels[3]].max()- df[labels[3]].min())
    return metrics

def increment(i, evencol, evenrow):
    i+=1
    evencol+=1
    evenrow+=1
    return i, evencol, evenrow


def plotTimeseries3Models(preds1, preds2, preds3, true_smb, train_set, target_dataset, points_RCM, region, 
                                                    cmap = 'viridis', figsize = (15, 10), labels = None, fontsize = 24):
        dsRCM = createLowerTarget(
                                target_dataset, region=region, Nx=64, Ny=64, print_=False
                        )
        fig = plt.figure(figsize=figsize)    
        colors = ['grey','#0072b2', '#009e73','#d55e00']
    
        fontsize = 20
    
        M = 4
        i, evencol, evenrow= 1, 1, 1
    
        """
        grids = {'ts': [[0, 0,2], [0, 2, 4], [3, 0,2], [3, 2, 4]],
                'ds': [[1, 0,2], [1, 2, 4], [4, 0,2], [4, 2, 4]],
                            'as': [[2, 0], [2,2], [5, 0], [5,2]],
                            'bx': [[2, 1], [2,3], [5, 1], [5,3]]   
        }"""
    
        gs = gridspec.GridSpec(4, 4, width_ratios=[2.5, 1, 2.5, 1], height_ratios= [2, 1,2, 1])
    
        grids = {'ts': [[0, 0,2], [0, 2, 4], [2, 0,2], [2, 2, 4]],
                            'as': [[1, 0], [1,2], [3, 0], [3,2]],
                            'ds': [[1, 1], [1,3], [3, 1], [3,3]]   
        }
            
    
        for p in points_RCM:
                dfPixels = getPixels(p, preds1, preds2, preds3, true_smb, train_set, labels, target_dataset)
                metricsPixels = getMetricsPixels(dfPixels, labels) # metrics
            
                # ------------------ TIMESERIES
                type_ = 'ts'
                grid = grids[type_]
                ax = plt.subplot(gs[grid[i-1][0], grid[i-1][1]:grid[i-1][2]])
                if grid[i-1][1] == 0:
                        ax.set_ylabel('[mmWe/day]', fontsize = fontsize)
                ax.set_title("Point: P{}".format(i), fontsize = fontsize)
            
                dfPixels[labels[3]].plot(label=labels[3], color=colors[0], alpha=0.5, ax = ax)
                dfPixels[labels[0]].plot(label=labels[0], color=colors[1], ax = ax, alpha = 0.8, linewidth = 1.5)
                dfPixels[labels[1]].plot(label=labels[1], color=colors[2], ax = ax, alpha = 0.8, linestyle = ':', linewidth = 2.5)
                dfPixels[labels[2]].plot(label=labels[2], color=colors[3], ax = ax, alpha = 0.8, linestyle = '--', linewidth = 2)
                if i == 4:
                        ax.legend(loc = 'upper right', ncol = 2, fontsize = 16)
                # legend text: 
                textstrUPRCM = '\n'.join((
                        r'$\mathrm{RMSE}_{U}=%.2f, \mathrm{NRMSE}_{U}=%.2f, r_{U}=%.2f$' % (metricsPixels["rmse"], metricsPixels["nrmse"], metricsPixels["pearson"], ),
                        r'$\mathrm{RMSE}_{G}=%.2f, \mathrm{NRMSE}_{G}=%.2f, r_{G}=%.2f$' % (metricsPixels["rmseGCM"], metricsPixels["nrmseGCM"], metricsPixels["pearsonGCM"], ),
                        r'$\mathrm{RMSE}_{Gtr}=%.2f, \mathrm{NRMSE}_{Gtr}=%.2f, r_{Gtr}=%.2f$' % (metricsPixels["rmseGCMtr"], metricsPixels["nrmseGCMtr"], metricsPixels["pearsonGCMtr"], )))
            
                if i == 1:
                        ax.set_ylim(top = 15)
                        ax.text(0.02, 0.95, textstrUPRCM, transform=ax.transAxes, fontsize=16,
                                verticalalignment='top')
                if i == 2:
                        ax.set_ylim(top = 15)
                        ax.text(0.02, 0.95, textstrUPRCM, transform=ax.transAxes, fontsize=16,
                                verticalalignment='top')
                if i == 3:
                        ax.set_ylim(top = 7)
                        ax.text(0.02, 0.95, textstrUPRCM, transform=ax.transAxes, fontsize=16,
                                verticalalignment='top')
                if i == 4:
                        ax.set_ylim(top = 3)
                        ax.text(0.02, 0.95, textstrUPRCM, transform=ax.transAxes, fontsize=15,
                                verticalalignment='top')
                ax.grid(axis = 'y')   
                ax.tick_params(axis='both', which='major', labelsize=16)
            
                # ------------------ ANNUAL SMB
                yearlySMB = dfPixels.resample("y").sum() # yearly sum
                yearlySMB.index = yearlySMB.index.strftime("%Y")
            
                grid = grids['as']
                ax = plt.subplot(gs[grid[i-1][0], grid[i-1][1]])
                yearlySMB.plot(
                        kind="bar", ax=ax, color=colors, alpha=0.8
                )
                ax.grid(axis = 'y')
            
                if i == 1 :
                        ax.set_ylim(bottom = -15, top = 30)
                if i == 2:
                        ax.set_ylim([-65, 5])
                if i == 3:
                        ax.set_ylim(top = 40)
                if i == 4:
                        ax.set_ylim(top = 15)
                    
                ax.set_xticklabels(yearlySMB.index, rotation=45)
                ax.tick_params(axis='both', which='major', labelsize=16)
            
                if i == 4:
                        ax.legend(loc="upper right", ncol =2, fontsize = 16)
                else:
                        ax.get_legend().remove()
                if grid[i-1][1] == 0:
                        ax.set_ylabel('[mmWe]', fontsize = fontsize)
                    
                ax.set_title("Annual SMB", fontsize = fontsize)
            
                """
                # ------------------ Boxplot SMB
                grid = grids['bx']
                ax = plt.subplot(gs[grid[i-1][0], grid[i-1][1]])
                order = ['RCM Truth', '$\mathrm{\operatorname{\hat{F}_{U}(UPRCM)}}$', '$\mathrm{\operatorname{\hat{F}_{U}(GCM)}}$', 
                    '$\mathrm{\operatorname{\hat{F}_{G}(GCM)}}$']
                im = sns.boxplot(data = yearlySMB, palette = colors, 
                                                    boxprops=dict(alpha=.8), order = order, showmeans=True, 
                                                    meanprops={
                                                "markerfacecolor":"white", 
                                                "markeredgecolor":"black",})
            
                ax.set_xticklabels(order, rotation=45)
                for violin in ax.collections[::2]:
                    violin.set_alpha(0.8)
                ax.set_title("Boxplot", fontsize = fontsize)
                # add text:
                medianGCM1 = np.median(yearlySMB[labels[1]])
                medianGCM2 = np.median(yearlySMB[labels[2]])
                medianUPRCM = np.median(yearlySMB[labels[0]])
                medianRCM = np.median(yearlySMB[labels[3]])
            
                textstrBoxplots= '\n'.join((
                        r'$\mathrm{\tilde{x}}_{U}=%.1f, \mathrm{\tilde{x}}_{R}=%.1f$' % (medianUPRCM, medianRCM, ),
                        r'$\mathrm{\tilde{x}}_{G}=%.1f, \mathrm{\tilde{x}}_{Gtr}=%.1f$' % (medianGCM1, medianGCM2, )))
            
                #ax.text(0.02, 0.95, textstrBoxplots, transform=ax.transAxes, fontsize=14,verticalalignment='top')
                ax.grid(axis = 'y')
                """
            
                # ------------------ DISTRIBUTION SMB
                grid = grids['ds']
                ax = plt.subplot(gs[grid[i-1][0], grid[i-1][1]])
                sns.kdeplot(data = dfPixels, x = labels[3], label=labels[3], color=colors[0], alpha=0.8)
                sns.kdeplot(data = dfPixels, x = labels[0], label=labels[0],color=colors[1], ax = ax, alpha = 0.8, linewidth = 1.5)
                sns.kdeplot(data = dfPixels, x = labels[1], label=labels[1], color=colors[2], ax = ax, alpha = 0.8, linestyle = ':', linewidth = 2.5)
                sns.kdeplot(data = dfPixels, x = labels[2], label=labels[2], color=colors[3], ax = ax, alpha = 0.8, linestyle = '--', linewidth = 2)
            
                """
                if i == 4:
                  ax.legend(loc="upper right", ncol =2, fontsize = 16)
                if i == 2:
                  ax.legend(loc="upper left", ncol =2, fontsize = 16)"""
            
                ax.set_title("PDF", fontsize = fontsize)
                ax.set_xlabel('[mmWe/day]', fontsize = fontsize)
                ax.set_ylabel('Density', fontsize = fontsize)
                ax.tick_params(axis='both', which='major', labelsize=16)
            
                i+=1
        plt.tight_layout()

            
def getMinMax2Metrics(metric):
    vmin = np.nanmin([metric[0], metric[1], metric[2]])
    vmax = np.nanmax([metric[0], metric[1], metric[2]])
    
    return vmin, vmax


def addMeanLegend(ax, mean,std, min, max, fontsize = 16):
    textstrBoxplots = "\n".join((r"$\mathrm{\mu}=%.2f, \mathrm{std}=%.2f$" % (mean, std),
                                r"$\mathrm{\mathrm{min}}=%.2f, \mathrm{max}=%.2f$" % (min, max),))  
    ax.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
    )
    
    
def getCorrdf(metric, labels):
    corrdf = pd.DataFrame(
        {
            labels[0]: metric[0].flatten(),
            labels[1]: metric[1].flatten(),
            labels[2]: metric[2].flatten(),
        }
    )
    return corrdf


def CompareMetrics3Models(
    PearsonCorr,
    Wasserstein,
    RMSE,
    target_dataset,
    labels,
    region: str,
    figsize=(20, 15),
    cmap="viridis",
    fontsize_axes=26,
    clb_font_size = 18,
    fontsize_ticks=20,
):
    fig = plt.figure(figsize=figsize)
    
    colors = ["#0072b2", "#009e73", "#d55e00"]
    M, i = 3, 1
    
    # Correlation
    ax1 = plt.subplot(M, 4, i, projection=ccrs.SouthPolarStereo())
    plotPearsonCorr(
        target_dataset,
        PearsonCorr[0],
        np.nanmean(PearsonCorr[0]),
        ax1,
        region=region,
        cmap=cmap,
        colorbar=False,
    )
    addMeanLegend(ax1, np.nanmean(PearsonCorr[0]), np.nanstd(PearsonCorr[0]), np.nanmin(PearsonCorr[0]), np.nanmax(PearsonCorr[0]), fontsize = fontsize_ticks)
    i += 1
    
    ax2 = plt.subplot(M, 4, i, projection=ccrs.SouthPolarStereo())
    plotPearsonCorr(
        target_dataset,
        PearsonCorr[1],
        np.nanmean(PearsonCorr[1]),
        ax2,
        region=region,
        cmap=cmap,
        colorbar=False,
    )
    addMeanLegend(ax2, np.nanmean(PearsonCorr[1]), np.nanstd(PearsonCorr[1]), np.nanmin(PearsonCorr[1]), np.nanmax(PearsonCorr[1]), fontsize = fontsize_ticks)
    
    i += 1
    ax3 = plt.subplot(M, 4, i, projection=ccrs.SouthPolarStereo())
    im = plotPearsonCorr(
        target_dataset,
        PearsonCorr[2],
        np.nanmean(PearsonCorr[2]),
        ax3,
        region=region,
        cmap=cmap,
        colorbar=False,
    )
    addMeanLegend(ax3, np.nanmean(PearsonCorr[2]), np.nanstd(PearsonCorr[2]), np.nanmin(PearsonCorr[2]), np.nanmax(PearsonCorr[2]), fontsize = fontsize_ticks)
    
    for j, ax in enumerate([ax1, ax2, ax3]):
        ax.set_title("{}: Correlation".format(labels[j]), fontsize=fontsize_axes)
    clb = fig.colorbar(im, ax=[ax1, ax2, ax3], fraction=0.046, pad=0.04)
    clb.ax.tick_params(labelsize=clb_font_size)
    clb.set_label("")
    i += 1
    
    # Boxplot:
    ax4 = plt.subplot(M, 4, i)
    corrdf = getCorrdf(PearsonCorr, labels)
    im = sns.boxplot(
        data=corrdf,
        palette=colors,
        boxprops=dict(alpha=0.8),
        showmeans=True,
        ax=ax4,
        meanprops={
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    ax4.tick_params(axis='both', which='major', labelsize=16)
    for bx in ax4.collections[::2]:
        bx.set_alpha(0.8)
    ax4.set_title("Correlation", fontsize = fontsize_axes)
    i += 1
    
    # Wasserstein:
    vmin, vmax = getMinMax2Metrics(Wasserstein)
    ax5 = plt.subplot(M, 4, i, projection=ccrs.SouthPolarStereo())
    im = plotWasserstein(
        target_dataset,
        Wasserstein[0],
        np.nanmean(Wasserstein[0]),
        ax5,
        vmin,
        vmax,
        region=region,
        cmap=cmap,
        colorbar=False,
    )
    addMeanLegend(ax5, np.nanmean(Wasserstein[0]), np.nanstd(Wasserstein[0]), np.nanmin(Wasserstein[0]), np.nanmax(Wasserstein[0]), fontsize = fontsize_ticks)
    i += 1
    ax6 = plt.subplot(M, 4, i, projection=ccrs.SouthPolarStereo())
    plotWasserstein(
        target_dataset,
        Wasserstein[1],
        np.nanmean(Wasserstein[1]),
        ax6,
        vmin,
        vmax,
        region=region,
        cmap=cmap,
        colorbar=False,
    )
    addMeanLegend(ax6, np.nanmean(Wasserstein[1]), np.nanstd(Wasserstein[1]), np.nanmin(Wasserstein[1]), np.nanmax(Wasserstein[1]), fontsize = fontsize_ticks)
    
    
    i += 1
    ax7 = plt.subplot(M, 4, i, projection=ccrs.SouthPolarStereo())
    im = plotWasserstein(
        target_dataset,
        Wasserstein[2],
        np.nanmean(Wasserstein[2]),
        ax7,
        vmin,
        vmax,
        region=region,
        cmap=cmap,
        colorbar=False,
    )
    addMeanLegend(ax7, np.nanmean(Wasserstein[2]), np.nanstd(Wasserstein[2]), np.nanmin(Wasserstein[2]), np.nanmax(Wasserstein[2]), fontsize = fontsize_ticks)
    
    
    for j, ax in enumerate([ax5, ax6, ax7]):
        ax.set_title("{}: Wasserstein".format(labels[j]), fontsize=fontsize_axes)
    l_f = LogFormatter(10, labelOnlyBase=False)
    clb = fig.colorbar(im, ax=[ax5, ax6, ax7], fraction=0.046, pad=0.04, format = l_f)
    clb.ax.tick_params(labelsize=clb_font_size)
    clb.set_label("")
    
    # boxplot:
    i += 1
    ax8 = plt.subplot(M, 4, i)
    corrdf = getCorrdf(Wasserstein, labels)
    im = sns.boxplot(
        data=corrdf,
        palette=colors,
        boxprops=dict(alpha=0.8),
        showmeans=True,
        ax=ax8,
        meanprops={
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    for bx in ax8.collections[::2]:
        bx.set_alpha(0.8)
    ax8.set_title("Wasserstein distance", fontsize = fontsize_axes)
    ax8.set_yscale("log")
    ax8.set_yticklabels([0,0, 0.1, 1, 10], minor=False)
    ax8.tick_params(axis='both', which='major', labelsize=16)
    i += 1
    
    # RMSE:
    vmin, vmax = getMinMax2Metrics(RMSE)
    ax9 = plt.subplot(M, 4, i, projection=ccrs.SouthPolarStereo())
    plotNRMSE(
        target_dataset,
        RMSE[0],
        np.nanmean(RMSE[0]),
        ax9,
        vmin,
        vmax,
        region=region,
        cmap=cmap,
        colorbar=False,
    )
    addMeanLegend(ax9, np.nanmean(RMSE[0]), np.nanstd(RMSE[0]), np.nanmin(RMSE[0]), np.nanmax(RMSE[0]), fontsize = fontsize_ticks)
    
    i += 1
    ax10 = plt.subplot(M, 4, i, projection=ccrs.SouthPolarStereo())
    plotNRMSE(
        target_dataset,
        RMSE[1],
        np.nanmean(RMSE[1]),
        ax10,
        vmin,
        vmax,
        region=region,
        cmap=cmap,
        colorbar=False,
    )
    addMeanLegend(ax10, np.nanmean(RMSE[1]), np.nanstd(RMSE[1]), np.nanmin(RMSE[1]), np.nanmax(RMSE[1]), fontsize = fontsize_ticks)
    i += 1
    ax11 = plt.subplot(M, 4, i, projection=ccrs.SouthPolarStereo())
    im = plotNRMSE(
        target_dataset,
        RMSE[2],
        np.nanmean(RMSE[2]),
        ax11,
        vmin,
        vmax,
        region=region,
        cmap=cmap,
        colorbar=False,
    )
    addMeanLegend(ax11, np.nanmean(RMSE[2]), np.nanstd(RMSE[2]), np.nanmin(RMSE[2]), np.nanmax(RMSE[2]), fontsize = fontsize_ticks)
    
    for j, ax in enumerate([ax9, ax10, ax11]):
        ax.set_title("{}: RMSE".format(labels[j]), fontsize=fontsize_axes)
    l_f = LogFormatter(10, labelOnlyBase=False)
    clb = fig.colorbar(im, ax=[ax9, ax10, ax11], fraction=0.046, pad=0.04, format = l_f)
    clb.ax.tick_params(labelsize=clb_font_size)
    clb.set_label("")
    i += 1
    
    # boxplot:
    ax12 = plt.subplot(M, 4, i)
    corrdf = getCorrdf(RMSE, labels)
    im = sns.boxplot(
        data=corrdf,
        palette=colors,
        boxprops=dict(alpha=0.8),
        showmeans=True,
        ax=ax12,
        meanprops={
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    ax12.set_title("RMSE", fontsize = fontsize_axes)
    ax12.set_yscale("log")
    ax12.set_yticklabels([0,0, 1, 10], minor=False)
    ax12.tick_params(axis='both', which='major', labelsize=16)
    for bx in ax12.collections[::2]:
        bx.set_alpha(0.8)
        
        
        
        
def plotPredictions3Models(
    preds,
    preds_GCM,
    preds_GCM_tr,
    true_smb,
    r,
    UPRCM,
    target_dataset,
    points_RCM,
    figsize=(15, 5),
    fontsize=14,
    cmap="RdYlBu_r",
    fontsize_axes = 18,
    fontsize_suptitle = 24,
):
    
    fig = plt.figure(figsize=figsize)
    map_proj = ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)
    
    # Random time:
    #randTime = rn.randint(0, len(preds) - 1)
    randTime = 4
    dt = pd.to_datetime([UPRCM.time.isel(time=randTime).values])
    time = str(dt.date[0].strftime("%m/%Y"))
    
    # GCM and RCM over domains
    dsGCM = createLowerInput(UPRCM, region="Larsen", Nx=35, Ny=25, print_=False)
    dsGCM = dsGCM.where(dsGCM.y > 0, drop=True)
    dsRCM = createLowerTarget(
        target_dataset, region="Larsen", Nx=64, Ny=64, print_=False
    )
    
    # samples at that time:
    sampletarget_, samplepred_, samplepredGCM_, samplepredGCM_tr, sampleGCM_ = (
        true_smb[randTime],
        preds[randTime],
        preds_GCM[randTime],
        preds_GCM_tr[randTime],
        np.expand_dims(dsGCM.SMB.isel(time=1).values, 2),
    )
    
    # apply mask over ice/land:
    masktarget = np.expand_dims(createMask(sampletarget_, onechannel=True), 2)
    maskGCM = np.expand_dims(createMask(sampleGCM_, onechannel=True), 2)
    
    # mean values:
    meanGCM = np.expand_dims(dsGCM.SMB.mean(dim="time").values, 2)
    meanTarget = np.array(true_smb).mean(axis=0)
    meanPred = np.array(preds).mean(axis=0)
    meanPredGCM = np.array(preds_GCM).mean(axis=0)
    meanPredGCM_tr = np.array(preds_GCM_tr).mean(axis=0)
    
    # metrics for plots:
    scUPRCM, scGCM,  scGCM_tr, rmseUPRCM, rmseGCM, rmseGCM_tr = metrics_geoplot(
        masktarget, sampletarget_, samplepred_, samplepredGCM_, samplepredGCM_tr, nrmse=False
    )
    
    meanscUPRCM, meanscGCM, meanscGCM_tr, meanrmseUPRCM, meanrmseGCM, meanrmseGCM_tr = metrics_geoplot(
        masktarget, meanTarget, meanPred, meanPredGCM, meanPredGCM_tr, nrmse=False
    )

    sampletarget_ = applyMask(sampletarget_, masktarget)
    samplepred_ = applyMask(samplepred_, masktarget)
    samplepredGCM_ = applyMask(samplepredGCM_, masktarget)
    samplepredGCM_tr = applyMask(samplepredGCM_tr, masktarget)
    sampleGCM_ = applyMask(sampleGCM_, maskGCM)
    
    meanTarget = applyMask(meanTarget, masktarget)
    meanPred = applyMask(meanPred, masktarget)
    meanPredGCM = applyMask(meanPredGCM, masktarget)
    meanPredGCM_tr = applyMask(meanPredGCM_tr, masktarget)
    meanGCM = applyMask(meanGCM, maskGCM)
    
    
    # create xarray for mean GCM smb:
    coords = {"y": dsGCM.coords["y"], "x": dsGCM.coords["x"]}
    GCM_SMB = xr.Dataset(coords=coords, attrs=dsGCM.attrs)
    GCM_SMB["Mean SMB"] = xr.Variable(
        dims=("y", "x"), data=meanGCM[:, :, 0], attrs=dsGCM["SMB"].attrs
    )
    GCM_SMB["SMB"] = xr.Variable(
        dims=("y", "x"), data=sampleGCM_[:, :, 0], attrs=dsGCM["SMB"].attrs
    )
    
    divnorm=colors.TwoSlopeNorm(vcenter=0.)
    
    # Random time plots:
    vmin, vmax = getMinMaxCB(sampleGCM_, sampletarget_, samplepred_,samplepredGCM_, samplepredGCM_tr)
    
    #----------------- RCM
    ax1 = plt.subplot(2, 5, 1, projection=ccrs.SouthPolarStereo())
    im = plotTarget(
        target_dataset, sampletarget_, ax1, vmin, vmax, region="Larsen", cmap=cmap
    )
    ax1.set_title(f"{time}: RCM (Truth)", fontsize = fontsize_axes)
    textstrBoxplots = "\n".join((r"$\mathrm{\mu}=%.1f$" % (np.nanmean(sampletarget_),),))
    ax1.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax1.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    
    #----------------- GCM
    ax2 = plt.subplot(2, 5, 2, projection=ccrs.SouthPolarStereo())
    GCM_SMB.SMB.plot(
        x="x",
        ax=ax2,
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=False,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        norm = divnorm
    )
    ax2.coastlines("10m", color="black", linewidth=1)
    ax2.set_title(f"{time}: UPRCM", fontsize = fontsize_axes)
    textstrBoxplots = "\n".join((r"$\mathrm{\mu}=%.1f$" % (GCM_SMB.SMB.mean(),),))
    ax2.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax2.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    
    #----------------- Pred Em(UPRCM) - UPRCM
    ax3 = plt.subplot(2, 5, 3, projection=ccrs.SouthPolarStereo())
    im2 = plotPred(
        target_dataset, samplepred_, ax3, vmin, vmax, region="Larsen", cmap=cmap
    )
    ax3.set_title("{}: {}".format(time, '$\mathrm{\operatorname{\hat{F}_{U}(UPRCM)}}$'), fontsize = fontsize_axes)
    textstrBoxplots = "\n".join(
        (
            r"$\mathrm{\mu}=%.1f, \mathrm{rmse}=%.2f$"
            % (np.nanmean(samplepred_), rmseUPRCM),
        )
    )
    ax3.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax3.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    
    #----------------- Pred Em(UPRCM) - GCM
    ax4 = plt.subplot(2, 5, 4, projection=ccrs.SouthPolarStereo())
    im2 = plotPred(
        target_dataset, samplepredGCM_, ax4, vmin, vmax, region="Larsen", cmap=cmap
    )
    ax4.set_title("{}: {}".format(time, '$\mathrm{\operatorname{\hat{F}_{U}(GCM)}}$'), fontsize = fontsize_axes)
    textstrBoxplots = "\n".join(
        (
            r"$\mathrm{\mu}=%.1f,\mathrm{rmse}=%.2f$"
            % (np.nanmean(samplepredGCM_), rmseGCM),
        )
    )
    ax4.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax4.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    
    #----------------- Pred Em(GCM) - GCM
    ax5 = plt.subplot(2, 5, 5, projection=ccrs.SouthPolarStereo())
    im2 = plotPred(
        target_dataset, samplepredGCM_tr, ax5, vmin, vmax, region="Larsen", cmap=cmap
    )
    ax5.set_title("{}: {}".format(time, '$\mathrm{\operatorname{\hat{F}_{G}(GCM)}}$'), fontsize = fontsize_axes)
    textstrBoxplots = "\n".join(
        (
            r"$\mathrm{\mu}=%.1f, \mathrm{rmse}=%.2f$"
            % (np.nanmean(samplepredGCM_tr), rmseGCM_tr),
        )
    )
    ax5.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax5.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    
    clb = fig.colorbar(im, ax=[ax1, ax2, ax3, ax4, ax5], fraction=0.046, pad=0.04)
    clb.set_label("SMB [mmWe/day]")
    
    """
    for ax in [ax2]:
        for p in points_RCM:
            ax.scatter(
                dsRCM.isel(x=p["x"]).x.values,
                dsRCM.isel(y=p["y"]).y.values,
                marker="x",
                s=100,
                color="red",
            )"""
    
    #----------------- Mean RCM
    # Mean values:
    vmin, vmax = getMinMaxCB(meanGCM, meanTarget, meanPred,meanPredGCM,meanPredGCM_tr)
    
    ax6 = plt.subplot(2, 5, 6, projection=ccrs.SouthPolarStereo())
    plotTarget(target_dataset, meanTarget, ax6, vmin, vmax, region="Larsen", cmap=cmap)
    ax6.set_title("Mean: RCM (Truth)", fontsize = fontsize_axes)
    textstrBoxplots = "\n".join((r"$\mathrm{\mu}=%.1f$" % (np.nanmean(meanTarget)),))
    ax6.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax6.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    
    #----------------- Mean GCM
    ax7 = plt.subplot(2, 5, 7, projection=ccrs.SouthPolarStereo())
    GCM_SMB["Mean SMB"].plot(
        x="x",
        ax=ax7,
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=False,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax7.coastlines("10m", color="black", linewidth=1)
    ax7.set_title(f"Mean: UPRCM", fontsize = fontsize_axes)
    textstrBoxplots = "\n".join((r"$\mathrm{\mu}=%.1f$" % (np.nanmean(GCM_SMB["Mean SMB"].values),),))
    ax7.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax7.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    
    #----------------- Mean Em(UPRCM) - UPRCM
    ax8 = plt.subplot(2, 5, 8, projection=ccrs.SouthPolarStereo())
    plotTarget(target_dataset, meanPred, ax8, vmin, vmax, region="Larsen", cmap=cmap)
    ax8.set_title("Mean: {}".format('$\mathrm{\operatorname{\hat{F}_{U}(UPRCM)}}$'), fontsize = fontsize_axes)
    textstrBoxplots = "\n".join(
        (
            r"$\mathrm{\mu}=%.1f, \mathrm{rmse}=%.2f$"
            % (np.nanmean(meanPred), meanrmseUPRCM),
        )
    )
    ax8.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax8.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    
    #----------------- Mean Em(UPRCM) - GCM
    ax9 = plt.subplot(2, 5, 9, projection=ccrs.SouthPolarStereo())
    imMean = plotTarget(
        target_dataset, meanPredGCM, ax9, vmin, vmax, region="Larsen", cmap=cmap
    )
    ax9.set_title("Mean: {}".format('$\mathrm{\operatorname{\hat{F}_{U}(GCM)}}$'), fontsize = fontsize_axes)
    textstrBoxplots = "\n".join(
        (
            r"$\mathrm{\mu}=%.1f, \mathrm{rmse}=%.2f$"
            % (np.nanmean(meanPredGCM), meanrmseGCM),
        )
    )
    ax9.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax9.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    
    #----------------- Mean Em(GCM) - GCM
    ax10 = plt.subplot(2, 5, 10, projection=ccrs.SouthPolarStereo())
    imMean = plotTarget(
        target_dataset, meanPredGCM_tr, ax10, vmin, vmax, region="Larsen", cmap=cmap
    )
    ax10.set_title("Mean: {}".format('$\mathrm{\operatorname{\hat{F}_{G}(GCM)}}$'), fontsize = fontsize_axes)
    textstrBoxplots = "\n".join(
        (
            r"$\mathrm{\mu}=%.1f,  \mathrm{rmse}=%.2f$"
            % (np.nanmean(meanPredGCM_tr), meanrmseGCM_tr),
        )
    )
    ax10.text(
        0.02,
        0.95,
        textstrBoxplots,
        transform=ax10.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    
    clb = fig.colorbar(imMean, ax=[ax6, ax7, ax8, ax9, ax10], fraction=0.046, pad=0.04)
    clb.set_label("SMB [mmWe/day]")
    
    #plt.suptitle(f"Random month: {time}", fontsize = fontsize_suptitle)
    
def metrics_geoplot(mask, sampletarget_, samplepred_, samplepredGCM_, samplepredGCM_tr, nrmse = False):
    # correlation:
    scUPRCM = scc(mask*sampletarget_, mask*samplepred_) # spatial correlation
    scGCM = scc(mask*sampletarget_, mask*samplepredGCM_) # spatial correlation
    scGCM_tr = scc(mask*sampletarget_, mask*samplepredGCM_tr) # spatial correlation
        
    # rmse
    rmseUPRCM = np.mean(calculateRMSE(np.expand_dims(sampletarget_, 0), np.expand_dims(samplepred_, 0), 
                                                                            normalised = nrmse, ignoreSea=True)) # rmse
    rmseGCM = np.mean(calculateRMSE(np.expand_dims(sampletarget_, 0), np.expand_dims(samplepredGCM_, 0), 
                                                                        normalised = nrmse, ignoreSea=True)) # rmse
    rmseGCM_tr = np.mean(calculateRMSE(np.expand_dims(sampletarget_, 0), np.expand_dims(samplepredGCM_tr, 0), 
                                                                        normalised = nrmse, ignoreSea=True)) # rmse
        
    return scUPRCM, scGCM,  scGCM_tr, rmseUPRCM, rmseGCM, rmseGCM_tr

def applyMask(sample, mask):
    sample = mask*sample
    sample[sample == 0] = 'nan'
    return sample

def getMinMaxCB(sampleGCM_, sampletarget_, samplepred_, samplepredGCM_, samplepredGCM_tr):
    # Get min/max for colorbar:
    vmin = np.nanmin([np.nanmin([sampletarget_, samplepred_, samplepredGCM_, samplepredGCM_tr]), np.nanmin(sampleGCM_)])
    vmax = np.nanmax([np.nanmax([sampletarget_, samplepred_, samplepredGCM_, samplepredGCM_tr]), np.nanmax(sampleGCM_)])
    return vmin, vmax 
