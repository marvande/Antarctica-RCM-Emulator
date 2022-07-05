from training import *
from metrics import *
from dataFunctions import *
import seaborn as sns
from numpy.core.fromnumeric import mean
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
# Set seed:
seed_all(SEED)


def predict(net, device, test_set, model, dir_checkpoint: str = Path("./checkpoints/"), log_:bool = True):
    if log_:
        logging.info(f"Loading saved model {model}")
        logging.info(f"Using device {device}")

    # test loader:
    MSE = nn.MSELoss()
    loader_args = dict(
        batch_size=1, num_workers=4, pin_memory=True, worker_init_fn=seed_worker
    )
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # Load saved pt model
    net.to(device=device)
    net.load_state_dict(torch.load(str(model), map_location=device))
    if log_:
        logging.info("Saved model loaded!")

    preds, x, y, z, r = [], [], [], [], []

    pred_mse, pred_nrmse_, pred_rmse =0,0,0
    
    """for batch in tqdm(
        test_loader,
        x
    ):"""
    for batch in test_loader:
        seed_all(SEED)
        X_test, Z_test, Y_test, R_test = batch[0], batch[1], batch[2], batch[3]
        X_test_cuda = X_test.to(device=device, dtype=torch.float32)  # send to device
        Z_test_cuda = Z_test.to(device=device, dtype=torch.float32)  # send to device
        true_smb = Y_test.to(device=device, dtype=torch.float32)  # send to device

        net.eval()
        prediction = net(X_test_cuda, Z_test_cuda)
        
        # calculate nrmse loss on prediction:
        mse = MSE(prediction, true_smb)  # MSE loss
        rmse = torch.sqrt(mse)  # rmse
        nrmse = torch.sqrt(mse) / (
            torch.max(true_smb) - torch.min(true_smb)
        )
    
        pred_rmse += rmse.item()  # train rmse
        pred_nrmse_ += nrmse.item()  # train nrmse
        pred_mse += mse.item()  # train mse
        
        prediction = prediction.cpu().detach().numpy()  # send to device
        preds.append(prediction.transpose(0, 2, 3, 1)[0])  # change to numpy
        x.append(X_test.numpy().transpose(0, 2, 3, 1)[0])
        z.append(Z_test.numpy().transpose(0, 2, 3, 1)[0])
        y.append(Y_test.numpy().transpose(0, 2, 3, 1)[0])
        r.append(R_test.numpy()[0])
        
    loss_rmse = pred_rmse / len(test_loader)
    loss_nrmse = pred_nrmse_ / len(test_loader)
    loss_mse = pred_mse / len(test_loader)

    return preds, x, z, y, r, loss_mse, loss_nrmse, loss_rmse
                

    
    
def plotMultiplePredictions(preds, x, z, true_smb, r, 
                            UPRCM, 
                            VAR_LIST, 
                            target_dataset, 
                            points_RCM,
                            regions,
                            figsize=(15, 5), 
                            N = 10
                        ):
    fig = plt.figure(figsize=figsize)
    map_proj = ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)
    
    for i in range(N):
        randTime = rn.randint(0, len(preds)-1)
        sample2dtest_, sample_z, sampletarget_, samplepred_  = x[randTime], z[randTime], true_smb[randTime], preds[randTime]
        region = regions[r[randTime]] # region of sample
        dt = pd.to_datetime([UPRCM.time.isel(time=randTime).values])
        time = str(dt.date[0])
        
        sample2dtest_ = resize(sample2dtest_, 25, 48, print_=False)
        
        masktarget = np.expand_dims(createMask(sampletarget_, onechannel = True),2)
        
        dsGCM = createLowerInput(UPRCM, region='Larsen', Nx=35, Ny=25, print_=False)
        dsGCM = dsGCM.where(dsGCM.y > 0, drop=True)
        dsRCM = createLowerTarget(
                    target_dataset, region=region, Nx=64, Ny=64, print_=False
                )
        
        # apply mask to show only values on ice/land
        sampletarget_ = masktarget*sampletarget_
        sampletarget_[sampletarget_ == 0] = 'nan'
        
        samplepred_ = masktarget*samplepred_
        samplepred_[samplepred_ == 0] = 'nan'
        
        min_RCM = np.nanmin([sampletarget_, samplepred_])
        max_RCM = np.nanmax([sampletarget_, samplepred_])
        
        sampleGCM_ = dsGCM.SMB.isel(time = randTime).values
        min_GCM_Like = np.min(sampleGCM_)
        max_GCM_Like = np.max(sampleGCM_)
        
        vmin = np.nanmin([min_RCM, min_GCM_Like])
        vmax = np.nanmax([max_RCM, max_GCM_Like])
        
        M = 3
        for m in range(M):
            if m == 0:
                ax1 = plt.subplot(N, M,(i * M) + m + 1, projection=ccrs.SouthPolarStereo())
                dsGCM.SMB.isel(time = randTime).plot(x='x', ax = ax1, transform=ccrs.SouthPolarStereo(),
                                                                    add_colorbar=False,vmin = vmin, vmax = vmax,cmap="RdYlBu_r")
                ax1.coastlines("10m", color="black")
                #ax1.gridlines()
                ax1.set_title(f"{time}: GCM SMB")
            if m == 1:
                ax2 = plt.subplot(N, M, (i * M) + m + 1, projection=ccrs.SouthPolarStereo())
                im = plotTarget(target_dataset, sampletarget_, ax2, vmin, vmax, region=region)
            if m == 2:
                ax3 = plt.subplot(N, M, (i * M) + m + 1, projection=ccrs.SouthPolarStereo())
                plotPred(target_dataset, samplepred_, ax3, vmin, vmax, region=region)
        clb = fig.colorbar(im, ax=[ax1, ax2, ax3])
        clb.set_label('SMB [mmWe/day]')    
    
    
"""
plotMetrics: plots the time series of three points, the pearson correlation plot and (mean) target/predictions 
@input: 
- xr.Dataset target_dataset: target dataset
- xr.Dataset UPRCM: gcm like dataset
- PearsonCorr: pearson correlation at each pixel
- np.array true_smb_Larsen: true smb values
- np.array preds_Larsen: predictions of smb values by model
- torch dataset train_set: training set used to train model
- str region: region of interest (e.g. Larsen)
"""
        
def plotRidge(
    points,
    PearsonCorr,
    true_smb_Larsen,
    preds_Larsen,
    target_dataset,
    UPRCM,
    train_set,
    region: str,
    N: int=4,
    marker:str="x",
    figsize = (20,10),
    preds_Larsen_2 = None # if second model to compare
):
    f = plt.figure(figsize=figsize)
    M = int(N/2+1)
    ax1 = plt.subplot(M, 4, 1, projection=ccrs.SouthPolarStereo())
    meanPearson = np.nanmean(PearsonCorr)
    plotPearsonCorr(
        target_dataset,
        PearsonCorr,
        meanPearson,
        ax1,
        np.nanmin(PearsonCorr),
        np.nanmax(PearsonCorr),
        region=region,
    )
    ds = createLowerTarget(target_dataset, region=region, Nx=64, Ny=64, print_=False)
    #dsRCM = createLowerTarget(interp_dataset, region=region, Nx=64, Ny=64, print_=False)
    
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']
    randTime = rn.randint(0, len(true_smb_Larsen) - 1)
    dt = pd.to_datetime([UPRCM.time.isel(time=randTime).values])
    time = str(dt.date[0])
    meanTarget = np.nanmean(np.array(true_smb_Larsen), axis = 0)
    
    vmin = np.min([meanTarget, true_smb_Larsen[randTime], preds_Larsen[randTime]])
    vmax = np.max([meanTarget, true_smb_Larsen[randTime], preds_Larsen[randTime]])
    
    ax2 =  plt.subplot(M, 4, 2, projection=ccrs.SouthPolarStereo())
    plotTarget(target_dataset, meanTarget, ax2, vmin, vmax, region=region)
    ax2.set_title(f'Target: mean SMB, {region}')
    
    ax3 = plt.subplot(M, 4, 3, projection=ccrs.SouthPolarStereo())
    plotTarget(target_dataset, true_smb_Larsen[randTime], ax3, vmin, vmax, region=region)
    
    ax4 = plt.subplot(M, 4, 4, projection=ccrs.SouthPolarStereo())
    plotPred(target_dataset, preds_Larsen[randTime], ax4, vmin, vmax, region=region)
    
    
    axs = [ax1, ax2, ax3, ax4]
    for ax in axs:
        for p in points:
            ax.scatter(
                ds.isel(x=p["x"]).x.values,
                ds.isel(y=p["y"]).y.values,
                marker=marker,
                s=100,
                color="red",
            )
    plt.suptitle(time)
    
    # Plot timeseries
    p = points[0]
    randomPixel_pred = np.array(preds_Larsen)[:, p["y"], p["x"], 0]
    randomPixel_targ = np.array(true_smb_Larsen)[:, p["y"], p["x"], 0]
    
    df = pd.DataFrame(
        data={"pred": randomPixel_pred, "target": randomPixel_targ},
        index=target_dataset.time.values[len(train_set) :],
    )
    
    ax5 = plt.subplot(M, 4, (5, 6))
    ax5.plot(df["target"], label="RCM Truth", color="grey")
    ax5.plot(df["pred"], label="Emulator", color=colors[0],  linestyle="--")
    
    if preds_Larsen_2 != None:
        randomPixel_pred_2 = np.array(preds_Larsen_2)[:, p["y"], p["x"], 0]
        df['predictions MSE'] = randomPixel_pred_2
        ax5.plot(df["predictions MSE"], label="Emulator-MSE", color=colors[3], linestyle="-.")
        
    pearson = np.corrcoef(df["pred"], df["target"])[0, 1]
    rmse = mean_squared_error(y_pred = df["pred"], y_true = df["target"], squared = False)
    nrmse = rmse/(df["target"].max()- df["target"].min())
    ax5.set_title("Point:{}, pearson:{:.2f}, rmse:{:.2f}, nrmse:{:.2f}".format(p, pearson, rmse, nrmse))
    
    i = 7
    for p in points[1:]:
        randomPixel_pred = np.array(preds_Larsen)[:, p["y"], p["x"], 0]
        randomPixel_targ = np.array(true_smb_Larsen)[:, p["y"], p["x"], 0]
        
        df = pd.DataFrame(
            data={"pred": randomPixel_pred, "target": randomPixel_targ},
            index=target_dataset.time.values[len(train_set) :],
        )
        
        
        # ax = plt.subplot(2, 3, i, sharey=ax5)
        ax = plt.subplot(M, 4, (i, i+1))
        df["target"].plot(label="RCM Truth", color="grey", ax = ax)
        df["pred"].plot(label="Emulator", color=colors[0],  linestyle="--", ax = ax)
        
        if preds_Larsen_2 != None:
            randomPixel_pred_2 = np.array(preds_Larsen_2)[:, p["y"], p["x"], 0]
            df['predictions MSE'] = randomPixel_pred_2
            df["predictions MSE"].plot(label="Emulator-MSE", color=colors[3], linestyle="-.", ax = ax)
            
        pearson = np.corrcoef(df["pred"], df["target"])[0, 1]
        rmse = mean_squared_error(y_true = df["target"], y_pred = df["pred"], squared = False)
        nrmse = rmse/(df["target"].max()- df["target"].min())
        ax.set_title("Point:{}, pearson:{:.2f}, rmse:{:.2f}, nrmse:{:.2f}".format(p, pearson, rmse, nrmse))
        if i == 9:
            ax.legend(loc = 'upper right')
        i += 2
    plt.suptitle(f"Three time series at different coordinates {time}")
    plt.tight_layout()
    
    
def plotTimeseries(preds, true_smb, train_set, target_dataset, points_RCM, region, N, rollingMean = None, figsize=(15, 10)):
    dsRCM = createLowerTarget(
                target_dataset, region=region, Nx=64, Ny=64, print_=False
            )
    # Plot timeseries
    fig = plt.figure(figsize=figsize)
    p = points_RCM[0]
    randomPixel_pred = np.array(preds)[:, p["y"], p["x"], 0]
    randomPixel_targ = np.array(true_smb)[:, p["y"], p["x"], 0]
    df = pd.DataFrame(
        data={"pred": randomPixel_pred, "target": randomPixel_targ},
        index=target_dataset.time.values[len(train_set) :],
    )
    
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']
    
    M = int(N/2+1)
    i = 1
    evencol = 1
    for p in points_RCM:
        randomPixel_pred = np.array(preds)[:, p["y"], p["x"], 0]
        randomPixel_targ = np.array(true_smb)[:, p["y"], p["x"], 0]
        df = pd.DataFrame(
            data={"pred": randomPixel_pred, "target": randomPixel_targ},
            index=target_dataset.time.values[len(train_set) :],
        )
        # ax = plt.subplot(2, 3, i, sharey=ax5)
        ax = plt.subplot(M, 2, i)
        df["target"].plot(label="RCM Truth", color="black", alpha=0.6, ax = ax)
        df["pred"].plot(label="Emulator", color=colors[0],  linestyle="--", ax = ax)
        
        if rollingMean != None:
            df["target"].rolling(rollingMean).mean().plot(label="target-mean",  linestyle="--", color=colors[2], alpha=0.5, ax = ax)
            df["target"].rolling(rollingMean).mean().plot(label="pred-mean",  linestyle="--", color=colors[3], alpha=0.5, ax = ax)
            
        pearson = np.corrcoef(df["pred"], df["target"])[0, 1]
        rmse = mean_squared_error(y_true = df["target"], y_pred = df["pred"], squared = False)
        nrmse = rmse/(df["target"].max()- df["target"].min())
        ax.set_title("SMB for point:{}, pearson:{:.2f}, rmse:{:.2f}, nrmse:{:.2f}".format(p, pearson, rmse, nrmse))
        ax.grid(axis = 'y')
        if (evencol % 2) == 1:
            ax.set_ylabel('[mmWe/day]')
        if i == 5:
            ax.legend(loc = 'upper right')
        i += 1
        evencol+=1
    plt.tight_layout()
        
        
def plotTimeseries2Models(preds1, preds2, true_smb, train_set, target_dataset, points_RCM, region, N, cmap = 'viridis', figsize = (15, 10), labels = None):
    dsRCM = createLowerTarget(
                target_dataset, region=region, Nx=64, Ny=64, print_=False
            )
    # Plot timeseries
    fig = plt.figure(figsize=figsize)
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9'] # color blind friendly colors
    
    M = int(N/2+1)
    i, evencol= 1, 1
    metrics = []
    for p in points_RCM:
        randomPixel_pred1= np.array(preds1)[:, p["y"], p["x"], 0]
        randomPixel_pred2 = np.array(preds2)[:, p["y"], p["x"], 0]
        randomPixel_targ = np.array(true_smb)[:, p["y"], p["x"], 0]
        df = pd.DataFrame(
            data={"pred1": randomPixel_pred1, "pred2": randomPixel_pred2, "target": randomPixel_targ},
            index=target_dataset.time.values[len(train_set) :],
        )
        ax = plt.subplot(M, 2, i)
        df["target"].plot(label="RCM Truth", color="grey", alpha=0.8, ax = ax)
        
        if labels == None:
            labels = ['UPRCM', 'GCM']
        
        df["pred1"].plot(label=labels[0], color=colors[0], ax = ax, linewidth = 1.5)
        df["pred2"].plot(label=labels[1], color=colors[3], ax = ax, linestyle = '--', linewidth = 2)
        
        pearson = np.corrcoef(df["pred1"], df["target"])[0, 1]
        pearsonGCM = np.corrcoef(df["pred2"], df["target"])[0, 1]
        rmse = mean_squared_error(y_true = df["target"], y_pred = df["pred1"], squared = False)
        rmseGCM = mean_squared_error(y_true = df["target"], y_pred = df["pred2"], squared = False)
        nrmse = rmse/(df["target"].max()- df["target"].min())
        nrmseGCM = rmseGCM/(df["target"].max()- df["target"].min())
        ax.set_title("Point: P{} ({}, {})".format(i, p['x'], p['y']), fontsize = 18)
        
        if (evencol % 2) == 1:
            ax.set_ylabel('[mmWe/day]', fontsize = 18)
        if i == 4:
            ax.legend(loc = 'upper right')
            
        metrics.append({'point':p, 'rmse':rmse, 'nrmse':nrmse, 'pearson':pearson})
        
        # rotation of xlabels
        """
        textstrUPRCM = '\n'.join((
            r'$\mathrm{RMSE}_{U}=%.2f, \mathrm{RMSE}_{G}=%.2f$' % (rmse, rmseGCM, ),
            r'$\mathrm{nrmse}_{U}=%.2f, \mathrm{nrmse}_{G}=%.2f$' % (nrmse, nrmseGCM, ),
            r'$r_{UPRCM}=%.2f, r_{G}=%.2f$' % (pearson, pearsonGCM, )))"""
        textstrUPRCM = '\n'.join((
            r'$\mathrm{RMSE}_{U}=%.2f, r_{U}=%.2f$' % (rmse, pearson, ),
            r'$\mathrm{RMSE}_{G}=%.2f, r_{G}=%.2f$' % (rmseGCM, pearsonGCM, )))
        #props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        
        if i == 1:
            ax.set_ylim(top = 10)
            ax.text(0.02, 0.95, textstrUPRCM, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')
        if i == 2:
            ax.set_ylim(top = 10)
            ax.text(0.02, 0.95, textstrUPRCM, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')
        if i == 3:
            ax.set_ylim(top = 7)
            ax.text(0.02, 0.95, textstrUPRCM, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')
        if i == 4:
            ax.set_ylim(top = 3)
            ax.text(0.02, 0.95, textstrUPRCM, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')
            
        ax.grid(axis = 'y')
        
        i += 1
        evencol+=1
        
    plt.tight_layout()
    return metrics

        
from matplotlib import gridspec

def annualSMB(
    preds,
    true_smb,
    train_set,
    target_dataset,
    points_RCM,
    predsGCM=None,
    figsize=(15, 10),
    labels = None
):
    N = len(points_RCM)
    M = int(N / 2 + 1)
    fig = plt.figure(figsize=figsize)
    i = 1
    m = 1
    
    gs = gridspec.GridSpec(4, 2, width_ratios=[2.5, 1]) 
    
    metrics = []
    for p in points_RCM:
        randomPixel_pred = np.array(preds)[:, p["y"], p["x"], 0]
        randomPixel_targ = np.array(true_smb)[:, p["y"], p["x"], 0]
        randomPixel_pred_GCM = np.array(predsGCM)[:, p["y"], p["x"], 0]
        
        if labels == None:
            labels = ['UPRCM', 'GCM']
        
        df = pd.DataFrame(
            data={labels[0]: randomPixel_pred, "RCM Truth": randomPixel_targ, labels[1]: randomPixel_pred_GCM},
            index=target_dataset.time.values[len(train_set) :],
        )
        
        colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"]
        
        # yearly sum
        yearlySMB = df.resample("y").sum()
        yearlySMB.index = yearlySMB.index.strftime("%Y")
        
        # yearly mean
        #yearlySMBMean =yearlySMB.rolling(2).mean()
        #ax = plt.subplot(4, 2, i)
        ax = plt.subplot(gs[i-1])
        
        if i == 1 or i == 2:
            ax.set_ylim(bottom = -5, top = 30)
        if i == 3 or i == 4:
            ax.set_ylim([-40, 5])
        if i == 5 or i == 6:
            ax.set_ylim(bottom = -5, top = 40)
        if i == 7 or i == 8:
            ax.set_ylim(top = 10)
            
        yearlySMB.plot(
            kind="bar", ax=ax, color=["#0072B2", "grey", "#CC79A7"], alpha=0.8
        )
        
        ax.set_xticklabels(yearlySMB.index, rotation=45)
        if i == 5:
            ax.legend(loc="upper right")
        else:
            ax.get_legend().remove()
        pearson = np.corrcoef(yearlySMB[labels[0]], yearlySMB["RCM Truth"])[0, 1]
        rmse = mean_squared_error(
            y_true=yearlySMB["RCM Truth"], y_pred=yearlySMB[labels[0]], squared=False
        )
        nrmse = rmse / (yearlySMB["RCM Truth"].max() - yearlySMB["RCM Truth"].min())
        
        metrics.append({"point": p, "rmse": rmse, "nrmse": nrmse, "pearson": pearson})
        ax.set_title("Point: P{} ({}, {})".format(m, p["x"], p["y"]), fontsize = 18)
        ax.grid(axis="y")
        ax.set_ylabel('[mmWe]', fontsize = 18)
        
        pearson = np.corrcoef(df[labels[0]], df["RCM Truth"])[0, 1]
        pearsonGCM = np.corrcoef(df[labels[1]], df["RCM Truth"])[0, 1]
        rmse = mean_squared_error(y_true = df["RCM Truth"], y_pred = df[labels[0]], squared = False)
        rmseGCM = mean_squared_error(y_true = df["RCM Truth"], y_pred = df[labels[1]], squared = False)
        
        textstrUPRCM = '\n'.join((
            r'$\mathrm{RMSE}_{U}=%.2f, r_{U}=%.2f$' % (rmse, pearson, ),
            r'$\mathrm{RMSE}_{G}=%.2f, r_{G}=%.2f$' % (rmseGCM, pearsonGCM, )))
        
        #ax.text(0.02, 0.95, textstrUPRCM, transform=ax.transAxes, fontsize=14,verticalalignment='top')
        
        i += 1
        #ax = plt.subplot(4, 2, i)
        ax = plt.subplot(gs[i-1])
        """
        im = sns.violinplot(data = yearlySMB, palette = ["#0072B2", "grey", "#CC79A7"], 
                            split=True, inner="quartile")"""
        im = sns.boxplot(data = yearlySMB, palette = ["#0072B2", "grey", "#CC79A7"], 
                         boxprops=dict(alpha=.8), showmeans=True, 
                         meanprops={
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",})
        for violin in ax.collections[::2]:
          violin.set_alpha(0.8)
            
            
        # add text:
        medianGCM = np.median(yearlySMB[labels[1]])
        medianUPRCM = np.median(yearlySMB[labels[0]])
        medianRCM = np.median(yearlySMB['RCM Truth'])
        
        textstrBoxplots= '\n'.join((
            r'$\mathrm{\tilde{x}}_{U}=%.1f, \mathrm{\tilde{x}}_{R}=%.1f$' % (medianUPRCM, medianRCM, ),
            r'$\mathrm{\tilde{x}}_{G}=%.1f$' % (medianGCM, )))
        
        ax.text(0.02, 0.95, textstrBoxplots, transform=ax.transAxes, fontsize=14,
                verticalalignment='top')
        ax.grid(axis = 'y')
        
        if i == 1 or i == 2:
            ax.set_ylim(top = 30)
        if i == 3 or i == 4:
            ax.set_ylim([-40, 5])
        if i == 5 or i == 6:
            ax.set_ylim(top = 40)
        if i == 7 or i == 8:
            ax.set_ylim(top = 10)
        i+=1
        m+=1
        
    plt.tight_layout()
    return metrics



    
def CompareMetrics(
        PearsonCorr,
        Wasserstein,
        RMSE,
        NRMSE, 
        target_dataset,
        models,
        region: str,
        figsize=(20, 15),
        cmap = 'viridis'):
        fig = plt.figure(figsize=figsize)
            
        M, N = 3, 3
        # Correlation
        vmin, vmax = np.nanmin([PearsonCorr[0],PearsonCorr[1]]), np.nanmax([PearsonCorr[0],PearsonCorr[1]])
        ax = plt.subplot(M, N, 1, projection=ccrs.SouthPolarStereo())
        plotPearsonCorr(
                target_dataset, PearsonCorr[0], np.nanmean(PearsonCorr[0]), ax, vmin, vmax, region=region, cmap = cmap
        )
        ax.set_title("{}: Correlation".format(models[0]))
        textstrBoxplots= '\n'.join((
            r'$\mathrm{\mu}=%.2f$' % (np.nanmean(PearsonCorr[0]) ),))
    
        ax.text(0.02, 0.95, textstrBoxplots, transform=ax.transAxes, fontsize=16,
                verticalalignment='top')
        ax = plt.subplot(M, N, 2, projection=ccrs.SouthPolarStereo())
        plotPearsonCorr(
                target_dataset, PearsonCorr[1], np.nanmean(PearsonCorr[1]), ax, vmin, vmax, region=region, cmap = cmap
        )
        ax.set_title("{}: Correlation".format(models[1]))
        textstrBoxplots= '\n'.join((
            r'$\mathrm{\mu}=%.2f$' % (np.nanmean(PearsonCorr[1]) ),))
    
        ax.text(0.02, 0.95, textstrBoxplots, transform=ax.transAxes, fontsize=16,
                verticalalignment='top')
    
    
        # boxplot:
        ax = plt.subplot(M,N,3)
        
        corrdf = pd.DataFrame({models[0]:PearsonCorr[0].flatten(), models[1]: PearsonCorr[1].flatten()})
        im = sns.boxplot(data = corrdf, palette = ["#0072B2", "#CC79A7"], 
                                                    boxprops=dict(alpha=.8), showmeans=True, ax = ax,
                                                    meanprops={
                                                "markerfacecolor":"white", 
                                                "markeredgecolor":"black",})
        for violin in ax.collections[::2]:
                    violin.set_alpha(0.8)
        ax.set_title('Correlation')
    
          
        # Wasserstein:
        vmin, vmax = np.nanmin([Wasserstein[0],Wasserstein[1]]), np.nanmax([Wasserstein[0],Wasserstein[1]])
        ax = plt.subplot(M, N, 4, projection=ccrs.SouthPolarStereo())
        im = plotWasserstein(
                target_dataset, Wasserstein[0], np.nanmean(Wasserstein[0]), ax, vmin, vmax, region=region, cmap = cmap
        )
        ax = plt.subplot(M, N, 5, projection=ccrs.SouthPolarStereo())
        textstrBoxplots= '\n'.join((
            r'$\mathrm{\mu}=%.2f$' % (np.nanmean(Wasserstein[0]) ),))
        
        ax.text(0.02, 0.95, textstrBoxplots, transform=ax.transAxes, fontsize=16,
                verticalalignment='top')
    
        #formatter = LogFormatter(10, labelOnlyBase=False) 
        #clb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=formatter)
        #clb.set_label("Wasserstein distance")
        ax.set_title("{}: Wasserstein".format(models[0]))
        plotWasserstein(
                target_dataset, Wasserstein[1], np.nanmean(Wasserstein[1]), ax, vmin, vmax, region=region, cmap = cmap
        )
        ax.set_title("{}: Wasserstein".format(models[1]))
        textstrBoxplots= '\n'.join((
            r'$\mathrm{\mu}=%.2f$' % (np.nanmean(Wasserstein[1]) ),))
    
        ax.text(0.02, 0.95, textstrBoxplots, transform=ax.transAxes, fontsize=16,
                verticalalignment='top')
    
        # boxplot:
        ax = plt.subplot(M, N,6)
        corrdf = pd.DataFrame({models[0]:Wasserstein[0].flatten(), models[1]: Wasserstein[1].flatten()})
        im = sns.boxplot(data = corrdf, palette = ["#0072B2", "#CC79A7"], 
                                                    boxprops=dict(alpha=.8), showmeans=True, ax = ax,
                                                    meanprops={
                                                "markerfacecolor":"white", 
                                                "markeredgecolor":"black",})

        for violin in ax.collections[::2]:
                    violin.set_alpha(0.8)
        ax.set_title('Wasserstein distance')
        ax.set_yscale('log')
    
        # RMSE:
        vmin, vmax = np.nanmin([RMSE[0],RMSE[1]]), np.nanmax([RMSE[0],RMSE[1]])
        ax = plt.subplot(M, N,7, projection=ccrs.SouthPolarStereo())
        plotNRMSE(
                target_dataset, RMSE[0], np.nanmean(RMSE[0]), ax, vmin, vmax, region=region, cmap = cmap
        )
        textstrBoxplots= '\n'.join((
            r'$\mathrm{\mu}=%.2f$' % (np.nanmean(RMSE[0]) ),))
    
        ax.text(0.02, 0.95, textstrBoxplots, transform=ax.transAxes, fontsize=16,
                verticalalignment='top')
        ax.set_title("{}: RMSE".format(models[0]))
        ax = plt.subplot(M, N, 8, projection=ccrs.SouthPolarStereo())
        plotNRMSE(
                target_dataset, RMSE[1], np.nanmean(RMSE[1]), ax, vmin, vmax, region=region, cmap = cmap
        )
        ax.set_title("{}: RMSE".format(models[1]))
        textstrBoxplots= '\n'.join((
            r'$\mathrm{\mu}=%.2f$' % (np.nanmean(RMSE[1]) ),))
        
        ax.text(0.02, 0.95, textstrBoxplots, transform=ax.transAxes, fontsize=16,
                verticalalignment='top')
        # boxplot:
        ax = plt.subplot(M, N,9)
        corrdf = pd.DataFrame({models[0]:RMSE[0].flatten(), models[1]: RMSE[1].flatten()})
        im = sns.boxplot(data = corrdf, palette = ["#0072B2", "#CC79A7"], 
                                                    boxprops=dict(alpha=.8), showmeans=True, ax = ax,
                                                    meanprops={
                                                "markerfacecolor":"white", 
                                                "markeredgecolor":"black",})
        ax.set_title('RMSE')
        ax.set_yscale('log')
        for violin in ax.collections[::2]:
                    violin.set_alpha(0.8)  
                    
                    
    
            
