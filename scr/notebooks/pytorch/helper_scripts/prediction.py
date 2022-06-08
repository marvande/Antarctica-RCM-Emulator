from training import *
from metrics import *
from dataFunctions import *

# Set seed:
seed_all(SEED)


def predict(net, device, test_set, model, dir_checkpoint: str = Path("./checkpoints/")):
    logging.info(f"Loading saved model {model}")
    logging.info(f"Using device {device}")

    # test loader:

    loader_args = dict(
        batch_size=1, num_workers=4, pin_memory=True, worker_init_fn=seed_worker
    )
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # Load saved pt model
    net.to(device=device)
    net.load_state_dict(torch.load(str(model), map_location=device))
    logging.info("Saved model loaded!")

    preds, x, y, z, r = [], [], [], [], []

    for batch in tqdm(
        test_loader,
        total=len(test_loader),
        desc="Testing round",
        unit="batch",
        leave=False,
    ):
        seed_all(SEED)
        X_test, Z_test, Y_test, R_test = batch[0], batch[1], batch[2], batch[3]
        X_test_cuda = X_test.to(device=device, dtype=torch.float32)  # send to device
        Z_test_cuda = Z_test.to(device=device, dtype=torch.float32)  # send to device
        true_smb = Y_test.to(device=device, dtype=torch.float32)  # send to device

        net.eval()
        prediction = net(X_test_cuda, Z_test_cuda)
        prediction = prediction.cpu().detach().numpy()  # send to device
        preds.append(prediction.transpose(0, 2, 3, 1)[0])  # change to numpy
        x.append(X_test.numpy().transpose(0, 2, 3, 1)[0])
        z.append(Z_test.numpy().transpose(0, 2, 3, 1)[0])
        y.append(Y_test.numpy().transpose(0, 2, 3, 1)[0])
        r.append(R_test.numpy()[0])

    return preds, x, z, y, r
                

def plotRandomPrediction(preds, x, z, true_smb, r, 
                            GCMLike, 
                            VAR_LIST, 
                            target_dataset, 
                            points_RCM,
                            regions,
                            figsize=(15, 5), 
                            fontsize = 14,
                            cmap="RdYlBu_r"
                        ):
    fig = plt.figure(figsize=figsize)
    
    map_proj = ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)
    
    randTime = rn.randint(0, len(preds)-1)
    sample2dtest_, sample_z, sampletarget_, samplepred_  = x[randTime], z[randTime], true_smb[randTime], preds[randTime]
    region = regions[r[randTime]] # region of sample
    dt = pd.to_datetime([GCMLike.time.isel(time=randTime).values])
    time = str(dt.date[0].strftime('%m/%Y'))
    
    
    sample2dtest_ = resize(sample2dtest_, 25, 48, print_=False)
    
    masktarget = np.expand_dims(createMask(sampletarget_, onechannel = True),2)
    
    dsGCM = createLowerInput(GCMLike, region='Larsen', Nx=35, Ny=25, print_=False)
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
    
    M, i  = 3, 0
    for m in range(M):
        if m == 0:
            ax1 = plt.subplot(1, M, (i * M) + m + 1, projection=ccrs.SouthPolarStereo())
            if region == 'Larsen':
                dsGCM.SMB.isel(time = randTime).plot(x='x', ax = ax1, 
                    transform=ccrs.SouthPolarStereo(),
                    add_colorbar=False,vmin = vmin, 
                    vmax = vmax,cmap=cmap)
                ax1.coastlines("10m", color="black", linewidth = 1)
                ax1.gridlines(color = 'grey')
                ax1.set_title(f"GCM: SMB")
        if m == 1:
            ax3 = plt.subplot(1, M, m + 1, projection=ccrs.SouthPolarStereo())
            im = plotTarget(target_dataset, sampletarget_, ax3, vmin, vmax, region=region, cmap = cmap)
        if m == 2:
            ax4 = plt.subplot(1, M, m + 1, projection=ccrs.SouthPolarStereo())
            im2 = plotPred(target_dataset, samplepred_, ax4, vmin, vmax, region=region, cmap = cmap)
            
    for ax in [ax3]:
        for p in points_RCM:
            ax.scatter(
                    dsRCM.isel(x=p["x"]).x.values,
                    dsRCM.isel(y=p["y"]).y.values,
                    marker="x",
                    s=100,
                    color="red",
                )
    plt.suptitle(f'Random month: {time}')
    #clb = fig.colorbar(im, ax=[ax1, ax3, ax4], location='bottom')
    #clb.ax.set_title('SMB [mmWe/day]', fontsize = 14)
    clb = fig.colorbar(im, ax=[ax1, ax3, ax4], fraction=0.046, pad=0.04)
    clb.set_label('SMB [mmWe/day]')  
    
    return time
    
    
    
def plotMultiplePredictions(preds, x, z, true_smb, r, 
                            GCMLike, 
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
        dt = pd.to_datetime([GCMLike.time.isel(time=randTime).values])
        time = str(dt.date[0])
        
        sample2dtest_ = resize(sample2dtest_, 25, 48, print_=False)
        
        masktarget = np.expand_dims(createMask(sampletarget_, onechannel = True),2)
        
        dsGCM = createLowerInput(GCMLike, region='Larsen', Nx=35, Ny=25, print_=False)
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
                ax1.gridlines()
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
- xr.Dataset GCMLike: gcm like dataset
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
    GCMLike,
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
    dt = pd.to_datetime([GCMLike.time.isel(time=randTime).values])
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
        
        
def plotTimeseries2Models(preds1, preds2, true_smb, train_set, target_dataset, points_RCM, region, N, cmap = 'viridis', figsize = (15, 10)):
    dsRCM = createLowerTarget(
                target_dataset, region=region, Nx=64, Ny=64, print_=False
            )
    # Plot timeseries
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    p = points_RCM[0]
    randomPixel_pred1= np.array(preds1)[:, p["y"], p["x"], 0]
    randomPixel_pred2 = np.array(preds2)[:, p["y"], p["x"], 0]
    randomPixel_targ = np.array(true_smb)[:, p["y"], p["x"], 0]
    df = pd.DataFrame(
        data={"pred1": randomPixel_pred1, "pred2": randomPixel_pred2, "target": randomPixel_targ},
        index=target_dataset.time.values[len(train_set) :],
    )
    
    #colors = plt.cm.cividis(np.linspace(0, 1, 10))
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9'] # color blind friendly colors
    
    M = int(N/2+1)
    ax5 = plt.subplot(M, 4, (1, 2))
    ax5.plot(df["target"], label="target", color="grey", alpha=0.8)
    ax5.plot(df["pred1"], label="RCM", color=colors[0])
    ax5.plot(df["pred2"], label="GCM", color=colors[3], linestyle = '--')
    pearson = np.corrcoef(df["pred1"], df["target"])[0, 1]
    pearsonGCM = np.corrcoef(df["pred2"], df["target"])[0, 1]
    rmse =  mean_squared_error(y_true = df["target"], y_pred = df["pred1"], squared = False)
    nrmse = rmse/(df["target"].max()- df["target"].min())
    ax5.set_title("Point:{}, pearson:{:.2f}, pearson GCM: {:.2f}, rmse:{:.2f}, nrmse:{:.2f}".format(p, pearson, pearsonGCM, rmse, nrmse))
    
    i = 3
    for p in points_RCM[1:]:
        randomPixel_pred1= np.array(preds1)[:, p["y"], p["x"], 0]
        randomPixel_pred2 = np.array(preds2)[:, p["y"], p["x"], 0]
        randomPixel_targ = np.array(true_smb)[:, p["y"], p["x"], 0]
        df = pd.DataFrame(
            data={"pred1": randomPixel_pred1, "pred2": randomPixel_pred2, "target": randomPixel_targ},
            index=target_dataset.time.values[len(train_set) :],
        )
        # ax = plt.subplot(2, 3, i, sharey=ax5)
        ax = plt.subplot(M, 4, (i, i+1))
        df["target"].plot(label="RCM Truth", color="grey", alpha=0.8, ax = ax)
        df["pred1"].plot(label="RCM", color=colors[0], ax = ax)
        df["pred2"].plot(label="GCM", color=colors[3], ax = ax, linestyle = '--')
        
        pearson = np.corrcoef(df["pred1"], df["target"])[0, 1]
        pearsonGCM = np.corrcoef(df["pred2"], df["target"])[0, 1]
        rmse = mean_squared_error(y_true = df["target"], y_pred = df["pred1"], squared = False)
        nrmse = rmse/(df["target"].max()- df["target"].min())
        ax.set_title("Point:{}, pearson:{:.2f}, pearson GCM: {:.2f}, rmse:{:.2f}, nrmse:{:.2f}".format(p, pearson, pearsonGCM, rmse, nrmse))
        if i == 5:
            ax.legend(loc = 'upper right')
        plt.tight_layout()
        i += 2
        
        
def annualSMB(preds, true_smb, train_set, target_dataset, points_RCM, predsGCM = None):
    N = len(points_RCM)
    M = int(N/2+1)
    fig = plt.figure(figsize=(15, 10))
    i = 1
    for p in points_RCM:
        randomPixel_pred = np.array(preds)[:, p["y"], p["x"], 0]
        randomPixel_targ = np.array(true_smb)[:, p["y"], p["x"], 0]
        
        
        df = pd.DataFrame(
                        data={"pred": randomPixel_pred, "target": randomPixel_targ},
                        index=target_dataset.time.values[len(train_set) :],
                ) 
        if predsGCM != None:
            randomPixel_pred_GCM = np.array(predsGCM)[:, p["y"], p["x"], 0]
            df['pred-GCM'] = randomPixel_pred_GCM
            
        colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']
        yearlySMB = df.resample('y').sum()
        yearlySMB.index = yearlySMB.index.strftime('%Y')
        ax = plt.subplot(M,2,i)
        yearlySMB.plot(kind = 'bar', ax = ax, color = ['#0072B2', 'grey', '#CC79A7'], alpha = 0.8)
        if i < 5:
            ax.set_xticklabels('')
        else:
            ax.set_xticklabels(yearlySMB.index, rotation = 45)
        if i == 4:
            ax.legend(fontsize = 12, loc = 'upper left')
        else:
            ax.get_legend().remove()
        pearson = np.corrcoef(yearlySMB["pred"], yearlySMB["target"])[0, 1]
        rmse = mean_squared_error(y_true = yearlySMB["target"], y_pred = yearlySMB["pred"], squared = False)
        nrmse = rmse/(yearlySMB["target"].max()- yearlySMB["target"].min())
        ax.set_title("Point:({}, {}), pearson:{:.2f}, rmse:{:.2f}, nrmse:{:.2f}".format(p['x'], p['y'],pearson, rmse, nrmse))
        i+=1