from training import *

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


def plotRandomPredictions(
    preds,
    x,
    z,
    true_smb,
    r,
    GCMLike,
    interp_dataset,
    VAR_LIST,
    target_dataset,
    regions,
    N=10,
):

    map_proj = ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)

    for i in range(N):
        randTime = rn.randint(0, len(preds) - 1)
        sample2dtest_, sample_z, sampletarget_, samplepred_ = (
            x[randTime],
            z[randTime],
            true_smb[randTime],
            preds[randTime],
        )
        region = regions[r[randTime]]  # region of sample
        dt = pd.to_datetime([GCMLike.time.isel(time=randTime).values])
        time = str(dt.date[0])

        if region != "Whole Antarctica":
            sample2dtest_ = resize(sample2dtest_, 25, 48, print_=False)
        else:
            sample2dtest_ = resize(sample2dtest_, 25, 90, print_=False)

        masktarget = np.expand_dims(createMask(sampletarget_, onechannel=True), 2)

        dsGCM = createLowerInput(GCMLike, region="Larsen", Nx=35, Ny=25, print_=False)
        dsGCM = dsGCM.where(dsGCM.y > 0, drop=True) # to have smaller plot same as Larsen region

        dsRCM = createLowerTarget(
            interp_dataset, region=region, Nx=64, Ny=64, print_=False
        )

        # apply mask to show only values on ice/land
        sampletarget_ = masktarget * sampletarget_
        sampletarget_[sampletarget_ == 0] = "nan"

        samplepred_ = masktarget * samplepred_
        samplepred_[samplepred_ == 0] = "nan"

        sampleinterp_ = dsRCM.SMB.isel(time=randTime).values
        sampleinterp_ = np.expand_dims(sampleinterp_, 2)
        sampleinterp_ = masktarget * sampleinterp_
        sampleinterp_[sampleinterp_ == 0] = "nan"

        min_RCM = np.nanmin([sampletarget_, samplepred_])
        max_RCM = np.nanmax([sampletarget_, samplepred_])

        sampleGCM_ = dsGCM.SMB.isel(time=randTime).values
        min_GCM_Like = np.min(sampleGCM_)
        max_GCM_Like = np.max(sampleGCM_)

        vmin = np.nanmin([min_RCM, np.nanmin(sampleinterp_)])
        vmax = np.nanmax([max_RCM, np.nanmax(sampleinterp_)])

        M = 4
        for m in range(M):
            if m == 0:
                ax = plt.subplot(
                    N, M, (i * M) + m + 1, projection=ccrs.SouthPolarStereo()
                )
                # plotTrain(GCMLike, sample2dtest_, 4, ax, time, VAR_LIST, region=region)
                if region == "Larsen":
                    dsGCM.SMB.isel(time=randTime).plot(
                        x="x",
                        ax=ax,
                        transform=ccrs.SouthPolarStereo(),
                        add_colorbar=True,
                        cmap="RdYlBu_r",
                    )
                    ax.coastlines("10m", color="black")
                    ax.gridlines()
                    ax.set_title(f"{time} GCM SMB")
            if m == 1:
                ax = plt.subplot(
                    N, M, (i * M) + m + 1, projection=ccrs.SouthPolarStereo()
                )
                plotInterp(target_dataset, sampleinterp_, ax, vmin, vmax, region=region)

            if m == 2:
                ax = plt.subplot(
                    N, M, (i * M) + m + 1, projection=ccrs.SouthPolarStereo()
                )
                im = plotTarget(
                    target_dataset, sampletarget_, ax, vmin, vmax, region=region
                )
            if m == 3:
                ax = plt.subplot(
                    N, M, (i * M) + m + 1, projection=ccrs.SouthPolarStereo()
                )
                plotPred(target_dataset, samplepred_, ax, vmin, vmax, region=region)
                