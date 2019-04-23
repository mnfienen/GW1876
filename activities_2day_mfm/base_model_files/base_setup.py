def redis_freyberg(fac=3):

    model_ws = b_d
    mf_nam = "freyberg.nam"

    mf = flopy.modflow.Modflow.load(nam_file, model_ws=b_d, verbose=True, version="mfnwt", exe_name="mfnwt")

    def resample_arr(arr,fac):
        new_arr = np.zeros((arr.shape[0] * fac, arr.shape[1] * fac))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                new_arr[i*fac:(i*fac)+fac,j*fac:(j*fac)+fac] = arr[i,j]
        return new_arr


    assert fac % 2 != 0
    perlen = mf.dis.perlen
    delr = mf.dis.delr.array[0] / fac
    delc = mf.dis.delc.array[0] / fac
    redis_model_ws = "redis"
    mfr = flopy.modflow.Modflow("freyberg",model_ws=redis_model_ws,
                                version="mfnwt",exe_name="mfnwt")
    flopy.modflow.ModflowDis(mfr,nrow=mf.nrow*fac,ncol=mf.ncol*fac,nlay=mf.nlay,
                             nper=perlen.shape[0],delr=delr,delc=delc,
                             top=resample_arr(mf.dis.top.array,fac),
                             botm=[resample_arr(a,fac) for a in mf.dis.botm.array],
                             steady=False)

    flopy.modflow.ModflowBas(mfr,ibound=[resample_arr(a,fac) for a in mf.bas6.ibound.array],
                             strt=[resample_arr(a,fac) for a in mf.bas6.strt.array])

    flopy.modflow.ModflowNwt(mfr)

    oc_spd = {(iper,0):["save head","save budget"] for iper in range(mfr.nper)}
    flopy.modflow.ModflowOc(mfr,stress_period_data=oc_spd)

    flopy.modflow.ModflowUpw(mfr,laytyp=mf.upw.laytyp,hk=[resample_arr(a,fac) for a in mf.upw.hk.array],
                             vka=[resample_arr(a,fac) for a in mf.upw.vka.array],
                             ss=[resample_arr(a,fac) for a in mf.upw.ss.array],
                             sy=[resample_arr(a,fac) for a in mf.upw.sy.array])
    #mfr.upw.hk = 30.
    #mfr.upw.vka = 3
    #mfr.upw.hk[1] = 0.1
    #mfr.upw.vka[1] = 0.1
    rech0 = resample_arr(mf.rch.rech[0].array,fac)
    rech1 = resample_arr(mf.rch.rech[1].array,fac)
    flopy.modflow.ModflowRch(mfr,rech={0:rech0,1:rech1})

    wel_spd0 = mf.wel.stress_period_data[0].copy()
    wel_spd0["i"] = (wel_spd0["i"] * fac) + int(fac/2.0)
    wel_spd0["j"] = (wel_spd0["j"] * fac) + int(fac/2.0)
    wel_spd1 = mf.wel.stress_period_data[1].copy()
    wel_spd1["i"] = (wel_spd1["i"] * fac) + int(fac/2.0)
    wel_spd1["j"] = (wel_spd1["j"] * fac) + int(fac/2.0)
    #print(mf.wel.stress_period_data[0]["i"],wel_spd["i"])
    #print(39 * fac + int(fac/2.0))
    wel_dat = {0:wel_spd0,1:wel_spd1}
    flopy.modflow.ModflowWel(mfr,stress_period_data=wel_dat)

    #drn_spd = mf.drn.stress_period_data[0].copy()
    #drn_spd["i"] = (drn_spd["i"] * fac) + int(fac / 2.0)
    #drn_spd["j"] = (drn_spd["j"] * fac) + int(fac / 2.0)
    drn_spd = []
    print(mf.drn.stress_period_data[0].dtype)
    drn_stage = mf.drn.stress_period_data[0]["elev"][0]
    drn_cond = mf.drn.stress_period_data[0]["cond"][0]
    i = mfr.nrow - 1
    ib = mfr.bas6.ibound[0].array
    for j in range(mfr.ncol):
        if ib[i,j] == 0:
            continue
        drn_spd.append([0,i,j,drn_stage,drn_cond / fac])




    flopy.modflow.ModflowDrn(mfr,stress_period_data={0:drn_spd})

    rdata = pd.DataFrame.from_records(mf.sfr.reach_data)
    sdata = pd.DataFrame.from_records(mf.sfr.segment_data[0])
    print(rdata.reachID)

    rdata = rdata.reindex(np.arange(mfr.nrow))
    #print(rdata.strthick)
    #return
    rdata.loc[:,'k'] = 0
    rdata.loc[:,'j'] = (rdata.loc[0,"j"] * fac) + int(fac / 2.0)
    rdata.loc[:,'rchlen'] = mfr.dis.delc.array
    rdata.loc[:,'i'] = np.arange(mfr.nrow)
    rdata.loc[:,"iseg"] = rdata.i + 1
    rdata.loc[:,"ireach"] = 1
    rdata.loc[:,"reachID"] = rdata.index.values
    rdata.loc[:,"outreach"] = rdata.reachID + 1
    rdata.loc[mfr.nrow-1,"outreach"] = 0
    rdata.loc[:,"node"] = rdata.index.values
    for col in ["strthick","thts","thti","eps","uhc","strhc1"]:
        rdata.loc[:,col] = rdata.loc[0,col]



    upstrm = 34
    dwstrm = 33.5
    total_length = mfr.dis.delc.array.max() * mfr.nrow
    slope = (upstrm - dwstrm) / total_length
    # print(rdata.dtype,slope)
    strtop = np.linspace(upstrm, dwstrm, mfr.nrow)
    # print(strtop)
    rdata.loc[:,"strtop"] = strtop
    rdata.loc[:,"slope"] = slope

    #print(sdata.nseg)
    sdata = sdata.reindex(np.arange(mfr.nrow))
    for column in sdata.columns:
        sdata.loc[:,column] = sdata.loc[0,column]
    sdata.loc[:,"nseg"] = np.arange(mfr.nrow) + 1
    sdata.loc[1:,"flow"] = 0
    sdata.loc[:,"width1"] = 5.
    sdata.loc[:,"width2"] = 5.
    sdata.loc[:,"elevup"] = strtop
    sdata.loc[:,"elevdn"] = strtop - slope
    sdata.loc[:,"outseg"] = sdata.nseg + 1
    sdata.loc[mfr.nrow-1,"outseg"] = 0

    #print(sdata)
    print(mf.sfr.isfropt)

    flopy.modflow.ModflowSfr2(mfr,nstrm=mfr.nrow,nss=mfr.nrow,isfropt=mf.sfr.isfropt,
                              segment_data=sdata.to_records(index=False),
                              reach_data=rdata.to_records(index=False),ipakcb=mf.sfr.ipakcb,
                              istcb2=mf.sfr.istcb2,reachinput=True)
    #flopy.modflow.ModflowLmt(mfr,output_file_format="formatted",package_flows=["SFR"])

    mfr.write_input()
    mfr.run_model()

    cbb = flopy.utils.CellBudgetFile(os.path.join(redis_model_ws, mfr.namefile.replace(".nam", ".cbc")), model=mfr)
    print(cbb.textlist)


    hds = flopy.utils.HeadFile(os.path.join(redis_model_ws, mfr.namefile.replace(".nam", ".hds")), model=mfr)
    hds.plot(colorbar=True)
    plt.show()

    mlist = flopy.utils.MfListBudget(os.path.join(redis_model_ws, mfr.namefile.replace(".nam", ".list")))
    df = mlist.get_dataframes(diff=True)[1]
    df.plot()
    plt.show()


