import os
import shutil
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.size']=12
import flopy
import pyemu
import prep_deps

t_d = "template_daily"


def run_draws_and_pick_truth(run=True):
    num_workers = 15
    
    pst = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))

    pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(t_d,"prior.jcb"))
    #pe.loc[:,should_fix] = 1.0
    pe.to_csv(os.path.join(t_d,"sweep_in.csv"))
    pe.shape


    m_d = "master_truth_sweep"
    if run:
        pst.pestpp_options["overdue_giveup_fac"] = 1.5
        pst.write(os.path.join(t_d,"freyberg.pst"))
        pyemu.os_utils.start_workers(t_d,"pestpp-swp","freyberg.pst",num_workers=num_workers,worker_root=".",master_dir=m_d)

    obs_df = pd.read_csv(os.path.join(m_d,"sweep_out.csv"),index_col=0)
    print('number of realization in the ensemble before dropping: ' + str(obs_df.shape[0]))

    obs_df = obs_df.loc[obs_df.failed_flag==0,:]
    print('number of realization in the ensemble **after** dropping: ' + str(obs_df.shape[0]))

    fore_df = obs_df.loc[:,["part_time","part_status"]]
    for forecast in pst.forecast_names:
        if forecast in fore_df.columns:
            continue
        print(forecast)
        if forecast.startswith("fa"):
            #find all days in the month-year of the forecast
            fore_values = obs_df.loc[:,obs_df.columns.map(lambda x: forecast[:5] in x and forecast[7:12] in x)].mean(axis=1)
            fore_df.loc[fore_values.index,forecast] = fore_values
        else:
            fore_df.loc[:,forecast] = obs_df.loc[:,forecast]
    
    forecast = pst.forecast_names[0]
    #forecast = "part_time"
    print(forecast)
    sorted_vals = fore_df.loc[:,forecast].sort_values()


    idx = sorted_vals.index[25]
    print(fore_df.loc[idx,:])
    print(obs_df.loc[idx,pst.forecast_names])

    pst = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))
    obs = pst.observation_data
    pst.observation_data.loc[:,"weight"] = 0.0
    pst.observation_data.loc[:,"obsval"] = obs_df.loc[idx,pst.obs_names]
    pst.observation_data.loc[fore_df.columns,"obsval"] = fore_df.loc[idx,:].values
    pst.observation_data.loc[obs.obsnme.apply(lambda x: "trgw" in x),"weight"] = 5.0  # this corresponds to an (expected) noise standard deviation of 20 cm...
    pst.observation_data.loc[obs.obsnme.apply(lambda x: "fo_gage_1" in x),"weight"] = 0.01  # corresponding to an (expected) noise standard deviation of 100 m^3/d...
    
    b_d = os.path.join("..","base_model_files")
    assert os.path.exists(b_d) 
    obs.loc[pst.forecast_names,"obsval"].to_csv(os.path.join(b_d,"forecast_truth.csv"))

    np.random.seed(seed=0)
    nz_obs = obs.loc[pst.nnz_obs_names,:].copy()
    trgw_obs = nz_obs.loc[nz_obs.obsnme.apply(lambda x: "trgw" in x),:].copy()
    trgw_snd = np.random.randn(trgw_obs.shape[0])
    trgw_noise = trgw_snd * 1./trgw_obs.weight
    pst.observation_data.loc[trgw_obs.obsnme,"obsval"] += trgw_noise

    # setup flow obs noise
    np.random.seed(seed=1)
    fo_obs = nz_obs.loc[nz_obs.obsnme.apply(lambda x: "fo_" in x),:].copy()
    fo_snd = np.random.randn(fo_obs.shape[0])
    fo_noise = fo_snd * (fo_obs.obsval * 0.2)
    #fo_noise = fo_snd * 1./fo_obs.weight
    pst.observation_data.loc[fo_obs.obsnme,"obsval"] += fo_noise

    # Just for fun, lets have some "model error"
    obs = pst.observation_data
    offset_names = obs.loc[obs.obsnme.apply(lambda x: "trgw_015_016" in x),"obsnme"]
    pst.observation_data.loc[offset_names[:300],"obsval"] -= 3
    pst.observation_data.loc[offset_names[300:],"obsval"] += 3
    offset_names = obs.loc[obs.obsnme.apply(lambda x: "trgw_009_001" in x),"obsnme"]
    trend = np.linspace(0.0,3.0,offset_names.shape[0])
    #pst.observation_data.loc[offset_names[:100],"obsval"] -= 3
    #pst.observation_data.loc[offset_names[600:],"obsval"] -= 3
    pst.observation_data.loc[offset_names,"obsval"] += trend

    #add a trend to the flow obs
    trend = np.linspace(0.4,1.6,fo_obs.shape[0])
    pst.observation_data.loc[fo_obs.obsnme,"obsval"] *= trend

    #add some "spikes"
    spike_idxs = np.random.randint(0,fo_obs.shape[0],40)
    spike_names = fo_obs.obsnme.iloc[spike_idxs]
    pst.observation_data.loc[spike_names,"obsval"] *= 3.5

    nz_obs = pst.observation_data.loc[pst.nnz_obs_names,:].copy()
    nz_obs.loc[:,"datetime"] = pd.to_datetime(nz_obs.obsnme.apply(lambda x: x.split("_")[-1]))
    m = flopy.modflow.Modflow.load("freyberg.nam",model_ws=m_d,check=False)
    with PdfPages(os.path.join(m_d,"obs_v_sim_truth.pdf")) as pdf:

        for nz_group in pst.nnz_obs_groups:
            nz_obs_group = nz_obs.loc[nz_obs.obgnme==nz_group,:]
            fig,ax = plt.subplots(1,1,figsize=(10,2))
            ax.plot(nz_obs_group.datetime,nz_obs_group.obsval,"r-")
            ax.plot(nz_obs_group.datetime,obs_df.loc[idx,nz_obs_group.obsnme],"b-")
            #ax.plot(nz_obs_group.datetime, pst_base.observation_data.loc[nz_obs_group.obsnme,"obsval"],"g-")
            if (nz_group.startswith("trgw")):
                i = int(nz_group.split('_')[1])
                j = int(nz_group.split('_')[2])
                t = m.dis.top.array[i,j]
                ax.plot(ax.get_xlim(),[t,t],"k--")
                
            ax.set_title(nz_group)
            pdf.savefig()
            plt.close(fig)
    

    out_obs = obs.loc[obs.obsnme.apply(lambda x: x in pst.nnz_obs_names or x in pst.forecast_names),:]
    out_obs.loc[:,"site"] = out_obs.obsnme
    temporal_obs = out_obs.loc[out_obs.obsnme.apply(lambda x: x.startswith("trgw") or x.startswith("fo_")),"obsnme"]
    out_obs.loc[temporal_obs,"site"] = out_obs.loc[temporal_obs,"obsnme"].apply(lambda x: "_".join(x.split('_')[:-1]))
    out_obs.loc[temporal_obs,"datetime"] = pd.to_datetime(out_obs.loc[temporal_obs,"obsnme"].apply(lambda x: x.split("_")[-1]))
    out_obs.loc[:,"value"] = out_obs.obsval
    out_obs.loc[:,["site","datetime","value"]].to_csv(os.path.join(b_d,"obs_data.csv"),index=False)
    

    par_df = pd.read_csv(os.path.join(m_d,"sweep_in.csv"),index_col=0)
    pst.parameter_data.loc[:,"parval1"] = par_df.loc[idx,pst.par_names]
    pst.write(os.path.join(m_d,"test.pst"))
    pyemu.os_utils.run("pestpp-ies.exe test.pst",cwd=m_d)
    pst = pyemu.Pst(os.path.join(m_d,"test.pst"))
    print(pst.phi)

    m = flopy.modflow.Modflow.load("freyberg.nam",model_ws=m_d)
    lst = flopy.utils.MfListBudget(os.path.join(m_d,"freyberg.list"))
    df = lst.get_dataframes(diff=True,start_datetime=m.start_datetime)[0]
    ax = df.plot(figsize=(20,20), grid=False,subplots=True)
    #a = ax.set_xticklabels(["historic","scenario"],rotation=90)
    plt.savefig(os.path.join(m_d,"truth_water_bud.pdf"))
    plt.close("all")

    
    #pst_base = pyemu.Pst(os.path.join(t_d,"freyberg.pst"))
    

def build_daily_model():
    org_d = os.path.join("..","base_model_files")
    org_nam = "freyberg.nam"

    m_org = flopy.modflow.Modflow.load(org_nam,model_ws=org_d,check=False)

    tr_d = "temp_daily"
    tr_nam = "freyberg"
    m_tr = flopy.modflow.Modflow(tr_nam,model_ws=tr_d,version="mfnwt")
    
    tr_nper = 730 #num transient stress periods
    freq = "d"
    steady = [True]
    for _ in range(tr_nper):
        steady.append(False)
    start_datetime = "12-31-2015"
    end = pd.date_range(start=start_datetime,periods=tr_nper+1,freq=freq)
    delt = end[1:] - end[:-1]
    perlen = list(delt.days.values)
    perlen.insert(0,1)
    print(end)
    #print(perlen)
    model_start_datetime = "12-31-2015"
    assert len(perlen) == tr_nper + 1,len(perlen)
    botm = [m_org.dis.botm[0].array,m_org.dis.botm[1].array,np.loadtxt(os.path.join(org_d,"truth_botm_layer_3.ref"))]
    _ = flopy.modflow.ModflowDis(m_tr,nper=tr_nper+1,nlay=m_org.nlay,nrow=m_org.nrow,ncol=m_org.ncol,delr=m_org.dis.delr.array,
                            delc=m_org.dis.delc.array,top=m_org.dis.top.array,botm=botm,steady=steady,
                            perlen=perlen)
    m_tr.dis.start_datetime = model_start_datetime

    _ = flopy.modflow.ModflowBas(m_tr,ibound=m_org.bas6.ibound.array,strt=m_org.bas6.strt.array,hnoflo=m_org.bas6.hnoflo)

    
    _ = flopy.modflow.ModflowUpw(m_tr,ipakcb=50,laytyp=[1,0,0],hk=m_org.upw.hk.array,
                                 vka=m_org.upw.vka.array,ss=m_org.upw.ss.array,sy=m_org.upw.sy.array)

    _ = flopy.modflow.ModflowNwt(m_tr,headtol=0.01,fluxtol=1.0)
    _ = flopy.modflow.ModflowOc(m_tr,stress_period_data={(kper,0):["save head","save budget"] for kper in range(m_tr.nper)})

    angles = np.linspace(-np.pi, np.pi, tr_nper)
    season_mults = 1.0 + 0.65*np.sin(1 + angles*2)
    wel_season_mults = np.roll(season_mults,int(tr_nper / 4))
    
    org_wel_data = m_org.wel.stress_period_data[0]
    org_rch = m_org.rch.rech[0].array
    wel_data = {0:org_wel_data}
    rech = {0:org_rch}
    for kper in range(1,m_tr.nper):
        kper_wel_data = org_wel_data.copy()
        kper_wel_data["flux"] *= wel_season_mults[kper-1]
        wel_data[kper] = kper_wel_data
        rech[kper] = org_rch * season_mults[kper-1]


    _ = flopy.modflow.ModflowWel(m_tr,stress_period_data=wel_data,ipakcb=50)

    _ = flopy.modflow.ModflowRch(m_tr,rech=rech,ipakcb=50)

    _ = flopy.modflow.ModflowGhb(m_tr,stress_period_data=m_org.ghb.stress_period_data,ipakcb=50)

    m_org.sfr.reach_data

    rdata = pd.DataFrame.from_records(m_org.sfr.reach_data)
    sdata = pd.DataFrame.from_records(m_org.sfr.segment_data[0])
    
    rdata = rdata.reindex(np.arange(m_tr.nrow))
    upstrm = 34
    dwstrm = 33.5
    total_length = m_tr.dis.delc.array.max() * m_tr.nrow
    slope = (upstrm - dwstrm) / total_length
    # print(rdata.dtype,slope)
    strtop = np.linspace(upstrm, dwstrm, m_tr.nrow)
    # print(strtop)
    rdata.loc[:,"strtop"] = strtop
    rdata.loc[:,"slope"] = slope

    #print(sdata.nseg)
    sdata = sdata.reindex(np.arange(m_tr.nrow))
    for column in sdata.columns:
        sdata.loc[:,column] = sdata.loc[0,column]
    sdata.loc[:,"nseg"] = np.arange(m_tr.nrow) + 1
    sdata.loc[1:,"flow"] = 0
    sdata.loc[:,"width1"] = 5.
    sdata.loc[:,"width2"] = 5.
    sdata.loc[:,"elevup"] = strtop
    sdata.loc[:,"elevdn"] = strtop - slope
    sdata.loc[:,"outseg"] = sdata.nseg + 1
    sdata.loc[m_tr.nrow-1,"outseg"] = 0
    

    sdata_dict = {0:sdata.to_records(index=False)}
    for kper in range(1,m_tr.nper):
        kper_sdata = sdata.to_records(index=False)
        kper_sdata["flow"] *= season_mults[kper-1]
        sdata_dict[kper] = kper_sdata

    _ = flopy.modflow.ModflowSfr2(m_tr,nstrm=m_tr.nrow,nss=m_tr.nrow,isfropt=m_org.sfr.isfropt,
                              segment_data=sdata_dict,
                              reach_data=rdata.to_records(index=False),ipakcb=m_org.sfr.ipakcb,
                              istcb2=m_org.sfr.istcb2,reachinput=True)

    m_tr.external_path = "."
    m_tr.write_input()

    prep_deps.prep_template(tr_d)

    pyemu.os_utils.run("mfnwt {0}".format(tr_nam),cwd=tr_d)

    lst = flopy.utils.MfListBudget(os.path.join(tr_d,tr_nam+".list"))
    flx,vol = lst.get_dataframes(diff=True,start_datetime=m_tr.start_datetime)
    flx.plot(subplots=True,figsize=(20,20))
    plt.savefig(os.path.join(tr_d,"lst.pdf"))

    hds = flopy.utils.HeadFile(os.path.join(tr_d,tr_nam+".hds"))
    top = m_tr.dis.top.array
    ibound = m_tr.bas6.ibound.array
    with PdfPages(os.path.join(tr_d,"hds.pdf")) as pdf:
        for kper in range(0,m_tr.nper,10): 
            print(kper) 
            data = hds.get_data(kstpkper=(0,kper))
            fig,axes = plt.subplots(2,3,figsize=(10,10))
            
            for k in range(m_tr.nlay):
                arr = data[k,:,:].copy()
                dtw = top - arr
                arr[ibound[k,:,:]<=0] = np.NaN
                dtw[ibound[k,:,:]<=0] = np.NaN
                cb = axes[0,k].imshow(arr)
                plt.colorbar(cb,ax=axes[0,k])
                cb = axes[1,k].imshow(dtw)
                plt.colorbar(cb,ax=axes[1,k])
            pdf.savefig()#os.path.join(tr_d,"hds.pdf"))
            plt.close(fig)

    mp_files = [f for f in os.listdir(org_d) if "mp" in f or "location" in f]
    [shutil.copy2(os.path.join(org_d,f),os.path.join(tr_d)) for f in mp_files]

    for k in range(m_tr.nlay):
        np.savetxt(os.path.join(tr_d,"prsity_layer_{0}.ref".format(k+1)),np.zeros((m_tr.nrow,m_tr.ncol))+0.1,fmt="%15.6E")

    pyemu.os_utils.run("mp6 freyberg.mpsim",cwd=tr_d)


def setup_interface_daily():
    b_d = "temp_daily"
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file,model_ws=b_d,check=False,forgive=False)

    # assign the executable name for the model
    m.exe_name = "mfnwt"

    # now let's run this in a new folder called temp so we don't overwrite the original data
    m.change_model_ws("temp",reset_external=True)

    # this writes all the MODFLOW files in the new location 
    m.write_input()

    # the following helps get the dependecies (both python and executables) in the right place
    prep_deps.prep_template(t_d="temp")

    pyemu.os_utils.run("{0} {1}".format(m.exe_name,m.name+".nam"),cwd=m.model_ws)

    props = []
    paks = ["upw.hk","upw.vka","upw.ss","upw.sy","bas6.strt","extra.prsity"]  #"extra" because not a modflow parameter
    for k in range(m.nlay):
        props.extend([[p,k] for p in paks])
    const_props = props.copy()
    props.append(["rch.rech",None])

    for kper in range(m.nper):
        const_props.append(["rch.rech",kper])

    spatial_list_props = [["wel.flux",2],["ghb.cond",0],["ghb.cond",1],["ghb.cond",2]]  # spatially by each list entry, across all stress periods
    temporal_list_props = [["wel.flux",kper] for kper in range(m.nper)]  # spatially uniform for each stress period

    spatial_list_props, temporal_list_props

    dry_kper = int(m.nper * 0.85)
    hds_kperk = [[kper,k] for k in range(m.nlay) for kper in [0,dry_kper,m.nper-1]]

    hds_kperk

    sfr_obs_dict = {}
    sfr_obs_dict["hw"] = np.arange(1,int(m.nrow/2))
    sfr_obs_dict["tw"] = np.arange(int(m.nrow/2),m.nrow)
    sfr_obs_dict["gage_1"] = [39]

    pst_helper = pyemu.helpers.PstFromFlopyModel(nam_file,new_model_ws=t_d,org_model_ws="temp",
                                                 const_props=const_props,spatial_list_props=spatial_list_props,
                                                 temporal_list_props=temporal_list_props,remove_existing=True,
                                                 grid_props=props,pp_props=props,sfr_pars=["strk"],hds_kperk=hds_kperk,
                                                 sfr_obs=sfr_obs_dict,build_prior=False,model_exe_name="mfnwt",
                                                 pp_space=4)
    prep_deps.prep_template(t_d=pst_helper.new_model_ws)

    pst = pst_helper.pst

    # check out hydraulic conductivity parameters
    pst.parameter_data.loc[pst.parameter_data.parnme.apply(lambda x: "hk" in x),:]

    # what about observations? in particular, the sfr flow-out observations?
    pst.observation_data.loc[pst.observation_data.obgnme.apply(lambda x: "flout" in x),:]


    obs = pst.observation_data
    flout_obs = obs.loc[obs.obgnme.apply(lambda x: "flout" in x),"obsnme"]
    obs.loc[flout_obs,"obgnme"] = flout_obs.apply(lambda x: "_".join(x.split('_')[:-1]))


    obs_locs = pd.read_csv(os.path.join("..","base_model_files","obs_loc.csv"))
    #build obs names that correspond to the obsnme values in the control file
    obs_locs.loc[:,"site"] = obs_locs.apply(lambda x: "trgw_{0:03d}_{1:03d}".format(x.row-1,x.col-1),axis=1)
    kij_dict = {site:(2,r-1,c-1) for site,r,c in zip(obs_locs.site,obs_locs.row,obs_locs.col)}


    binary_file = os.path.join(pst_helper.m.model_ws,nam_file.replace(".nam",".hds"))
    frun_line,tr_hds_df = pyemu.gw_utils.setup_hds_timeseries(binary_file,kij_dict=kij_dict,include_path=True,model=pst_helper.m)
    pst_helper.frun_post_lines.append(frun_line)

    tr_hds_df.head()

    [f for f in os.listdir(pst_helper.m.model_ws) if f.endswith(".ins")]

    df = pst_helper.pst.add_observations(os.path.join(pst_helper.m.model_ws,
                    nam_file.replace(".nam",".hds_timeseries.processed.ins")),pst_path=".")
    obs = pst_helper.pst.observation_data
    obs.loc[df.index,"obgnme"] = df.index.map(lambda x: "_".join(x.split("_")[:-1]))
    obs.loc[df.index,"weight"] = 1.0

    mp_files = [f for f in os.listdir(b_d) if "mp" in f or "location" in f]
    [shutil.copy2(os.path.join(b_d,f),os.path.join(pst_helper.new_model_ws,f)) for f in mp_files]

    pst_helper.frun_post_lines.append("pyemu.os_utils.run('mp6 freyberg.mpsim >mp6.stdout')")
    pst_helper.tmp_files.append("freyberg.mpenpt")  # placed at top of `forward_run.py`
    pst_helper.write_forward_run()

    out_file = "freyberg.mpenpt"
    ins_file = out_file + ".ins"
    with open(os.path.join(pst_helper.new_model_ws,ins_file),'w') as f:
        f.write("pif ~\n")
        f.write("l7 w w w !part_status! w w !part_time!\n")

    df = pst_helper.pst.add_observations(os.path.join(pst_helper.new_model_ws,ins_file),
                                         os.path.join(pst_helper.new_model_ws,out_file),
                                         pst_path=".")
    for k in range(m.nlay):
        np.savetxt(os.path.join(pst_helper.new_model_ws,"arr_org","prsity_layer_{0}.ref".format(k+1)),
                   np.zeros((m.nrow,m.ncol))+0.001,fmt="%15.6E")

    par = pst.parameter_data  
    tag_dict = {"hk":[0.1,10.0],"vka":[0.1,10],"strt":[0.95,1.05],"pr":[0.8,1.2],"rech":[0.8,1.2]}
    for t,[l,u] in tag_dict.items():
        t_pars = par.loc[par.parnme.apply(lambda x: t in x ),"parnme"]
        par.loc[t_pars,"parubnd"] = u
        par.loc[t_pars,"parlbnd"] = l

    arr_csv = os.path.join(pst_helper.new_model_ws,"arr_pars.csv")
    df = pd.read_csv(arr_csv,index_col=0)

    sy_pr = df.model_file.apply(lambda x: "sy" in x or "pr" in x)
    df.loc[:,"upper_bound"] = np.NaN
    df.loc[sy_pr,"upper_bound"] = 0.4
    df.to_csv(arr_csv)

    pst.control_data.noptmax = 0
    pst.write(os.path.join(pst_helper.new_model_ws,"freyberg.pst"))
    pyemu.os_utils.run("pestpp-ies freyberg.pst",cwd=pst_helper.new_model_ws)

    pst = pyemu.Pst(os.path.join(pst_helper.m.model_ws,"freyberg.pst"))

    pe = pst_helper.draw(100)   
    pe.enforce()  # always a good idea!
    pe.to_binary(os.path.join(pst_helper.new_model_ws,"prior.jcb"))
    pst_helper.pst.write(os.path.join(pst_helper.m.model_ws,nam_file.replace(".nam",".pst")))

    obs = pst_helper.pst.observation_data
    dts = pd.to_datetime(pst_helper.m.start_datetime) + pd.to_timedelta(np.cumsum(pst_helper.m.dis.perlen.array),unit='d')
    dts_str = list(dts.map(lambda x: x.strftime("%Y%m%d")).values)
    dry_kper = int(pst_helper.m.nper * 0.85)
    dry_dt = dts_str[dry_kper]
    print(dry_dt)
    swgw_forecasts = obs.loc[obs.obsnme.apply(lambda x: "fa" in x and ("hw" in x or "tw" in x) and dry_dt in x),"obsnme"].tolist()
    hds_fore_name = "hds_00_{0:03d}_{1:03d}_{2:03d}".format(int(pst_helper.m.nrow/3),int(pst_helper.m.ncol/10)
                                                           ,dry_kper)
    print(hds_fore_name)
    hds_forecasts = obs.loc[obs.obsnme.apply(lambda x: hds_fore_name in x),"obsnme"].tolist()
    forecasts = swgw_forecasts
    forecasts.extend(hds_forecasts)
    forecasts.append("part_time")
    forecasts.append("part_status")
    pst_helper.pst.pestpp_options["forecasts"] = forecasts


    pst_helper.pst.write(os.path.join(pst_helper.m.model_ws,nam_file.replace(".nam",".pst")))
    lst = flopy.utils.MfListBudget(os.path.join(pst_helper.m.model_ws,"freyberg.list"))
    # df = lst.get_dataframes(diff=True,start_datetime=pst_helper.m.start_datetime)[0]
    # df.plot(kind="bar",figsize=(30,30), grid=True,subplots=True)
    # plt.show()


def revise_base_model():
    b_d = os.path.join("..","base_model_files")
    org_nam = "freyberg.nam"
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file,model_ws=b_d,check=False,forgive=False)

    vka = [.3,0.03,3]
    hk = [3,0.3,30.]
    _ = flopy.modflow.ModflowUpw(m,ipakcb=50,laytyp=[1,0,0],hk=hk,
                                 vka=vka,ss=m.upw.ss.array,sy=m.upw.sy.array)

    
    m.wel.stress_period_data[0]["flux"] *= 2.5
    m.rch.rech[0] *= 0.85

    rdata = pd.DataFrame.from_records(m.sfr.reach_data)
    sdata = pd.DataFrame.from_records(m.sfr.segment_data[0])
    
    rdata = rdata.reindex(np.arange(m.nrow))
    upstrm = 34
    dwstrm = 33.5
    total_length = m.dis.delc.array.max() * m.nrow
    slope = (upstrm - dwstrm) / total_length
    # print(rdata.dtype,slope)
    strtop = np.linspace(upstrm, dwstrm, m.nrow)
    # print(strtop)
    rdata.loc[:,"strtop"] = strtop
    rdata.loc[:,"slope"] = slope

    #print(sdata.nseg)
    sdata = sdata.reindex(np.arange(m.nrow))
    for column in sdata.columns:
        sdata.loc[:,column] = sdata.loc[0,column]
    sdata.loc[:,"nseg"] = np.arange(m.nrow) + 1
    sdata.loc[1:,"flow"] = 0
    sdata.loc[:,"width1"] = 5.
    sdata.loc[:,"width2"] = 5.
    sdata.loc[:,"elevup"] = strtop
    sdata.loc[:,"elevdn"] = strtop - slope
    sdata.loc[:,"outseg"] = sdata.nseg + 1
    sdata.loc[m.nrow-1,"outseg"] = 0
    sdata.loc[:,"hcond1"] = 1.0
    sdata.loc[:,"hcond2"] = 1.0


    _ = flopy.modflow.ModflowSfr2(m,nstrm=m.nrow,nss=m.nrow,isfropt=m.sfr.isfropt,
                              segment_data={0:sdata.to_records(index=False)},
                              reach_data=rdata.to_records(index=False),ipakcb=m.sfr.ipakcb,
                              istcb2=m.sfr.istcb2,reachinput=True)

    # m.external_path = "."
    m.change_model_ws("temp")
    m.write_input()

    prep_deps.prep_template("temp")

    pyemu.os_utils.run("mfnwt {0}".format(m.name+".nam"),cwd="temp")

    lst = flopy.utils.MfListBudget(os.path.join("temp",m.name+".list"))
    flx,vol = lst.get_dataframes(diff=True,start_datetime=m.start_datetime)
    flx.plot(subplots=True,figsize=(20,20))
    plt.savefig(os.path.join("temp","lst.pdf"))

    hds = flopy.utils.HeadFile(os.path.join("temp",m.name+".hds"))
    top = m.dis.top.array
    ibound = m.bas6.ibound.array
    with PdfPages(os.path.join("temp","hds.pdf")) as pdf:
        for kper in range(0,m.nper): 
            print(kper) 
            data = hds.get_data(kstpkper=(0,kper))
            fig,axes = plt.subplots(2,3,figsize=(10,10))
            
            for k in range(m.nlay):
                arr = data[k,:,:].copy()
                dtw = top - arr
                arr[ibound[k,:,:]<=0] = np.NaN
                dtw[ibound[k,:,:]<=0] = np.NaN
                cb = axes[0,k].imshow(arr)
                plt.colorbar(cb,ax=axes[0,k])
                cb = axes[1,k].imshow(dtw)
                plt.colorbar(cb,ax=axes[1,k])
            pdf.savefig()#os.path.join(tr_d,"hds.pdf"))
            plt.close(fig)

    mp_files = [f for f in os.listdir(b_d) if "mp" in f or "location" in f]
    [shutil.copy2(os.path.join(b_d,f),os.path.join("temp",f)) for f in mp_files]

    for k in range(m.nlay):
        np.savetxt(os.path.join("temp","prsity_layer_{0}.ref".format(k+1)),np.zeros((m.nrow,m.ncol))+0.1,fmt="%15.6E")

    pyemu.os_utils.run("mp6 freyberg.mpsim",cwd="temp")


if __name__ == "__main__":
    #revise_base_model()
    build_daily_model()
    #setup_interface_daily()
    #run_draws_and_pick_truth(run=True)
