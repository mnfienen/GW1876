import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.size']=12
import matplotlib.pyplot as plt
import flopy
import pyemu

def redis_freyberg(fac=3,b_d="temp",nam_file="freyberg.nam",run=True):

   
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
                             steady=mf.dis.steady,perlen=mf.dis.perlen)

    flopy.modflow.ModflowBas(mfr,ibound=[resample_arr(a,fac) for a in mf.bas6.ibound.array],
                             strt=[resample_arr(a,fac) for a in mf.bas6.strt.array])

    flopy.modflow.ModflowNwt(mfr,headtol=1.0e-5,fluxtol=10,Continue=True)

    oc_spd = {(iper,0):["save head","save budget"] for iper in range(mfr.nper)}
    flopy.modflow.ModflowOc(mfr,stress_period_data=oc_spd)

    flopy.modflow.ModflowUpw(mfr,laytyp=mf.upw.laytyp,hk=[resample_arr(a,fac) for a in mf.upw.hk.array],
                             vka=[resample_arr(a,fac) for a in mf.upw.vka.array],
                             ss=[resample_arr(a,fac) for a in mf.upw.ss.array],
                             sy=[resample_arr(a,fac) for a in mf.upw.sy.array],ipakcb=mf.upw.ipakcb)
    
    rech_data = {}
    for kper in range(mfr.nper):
        rech_data[kper] = resample_arr(mf.rch.rech[kper].array,fac)
    flopy.modflow.ModflowRch(mfr,rech=rech_data)

    wel_spd0 = mf.wel.stress_period_data[0].copy()
    wel_spd0["i"] = (wel_spd0["i"] * fac) + int(fac/2.0)
    wel_spd0["j"] = (wel_spd0["j"] * fac) + int(fac/2.0)
    wel_spd1 = mf.wel.stress_period_data[1].copy()
    wel_spd1["i"] = (wel_spd1["i"] * fac) + int(fac/2.0)
    wel_spd1["j"] = (wel_spd1["j"] * fac) + int(fac/2.0)
    #print(mf.wel.stress_period_data[0]["i"],wel_spd["i"])
    #print(39 * fac + int(fac/2.0))
    wel_data = {}
    for kper in range(mfr.nper):
        wel_spd = mf.wel.stress_period_data[kper].copy()
        wel_spd["i"] = (wel_spd["i"] * fac) + int(fac/2.0)
        wel_spd["j"] = (wel_spd["j"] * fac) + int(fac/2.0)
        wel_data[kper] = wel_spd
    
    flopy.modflow.ModflowWel(mfr,stress_period_data=wel_data,ipakcb=mf.wel.ipakcb)

    #drn_spd = mf.drn.stress_period_data[0].copy()
    #drn_spd["i"] = (drn_spd["i"] * fac) + int(fac / 2.0)
    #drn_spd["j"] = (drn_spd["j"] * fac) + int(fac / 2.0)
    drn_spd = []
    print(mf.ghb.stress_period_data[0].dtype)
    drn_stage = mf.ghb.stress_period_data[0]["bhead"][0]
    drn_cond = mf.ghb.stress_period_data[0]["cond"][0]
    i = mfr.nrow - 1
    ib = mfr.bas6.ibound[0].array
    for j in range(mfr.ncol):
        if ib[i,j] == 0:
            continue
        drn_spd.append([0,i,j,drn_stage,drn_cond / fac])




    flopy.modflow.ModflowGhb(mfr,stress_period_data={0:drn_spd},ipakcb=mf.ghb.ipakcb)

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
    #mfr.run_model()
    if run:
        pyemu.os_utils.run("mfnwt {0}".format(nam_file),cwd=mfr.model_ws)



        # cbb = flopy.utils.CellBudgetFile(os.path.join(redis_model_ws, mfr.namefile.replace(".nam", ".cbc")), model=mfr)
        # print(cbb.textlist)

        # # reset top to be a amplified reflection on water table
        # hds = flopy.utils.HeadFile(os.path.join(redis_model_ws, mfr.namefile.replace(".nam", ".hds")), model=mfr)
        # mfr.dis.top = hds.get_data()[0,:,:] * 1.05

        # mfr.write_input()
        # #mfr.run_model()
        # pyemu.os_utils.run("mfnwt {0}".format(nam_file),cwd=mfr.model_ws)

        hds = flopy.utils.HeadFile(os.path.join(redis_model_ws, mfr.namefile.replace(".nam", ".hds")), model=mfr)
        #hds.plot(colorbar=True)
        #plt.show()

        lst = flopy.utils.MfListBudget(os.path.join(redis_model_ws, mfr.namefile.replace(".nam", ".list")))
        flx,vol = lst.get_dataframes(diff=True,start_datetime=mfr.start_datetime)
        flx.plot(subplots=True,figsize=(20,20))
        plt.savefig(os.path.join(redis_model_ws,"lst.pdf"))

        hds = flopy.utils.HeadFile(os.path.join(redis_model_ws, mfr.namefile.replace(".nam", ".hds")))
        top = mfr.dis.top.array
        ibound = mfr.bas6.ibound.array
        with PdfPages(os.path.join(redis_model_ws,"hds.pdf")) as pdf:
            for kper in range(0,mfr.nper,10): 
                print(kper) 
                data = hds.get_data(kstpkper=(0,kper))
                fig,axes = plt.subplots(2,3,figsize=(10,10))
            
                for k in range(mfr.nlay):
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
    return mfr

if __name__ == "__main__":
    redis_freyberg(5,b_d="temp_daily")

