import os
import shutil
import platform
import numpy as np
import matplotlib
font = {'size'   : 7}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,Button,RadioButtons
from matplotlib.patches import Rectangle as rect

RAW_MODEL_PATH = os.path.join("..","..","models","10par_xsec","raw_model_files")

HK_PATH = os.path.join("hk_Layer_1.ref")
if platform.system().lower() == "windows":
	EXE_PATH = "mf2005.exe"
else:
	EXE_PATH = "./mf2005"		
NAM_PATH = os.path.join("10par_xsec.nam")
HDS_PATH = os.path.join("10par_xsec.hds")
TRACK_POINTS = False
ITER = 0
NITER = 5

def run_model(k_array):
	np.savetxt(HK_PATH,k_array,fmt="%15.6E",delimiter='')
	os.system(EXE_PATH+" "+NAM_PATH)
	return np.loadtxt(HDS_PATH)


[shutil.copy2(os.path.join(RAW_MODEL_PATH,f),f) for f in os.listdir(RAW_MODEL_PATH)]
shutil.copy2(HK_PATH,HK_PATH+".init")
shutil.copy2(HDS_PATH,HDS_PATH+".init")

initial_hk = np.atleast_2d(np.loadtxt(HK_PATH+".init"))
initial_hds = np.loadtxt(HDS_PATH+".init")
#strt_path = os.path.join("model","ref_cal","strt_Layer_1.ref")
#if os.path.exists(strt_path):os.remove(strt_path)
#shutil.copy2(HDS_PATH,strt_path)
new_hds = initial_hds
np.savetxt(HK_PATH,initial_hk,fmt="%15.6E",delimiter='')

delx = 10
cell_nums = np.arange(delx,(initial_hk.shape[1]*delx)+delx,delx) - (0.5*delx)
fig = plt.figure(figsize=(8,8))
ax_cal = plt.axes((0.1,0.575,0.8,0.4))
ax_prd = plt.axes((0.1,0.15,0.8,0.4))

for i,(col) in enumerate(cell_nums):
    xmn,xmx = col-(delx*0.5),col+(delx*0.5)
    ymn,ymx = -1.0,0.0   
    if i == 0:
        c = 'm'
    elif i == cell_nums.shape[0]-1:
        c = 'g'
    else:
        c = '#E5E4E2'        
    a = 0.75
    r1 = rect((xmn,ymn),xmx-xmn,ymx-ymn,color=c,ec='k',alpha=a)
    ax_cal.add_patch(r1)
    r2 = rect((xmn,ymn),xmx-xmn,ymx-ymn,color=c,ec='k',alpha=a)
    ax_prd.add_patch(r2)    
    x,y = (xmn+xmx)/2.0,(ymn+ymx)/2.0
    ax_cal.text(x,y,i+1,ha='center',va='center',fontsize=12)
    ax_prd.text(x,y,i+1,ha='center',va='center',fontsize=12)
   
x = np.arange(0,100,delx)
ax_cal.scatter([cell_nums[3],cell_nums[5]],[2.1,2.5],marker='^',s=50,edgecolor="none",facecolor="r",label="observed")
cal, = ax_cal.plot(cell_nums,initial_hds[0,:],lw=1.5,color='b',marker='.',markersize=10,label="simulated")
prd, = ax_prd.plot(cell_nums,initial_hds[1,:],lw=1.5,color='b',marker='.',markersize=10)
#prd_sc = ax_prd.scatter([cell_nums[7]],[initial_hds[1,7]],marker='o',color='0.5',alpha=0.5,s=10)
ax_cal.plot(cell_nums,initial_hds[0,:],lw=1.5,color='0.5',ls="--",label="(initial)")
ax_prd.plot(cell_nums,initial_hds[1,:],lw=1.5,color='0.5',ls="--")

ax_cal.set_ylim(-1,8.5)
ax_prd.set_ylim(-1,8.5)
ax_cal.set_xticklabels([])
ax_prd.set_xticklabels([])
ax_cal.text(50,7.5,"Calibration",fontsize=15,ha="center")
ax_prd.text(50,7.5,"Forecast",fontsize=15,ha="center")
ax_cal.set_xlim(0,100)
ax_prd.set_xlim(0,100)
arrowprops=dict(connectionstyle="angle,angleA=0,angleB=90,rad=5",arrowstyle='->')
bbox_args = dict(fc="1.0")
ax_cal.annotate('Q=0.5 $m^3/d$',fontsize=12,xy=(95,0),xytext=(82.0,1.0),
             arrowprops=arrowprops,bbox=bbox_args)
ax_prd.annotate('Q=1.0 $m^3/d$',fontsize=12,xy=(95,0),xytext=(82.0,1.0),
             arrowprops=arrowprops,bbox=bbox_args)
ax_prd.text(0.0,-1.6,'Specified\nhead',ha='left',va='top',fontsize=12)
ax_prd.text(100,-1.6,'Specified\nflux',ha='right',va='top',fontsize=12)
ax_prd.text(50,-1.6,'Active model cells',ha='center',va='top',fontsize=12)
ax_prd.text(75,4.5,'?',ha='center',va='center',fontsize=50,alpha=0.25)


lower_95_k = np.zeros_like(initial_hk) + 0.5
lower_95_hds = run_model(lower_95_k)

upper_95_k = np.zeros_like(initial_hk) + 5.0
upper_95_hds = run_model(upper_95_k)

prior_range = [lower_95_hds[1,7],upper_95_hds[1,7]]
ax_prd.plot([cell_nums[7],cell_nums[7]],prior_range,lw=10,color="0.5",alpha=0.5,label="prior credible range")
#ax_prd.plot([cell_nums[7],cell_nums[7]],post_credible_range,lw=5,color="0.25",label="posterior credible range")

#ax_prd.fill_between(cell_nums,lower_95_hds[1,:],upper_95_hds[1,:],\
#	facecolor="0.5",edgecolor="none",alpha=0.25)
#ax_cal.fill_between(cell_nums,lower_95_hds[0,:],upper_95_hds[0,:],\
#	facecolor="0.5",edgecolor="none",alpha=0.25)
prior_rect = rect((0,0),0,0,facecolor="0.5",edgecolor="none",alpha=0.5)
post_rect = rect((0,0),0,0,facecolor="0.15",edgecolor="none",alpha=0.5)

handles,labels = ax_cal.get_legend_handles_labels()
handles.append(prior_rect)
handles.append(post_rect)
labels.append("Prior Credible Range")
labels.append("Posterior Credible Range")
ax_cal.legend(handles,labels,loc=2)

axcolor = '0.5'
ax_slider= plt.axes([0.1, 0.04, 0.8, 0.03], axisbg=axcolor)
s_K = Slider(ax_slider, "K", 0.5, 5.0, valinit=2.5)
def update(val):
	k = s_K.val
	new_hk = np.zeros_like(initial_hk) + k
	new_hds = run_model(new_hk)
	cal.set_ydata(new_hds[0,:])
	#ax_prd.plot(cell_nums,new_hds[1,:],lw=1.5,color='0.5',alpha=0.5)
	ax_prd.scatter([cell_nums[7]],[new_hds[1,7]],marker='s',color='0.05',alpha=0.5,s=30)
	prd.set_ydata(new_hds[1,:])		
	ax_prd.set_xlim(0,100)
	fig.canvas.draw_idle()
s_K.on_changed(update)

resetax = plt.axes([0.15, 0.4, 0.25, 0.04])
button = Button(resetax, 'Start Uncertainty Analysis', color='0.75', hovercolor='0.5')
def reset(event):
	plt.sca(ax_prd)
	plt.cla()	
	tracked = []
	new_hds = np.loadtxt(HDS_PATH)
	ax_prd.plot(cell_nums,initial_hds[1,:],lw=1.5,color='0.5',ls="--")
	ax_prd.text(50,7.5,"Forecast",fontsize=15,ha="center")
	prd, = ax_prd.plot(cell_nums,new_hds[1,:],lw=1.5,color='b',marker='.',markersize=10)		
	for i,(col) in enumerate(cell_nums):
	    xmn,xmx = col-(delx*0.5),col+(delx*0.5)
	    ymn,ymx = -1.0,0.0   
	    if i == 0:
	        c = 'm'
	    elif i == cell_nums.shape[0]-1:
	        c = 'g'
	    else:
	        c = '#E5E4E2'        
	    a = 0.75	    
	    r2 = rect((xmn,ymn),xmx-xmn,ymx-ymn,color=c,ec='k',alpha=a)
	    ax_prd.add_patch(r2)    
	    x,y = (xmn+xmx)/2.0,(ymn+ymx)/2.0	    
	    ax_prd.text(x,y,i+1,ha='center',va='center',fontsize=12)
	    #if i == 7:
	    #	r3 = rect((xmn,0.0),xmx-xmn,6-ymn,color='y',ec='k',alpha=0.5)
	    #	ax_prd.add_patch(r3)
	ax_prd.text(0.0,-1.6,'Specified\nhead',ha='left',va='top',fontsize=12)
	ax_prd.text(100,-1.6,'Specified\nflux',ha='right',va='top',fontsize=12)
	ax_prd.text(50,-1.6,'Active model cells',ha='center',va='top',fontsize=12)
	ax_prd.text(75,4.5,'?',ha='center',va='center',fontsize=50,alpha=0.25)
	ax_prd.annotate('Q=1.0 $m^3/d$',fontsize=12,xy=(95,0),xytext=(82.0,1.0),
         arrowprops=arrowprops,bbox=bbox_args)
	ax_prd.set_ylim(-1,8.5)
	ax_prd.set_xticklabels([])
	#ax_prd.set_yticklabels(['','0','1','2','3','4','5','6'])
	#ax_prd.fill_between(cell_nums,lower_95_hds[1,:],upper_95_hds[1,:],\
	#	facecolor="0.5",edgecolor="none",alpha=0.25)
	ax_prd.plot([cell_nums[7],cell_nums[7]],prior_range,lw=10,color="0.5",alpha=0.5,label="prior credible range")
button.on_clicked(reset)


plt.show()