import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib
from utils import *

def print_csi(exp, t, n_classes, dataset):
	for p in range(n_classes):
		plt.figure()
		plt.title('Location $p_{%d}$ (real)'%(p+1) , fontsize=20)
		plt.xlabel('CSI Index', fontsize=18)
		plt.ylabel('CSI Amplitude (Normalized)', fontsize=18)
		plt.axis([0, 120, 0, 1])
		plt.grid(True)

		for i in dataset[p]:
	
			plt.plot(i)
		font = {'family' : 'Verdana','weight' : 'normal','size'   : 12}	
		matplotlib.rc('font', **font)
		plt.show()
		#plt.savefig('visualizations/CSI/%s-p%d.eps'%(exp, p+1), dpi=1)  
		#plt.savefig('visualizations/CSI/%s-p%d.png'%(exp, p+1), dpi=100)		
		#plt.close() 

dataset1 = data_preproc(np.asarray(pickle.load(open('dataset/EXP1.pickle','rb'))), scale_range =(0,1))
X_tra1, _, _, _ = dataset1
X_tra1 = X_tra1.reshape(16,-1,120)


dataset2 = data_preproc(np.asarray(pickle.load(open('dataset/EXP2.pickle','rb'))), scale_range =(0,1))
X_tra2, _, _, _ = dataset2
X_tra2 = X_tra2.reshape(14,-1,120)


X_tra3_r1, _, _, _  = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r1.pickle','rb'))), scale_range =(0,1))
X_tra3_r2, _, _, _   = data_preproc(np.asarray(pickle.load(open('dataset/EXP3-r2.pickle','rb'))), scale_range =(0,1))
X_tra3_r1 = X_tra3_r1.reshape(18,-1,120)
X_tra3_r2 = X_tra3_r2.reshape(18,-1,120)
#print_csi('exp1', 'Comference Room', 16, X_tra1)
#print_csi('exp2', 'Lounge', 14, X_tra2)
print_csi('exp3-r1',' Tx1-Rx1', 18, X_tra3_r1)
print_csi('exp3-r2',' Tx2-Rx2', 18, X_tra3_r2)
