import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

#bond = 2.25
bond = 8.0 
fname = 'n2.%f.out'%(bond)
out = open(fname, 'r').readlines()
dE, dT = [], []
for l in out:
    if l[:7] == "cycle =":
        dE.append(float(l.split()[-4]))
        dT.append(float(l.split()[-1]))

plt.plot(dE, dT, '-', color='#38686A')
plt.xlabel("dE(k)")
plt.ylabel("||dT(k)||")
plt.xlim((-0.014, 0.014))
#plt.ylim((0.0, 0.3))
plt.subplots_adjust(left=0.16, bottom=0.1, right=0.95, top=0.95)
plt.title('bond %f'%(bond))
plt.show()
#plt.savefig("conv.png", dpi=250)