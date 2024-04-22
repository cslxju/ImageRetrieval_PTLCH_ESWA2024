import matplotlib.pyplot as plt
import json
import os
import numpy as np
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False#将True改为False，作用就是解决负号'-'显示为方块的问题  
# Precision Recall Curve data
pr_data = {
    # "DPSH": "/home/rh/log/alexnet/DPSH_cifar10-1_48.json",#第一次修改12.11 16.46注释，第二次修改12.21 0.27跑DPSH召回率
    # "DSH": "/home/rh/DeepHash-pytorch-master/log/alexnet/DSH_cifar10-1_48.json",#"../log/alexnet/DSH_cifar10-1_48.json",12.11 16.44修改
    "DHNSOFTMAX": "/home/rh/DELETEHASH/PITCOTSOFTMAX/COCO_coco_0.5_32.json",#12.11 16.46注释，第二次修改12.21 0.27跑DHN召回率
    # "DPNSOFTMAX-1": "/home/rh/DELETEHASH/PITCOT/DPN_cifar10_48.json",#12.11 16.46注释
    # "DPNSOFTMAX-2": "/home/rh/DELETEHASH/PITCOTNORM/SINGLE_cifar10_0.1_48.json"#12.11 16.46注释
}
N = 1000
# N = -1
for key in pr_data:
    path = pr_data[key]
    pr_data[key] = json.load(open(path))


# markers = "DdsPvo*xH1234h"
markers = ".........................."
method2marker = {}
i = 0
for method in pr_data:
    method2marker[method] = markers[i]
    i += 1

plt.figure(figsize=(15, 5))
plt.subplot(131)

for method in pr_data:
    P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
    checkpoint = {                  
                    'P': P,
                    'R': R,
                    'P_top':P,
                   
                }
    if not os.path.isdir('resultjson/'+'/'+"coco"+'/'+"[DPNCOTSOFTMAX]"+'/'+str(32)):
               os.makedirs('resultjson//'+"coco"+'//'+"[DPNCOTSOFTMAX]"'//'+str(32))
    P = checkpoint['P']          
    R = checkpoint['R']          
    P_top = checkpoint['P_top']
    np.savetxt ('resultjson/'+'/'+"coco"+'/'+"[DPNCOTSOFTMAX]"+'/'+str(32)+'/'+'p.txt',P,fmt='%3.5f') 
    np.savetxt ('resultjson/'+'/'+"coco"+'/'+"[DPNCOTSOFTMAX]"+'/'+str(32)+'/'+'r.txt',R,fmt='%3.5f') 
    np.savetxt ('resultjson/'+'/'+"coco"+'/'+"[DPNCOTSOFTMAX]"+'/'+str(32)+'/'+'P_top.txt',P_top,fmt='%3.5f') 
    print(len(P))
    print(len(R))
    plt.plot(R, P, linestyle="-", marker=method2marker[method], label=method)
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()

plt.subplot(132)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, R, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('recall')
plt.legend()

plt.subplot(133)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, P, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('precision')
plt.legend()
# plt.savefig("pr.png")
# plt.show()
