
import numpy as np
import matplotlib.pyplot as plt





data=np.loadtxt('/vol/work/mesa/bestneuronsnvscontext')
mat=np.zeros((250,510))
for el in data:
	row=int(el[0])
	col=int(el[1])
	mat[row,col]=el[2]


plt.imshow(mat,clim=(0.8040, 0.8352),cmap='binary')
plt.colorbar()
plt.grid()
plt.title('Context vs Number of Neurons over dev_accuracy')
plt.ylabel('context')
plt.xlabel('nneurons')
plt.ylim(0.0, 15.0)
plt.xlim(300.0,500.0)
plt.show()

