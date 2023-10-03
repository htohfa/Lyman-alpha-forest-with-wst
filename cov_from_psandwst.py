import numpy as np
import os
data=[]
directory = '/rhome/htohf001/bigdata/stz3/'
files = os.listdir(directory)
for file in files:
	with open(os.path.join(directory,file), 'r') as f:
		array= np.loadtxt(f)
		data.append(array[:,1:19])

print(np.shape(data))

data=  np.concatenate(data, axis=0)
print(np.shape(data))
k= np.cov(np.transpose(data))
print(np.shape(k))
np.savetxt('new_st_first_sec_z3.txt',k)

data=[]
directory = '/rhome/htohf001/bigdata/stz4/'
files = os.listdir(directory)
for file in files:
        with open(os.path.join(directory,file), 'r') as f:
                array= np.loadtxt(f)
                data.append(array[:,1:19])

print(np.shape(data))

data=  np.concatenate(data, axis=0)
print(np.shape(data))
k= np.cov(np.transpose(data))
print(np.shape(k))
np.savetxt('new_st_first_sec_z4.txt',k)


data=[]
directory = '/rhome/htohf001/bigdata/psz4/'
files = os.listdir(directory)
for file in files:
        with open(os.path.join(directory,file), 'r') as f:
                array= np.loadtxt(f)
                data.append(array)

print(np.shape(data))

data=  np.concatenate(data, axis=0)
print(np.shape(data))
k= np.cov(np.transpose(data))
print(np.shape(k))
print("z=3")
print(k)


data=[]
directory = '/rhome/htohf001/bigdata/psz3/'
files = os.listdir(directory)
for file in files:
        with open(os.path.join(directory,file), 'r') as f:
                array= np.loadtxt(f)
                data.append(array)

print(np.shape(data))

data=  np.concatenate(data, axis=0)
print(np.shape(data))
k= np.cov(np.transpose(data))
print(np.shape(k))
print("z=3")
print(k)
