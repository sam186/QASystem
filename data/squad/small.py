import matplotlib.pyplot as plt
import numpy as np

# import plotly.plotly as py
file = open('train.ids.context', 'r') 

JXs = []
for itere, line in enumerate(file):
	elements = np.array([int(i) for i in line.split()] )
	JXs.append(len(elements))
file.close()
JXs = np.array(JXs)
print('training set', len(JXs))

data_percent = 10
k = int(len(JXs)/100.0*data_percent)
idx = np.argpartition(JXs, k)[:k]

maxLen = (np.max(JXs[idx]))
print('max len', maxLen)
idx = np.append(idx, len(JXs)+1)
idx = np.sort(idx)
print('training set get ', len(idx), idx)

# file_names = ['train.ids.context', 'train.ids.question', 'train.question', 'train.span', 'train.answer', 'train.context']
# for file_name in file_names:
# 	file = open(file_name, 'r')
# 	file_w = open(file_name+'_'+str(data_percent), 'w') 
# 	idx_i = 0
# 	for itere, line in enumerate(file):
# 		if itere == idx[idx_i]:
# 			file_w.write(line)
# 			idx_i+=1
# 	file.close()
# 	file_w.close()
# 	print idx_i, ' written in ', file_name+'_'+str(data_percent)

maxLen = 100
# if getting val_data, with len of percent max train data len
file = open('val.ids.context', 'r') 
JXs = []
for itere, line in enumerate(file):
	elements = np.array([int(i) for i in line.split()] )
	JXs.append(len(elements))
file.close()
JXs = np.array(JXs)
idx = np.where(JXs<maxLen)[0] # with len of percent max train data len, not its percent
print(idx)
maxLen = np.max(JXs[idx])
print(np.max(JXs[idx]))
idx = np.append(idx, len(JXs)+1)
idx = np.sort(idx)
print('val set get ', len(idx), idx)

val_file_names = ['val.ids.context', 'val.ids.question', 'val.question', 'val.span', 'val.answer', 'val.context']      
for file_name in val_file_names:
	file = open(file_name, 'r')
	file_w = open(file_name+'_'+str(maxLen), 'w') 
	idx_i = 0
	for itere, line in enumerate(file):
		if itere == idx[idx_i]:
			file_w.write(line)
			idx_i+=1
	file.close()
	file_w.close()
	print idx_i, ' written in ', file_name+'_'+str(maxLen)
