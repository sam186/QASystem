import matplotlib.pyplot as plt
import numpy as np

# import plotly.plotly as py
file = open('train.ids.context', 'r') 

JXs = []
for itere, line in enumerate(file):
	elements = np.array([int(i) for i in line.split()] )
	JXs.append(len(elements))
file.close()
JSx = np.array(JXs)
print len(JXs)

data_percent = 10
k = int(len(JXs)/100.0*data_percent)
idx = np.argpartition(JXs, k)[:k]

print(max([JXs[i] for i in idx]))
idx = np.append(idx, len(JXs)+1)
idx = np.sort(idx)
print len(idx), idx

file_names = ['train.ids.context', 'train.ids.question', 'train.question', 'train.span', 'train.answer', 'train.context']
for file_name in file_names:
	file = open(file_name, 'r')
	file_w = open(file_name+'_'+str(data_percent), 'w') 
	idx_i = 0
	for itere, line in enumerate(file):
		if itere == idx[idx_i]:
			file_w.write(line)
			idx_i+=1
	file.close()
	file_w.close()
	print idx_i, ' written in ', file_name+'_'+str(data_percent)



# file_span = open('train.span', 'r')
# start_end = []
# for line in file_span:
# 	start_end.append([int(i) for i in line.split()])
# file_span.close()