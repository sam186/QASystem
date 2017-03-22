
import numpy as np
val_file_names = ['val.ids.context', 'val.ids.question', 'val.question', 'val.span', 'val.answer', 'val.context']    
file_names = ['train.ids.context', 'train.ids.question', 'train.question', 'train.span', 'train.answer', 'train.context']

file = open(file_names[0], 'r') 
JXs = []
for line in file:
	elements = np.array([int(i) for i in line.split()])
	JXs.append(len(elements))
file.close()
JXs = np.array(JXs)
print('set len', len(JXs))
JX_idx = np.argsort(JXs)

for file_name in file_names:
	print file_name
	file = open(file_name, 'r')
	x = []
	for line in file:
		x.append(line)
	file.close()
	assert len(x)==len(JX_idx)

	file_w = open(file_name+'_sorted', 'w') 
	for idx in JX_idx:
		file_w.write(x[idx])
	file_w.close()
