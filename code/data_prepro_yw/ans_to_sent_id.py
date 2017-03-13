import numpy as np

# import plotly.plotly as py
org_file_name = ['train.ids.context', 'train.span']
data_percent = 5

file_span = open(org_file_name[1]+'_'+str(data_percent), 'r')
start_end = []
for line in file_span:
	start_end.append([int(i) for i in line.split()])
file_span.close()

sent_id = []

file = open(org_file_name[0]+'_'+str(data_percent), 'r') 
itere = 0
not_lovable = []
for itere, line in enumerate(file):
	elements = np.array([int(i) for i in line.split()] )

	se = start_end[itere]	
	
file.close()


file = open('train.answer', 'r')
not_lovable.append(100000)
li = 0
for i,line in enumerate(file):
	if i==not_lovable[li]:
		li+=1
		print line
file.close()


s_num = []
s_max  =[]
for line in file:
	elements = np.array([int(i) for i in line.split()] )
	ind = np.where(elements==6)[0]
	s_num.append(len(ind))
	if len(ind)>1:
		tmp=np.append(ind[1:]-ind[:-1], (ind[0]+1))
		tmp = np.max(tmp)
		s_max.append(tmp)
	elif len(ind)==1:
		s_max.append(ind[0]+1)
	else:
		s_max.append(len(elements))
	# itere+=1
file.close()
print max(s_max), max(s_num)
start_end = np.array(start_end)
print np.max(start_end[1]-start_end[0])