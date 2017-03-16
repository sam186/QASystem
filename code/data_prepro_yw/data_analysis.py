import matplotlib.pyplot as plt
import numpy as np

# import plotly.plotly as py
file = open('train.ids.context', 'r') 
file_span = open('train.span', 'r')
start_end = []
for line in file_span:
	start_end.append([int(i) for i in line.split()])
file_span.close()


# itere = 0
# not_lovable = []
# for line in file:
# 	elements = np.array([int(i) for i in line.split()] )
# 	se = start_end[itere]	
# 	ans = elements[se[0]:se[1]+1]
# 	while len(ans)>0 and ans[0] in [6, 11]:
# 		ans = ans[1:]
# 	while len(ans)>0 and ans[-1] in [6, 11]:
# 		ans = ans[:-1]
# 	if np.sum(ans==6)>0:
# 		# print itere+1, np.sum(ans==6), se, elements[se[0]:se[1]+1]#, elements 
# 		not_lovable.append(itere)
# 	itere+=1
# file.close()
# print itere
# print len(not_lovable)

# file = open('train.answer', 'r')
# not_lovable.append(100000)
# li = 0
# for i,line in enumerate(file):
# 	if i==not_lovable[li]:
# 		li+=1
# 		print line
# file.close()


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