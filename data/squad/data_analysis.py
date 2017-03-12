import matplotlib.pyplot as plt
import numpy as np

# import plotly.plotly as py
file = open('train.ids.context', 'r') 
file_span = open('train.span', 'r')
start_end = []
for line in file_span:
	start_end.append([int(i) for i in line.split()])
file_span.close()


s_num = []
s_max  =[]
itere = 0
for line in file:
	elements = np.array([int(i) for i in line.split()] )
	se = start_end[itere]	
	ans = elements[se[0]:se[1]+1]
	ans = ans[1:-1]

	# ind = np.where(elements==6)[0]
	# s_num.append(len(ind))
	# if len(ind)>1:
	# 	tmp=np.append(ind[1:]-ind[:-1], (ind[0]+1))
	# 	tmp = np.max(tmp)
	# 	s_max.append(tmp)
	# elif len(ind)==1:
	# 	s_max.append(ind[0]+1)
	# else:
	# 	# s_max.append(len(elements))
	# 	pass
	# if s_max[-1]>200:
	# 	print itere, elements
	# print se, ans, elements
	if np.sum(ans==6)>0:
		print itere+1, np.sum(ans==6), se, ans#, elements
		# assert 0==1
	# assert itere<10
	itere+=1
file.close()
print max(s_num), max(s_max), np.argmax(np.array(s_max))

plt.hist(s_max)
plt.show()
# fig = plt.gcf()

# plot_url = py.plot_mpl(fig, filename='sentence len')