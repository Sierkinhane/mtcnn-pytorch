import numpy as np

"""
change (x1, y1, w, h) to (x1, y1, x2, y2)
"""

# original annotations file
with open('anno_train.txt') as f:
	contents = f.readlines()

coordinates = []
names = []
for content in contents:
	name = content.split(' ')[0]
	coordinate = np.array(content.split(' ')[1:-1], dtype=np.int32).reshape(-1, 4)
	names.append(name)
	coordinates.append(coordinate)

for coordinate in coordinates:
	coordinate[:, 2] = coordinate[:, 0] + coordinate[:, 2]
	coordinate[:, 3] = coordinate[:, 1] + coordinate[:, 3]

# modified annotations file
with open('anno_train_fixed.txt', 'w') as f:
	for n, c in zip(names, coordinates):
		a = str(list(c.reshape(1, -1)[0, :]))[1:-1].split(',')
		s = ''
		for i in a:
			s = s + i
		content = n + ' ' + s + '\n'

		f.write(content)
