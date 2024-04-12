import numpy as np
from itertools import product

def manual(conversion, relation):
    start = 2000000
    for i, j in zip(conversion[:-1], conversion[1:]):
        start = start * relation[i][j]     
    return start

relation = np.array([[1, 0.48, 1.52, 0.71],[2.05,1,3.26,1.56], [0.64, 0.3, 1, 0.46], [1.41, 0.61, 2.08, 1]], dtype=float)
start = 2000000
number_range = [0, 1, 2, 3]

middle_combinations = list(product(number_range, repeat=4))
final_combinations = np.array([[3] + list(middle) + [3] for middle in middle_combinations])
print(final_combinations.shape, final_combinations[:5])
max_pro = 0
for combination in final_combinations:
    pro = manual(combination, relation)
    if pro > max_pro:
        max_pro = pro
        max_combination = combination
print(max_pro, max_combination)