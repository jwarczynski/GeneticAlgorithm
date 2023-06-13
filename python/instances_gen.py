import random

MIN_VERTICES = 5
MAX_VERTICES = 6
STEP = 1
p = 0.5

# Model G(n,p)
def gnp(n,p):
    a = [[0 for j in range(n)] for i in range(n)]
    for i in range(1,n):
        for j in range(i):
            if random.random() <= p: 
                a[i][j] = a[j][i] = 1
    return a 


def save_to_file(matrix, v, path):
    with open(path, 'w') as f:
        f.write(f"{v}\n")
        for i in range(0,v):
            for j in range(0, v):
                if matrix[i][j] == 1 :
                    f.write(f"{i+1} {j+1}\n")
 
 
for v in range(MIN_VERTICES, MAX_VERTICES, STEP):
    mat = gnp(v, p)
    path = f"../test/{v}"
    save_to_file(mat, v, path)
    
