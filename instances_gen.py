import random as rand
import queue

def bfs(size, matrix):
    q = queue.LifoQueue() 
    vis = [0]*size
    start = 0
    vis[start] = 1
    q.put(start)
    while(not(q.empty())):
        u = q.get()
        for i in range(u+1, size):
            if matrix[u][i] > 0 and vis[i] == 0:
                q.put(i)
                vis[i] = 1
    left = []
    for i in range (size):
        if vis[i] == 0:
            left.append(i)
    return left



def matrix_gen(size, den):
    mat = []
    den-=3
    rand.seed()
    for i in range(size):
        mat.append([0]*size)
    for i in range(size):
        for j in range (i+1, size):
            a = rand.random() * 100
            if a <= den:
                mat[i][j] = 1
        
    tab = bfs(size, mat)
    while len(tab) > 0:
        for el in tab:
            a = rand.randint(0, el-1)
            mat[a][el] = 1
        tab = []
        tab = bfs(size, mat)
    for i in range(size):
        for j in range(i+1, size):
            mat[j][i] = mat[i][j]
    return mat

#main
#for i in range(5, 20):

v =30 #verticies
d = 50 #density
f = open("GC"+str(v)+"_"+str(d)+".txt", 'w')
mat = matrix_gen(v, d)
print(v)

f.write(str(v))
f.write("\n")
for i in range(0,v):
    for j in range(0, v):
        if mat[i][j] == 1 :
            f.write(str(i+1))
            f.write(" ")
            f.write(str(j+1))
            f.write("\n")
for line in mat:
    print(line)

f.close()