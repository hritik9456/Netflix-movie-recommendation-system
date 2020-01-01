import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
K=3
seed = 0
#for i in range(1,5):
mixture, post=common.init(X,4,3)
mix,pos,cost = naive_em.run(X,mixture,post)
mix1,pos1,cost1 = kmeans.run(X,mixture,post)
mix2,pos2,cost2 = em.run(X,mixture,post)
for j in range(15):
      mix,pos,cost = naive_em.run(X,mix,pos)
      mix1,pos1,cost1 = kmeans.run(X,mix1,pos1)
      mix2,pos2,cost2 = em.run(X,mix2,pos2)
      if j==4 or j==7:
         print(f"naive_em:    {mix},{cost} ,\n kmeans:    {mix1},{cost1}  ,\n em:    {mix2},{cost2}" )

common.plot(X,mix,pos,'Cost')
common.plot(X,mix1,pos1,'Cost1')
common.plot(X,mix2,pos2,'Cost2')


