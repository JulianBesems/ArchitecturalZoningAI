n_target = 10 # run up to this value of n

cost_array = {} # initialize data for "array"
from math import sqrt
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

width_limit = int(2*sqrt(n_target)+5)
for w in range(0,width_limit+2):
    for cc in range(width_limit,0,-1):
        cost_array[w,cc]={}
MAX = n_target**3 # "infinity", trivial upper bound on cost
opt = (n_target+1)*[MAX] # initialize array for optimal values

cost_array[0,width_limit][0,0,0,0,0,0]=0 # starting "town" with no columns
for w in range(0,width_limit+1):
    for cc in range(width_limit,0,-1):
        for (D_up_right, D_down_right, D_up_left, D_down_left,
                n_up, n_down), cost in cost_array[w,cc].items():

            D_up_left += n_up # add 1 horizontal unit to all left-distances
            D_down_left += n_down

            for c in range(cc,-1,-1): # decrease size c of new column one by one
                n = n_up+n_down + (w+1)*c # (w = previous value of w)
                if n <= n_target: # total number of occupied points so far
                    new_cost = cost + ( (D_up_left + D_down_left) * c +
                                        (n_up + n_down) * c*(c-1)/2 +
                                        (c+1)*c*(c-1)/6 * (2*w+1) +
                                        c*c * w*(w+1)/2 )
                    if c==0: # a completed town
                        opt[n] = min( new_cost, opt[n] )
                    else: # store cost of newly constructed partial town
                        ind = (D_up_left, D_down_left, D_up_right, D_down_right,
                                n_up, n_down) # exchange left and right when storing
                        cost_array[w+1, c][ind] = min ( new_cost,
                                                        cost_array[w+1, c].get(ind, MAX) )
                    # decrease c by 1:
                    if (c%2)==1: # remove an element from the top of the leftmost column
                        n_up += w
                        D_up_left += n_up + w*(w+1)/2
                        D_up_right += n_up + w*(w-1)/2
                    else: # remove from the bottom
                        n_down += w
                        D_down_left += n_down + w*(w+1)/2
                        D_down_right += n_down + w*(w-1)/2

"""for i in range(n_target):
    row = ""
    for j in range(1, n_target):
        if cost_array[i,j]:
            row = row + 'X'
        else:
            row = row + ' '
    print(row)"""

"""x = []
y = []
for n in range(1,n_target):
    x.append(n)
    y.append(opt[n])
plt.plot(x,y)

xn = np.array(x)
yn = np.array(y)
fn = scipy.optimize.curve_fit((lambda t,a,b,c,d: a*(t**b) + c * (t**d)), xn, yn)

print(fn)
with open("lowTownValues.p", "wb") as fp:
            pickle.dump([x,y], fp, protocol = pickle.HIGHEST_PROTOCOL)

with open("functionTownValues.p", "wb") as fp:
            pickle.dump(fn, fp, protocol = pickle.HIGHEST_PROTOCOL)

fy = []
for t in x:
    a,b,c,d = fn[0]
    print(t, int(y[t-1]))
    fy.append(a*(t**b) + c * (t**d))

plt.plot(x,fy)
plt.show()"""
