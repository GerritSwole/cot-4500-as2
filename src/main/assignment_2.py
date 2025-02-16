import numpy as np
import pandas as pd

print("\nQuestion 1, Neville Method\n")

x = [3.6,3.8,3.9]
Fx = [1.675,1.436,1.318]

def Neville(x,Fx):
    x = x
    Fx = Fx
    target = 3.7

    n = len(x)
    Q = np.zeros((n, n))
    Q[:, 0] = Fx

    for i in range(1, n):
        for j in range(1, i + 1):
            Q[i, j] = ((target - x[i - j]) * Q[i, j - 1] - (target - x[i]) * Q[i - 1, j - 1]) / (x[i] - x[i - j])

    result = Q[n - 1, n - 1]
    return round(result, 10)

print(Neville(x,Fx),"\n")

print("Question 2 Newton's Forward Difference Method\n")

x = [7.2,7.4,7.5,7.6]
Fx = [23.5492, 25.3913, 26.8224, 27.4589]

def NewtonForward(degree,x,Fx): 
    x = x
    Fx = Fx

    lim = len(x)
    difftab = np.zeros((lim, lim))
    difftab[:, 0] = Fx

    #make forward diff table
    for i in range(1, lim):
        for j in range(1, i + 1):
            difftab[i, j] = (difftab[i, j - 1] - difftab[i - 1, j - 1])/ (x[i] - x[i - j])
    
    return round(difftab[degree, degree], 10)

for i in range(1,4):
    print("Degree", i, ":", NewtonForward(i,x,Fx),"\n")

print("Question 3 approximating 7.3 using approximations from question 2\n")
x = [7.2,7.4,7.5,7.6]
Fx = [23.5492, 25.3913, 26.8224, 27.4589]

a1 = Fx[0] +NewtonForward(1,x,Fx)*(7.3-x[0])
a2 = a1 + NewtonForward(2,x,Fx)*(7.3-x[0])*(7.3-x[1])
a3 = a2 + NewtonForward(3,x,Fx)*(7.3-x[0])*(7.3-x[1])*(7.3-x[2])

print("Approximation of 7.3 at Degree 1:", a1)
print("\nApproximation of 7.3 at Degree 2:", a2)
print("\nApproximation of 7.3 at Degree 3:", a3,"\n")
    


print("Question 4 Divided Difference Hermite Matrix\n")

import numpy as np

x = [3.6,3.8,3.9]
Fx = [1.675,1.436,1.318]
dFx = [-1.195,-1.188,-1.182]

n = len(x)
Hx = np.repeat(x, 2)
Hf = np.repeat(Fx, 2)

size = 2*n
difftable = np.zeros((size, size-1))

# create intiial table
difftable[:, 0] = Hx
difftable[:, 1] = Hf
u = 1
for i in range(n):
    difftable[2 * i + 1, 2] = dFx[i] 
    if i != 0:
        difftable[2 * i, 2] = (Fx[i] - Fx[i-1]) / (x[i] - x[i-1])


#generate divided difference values

for i in range (size):
    for j in range(3, size-1):
        if i %2 == 0:
            difftable[i, j] = (difftable[i, j - 1] - difftable[i - 1, j - 1]) / (difftable[i,0] - difftable[i - 1,0])
        else:
            difftable[i, j] = (difftable[i, j - 1] - difftable[i - 1, j - 1]) / (difftable[i,0] - difftable[i - 2,0])

for i in range(3,5):
    for j in range(0,3):
        if j == 2 and i == 3:
            difftable[j, i] = difftable[j, i]
        else:
            difftable[j, i] = 0



pd.set_option("display.float_format", "{:.8e}".format)
df = pd.DataFrame(difftable)
print(df,"\n")



print("Question 5 Cubic Spline Interpolation\n")

x = [2,5,8,10]
Fx = [3,5,7,9]


#Matrix A Calculation
matA = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        if i == 0 and j == 0:
            matA[i,j] = 1
        elif i ==0 and j ==1:
            matA[i,j] = 0
        elif i == len(x)-1 and j == len(x)-2:
            matA[i,j] = 0
        elif i == len(x)-1 and j == len(x)-1:
            matA[i,j] = 1
        elif i == j:
            matA[i,j] = 2*(x[i+1] - x[i-1])
        elif abs(i-j) == 1:
            matA[i,j] = x[i] - x[i-1]
        else:
            matA[i,j] = 0
print(matA)

#Vector B Calulation
vecB = np.zeros(len(x))

for i in range(len(x)):
    if i == len(x)-1 or i == 0:
        vecB[i] = 0
    else:
        vecB[i] = ((3/(x[i+1] - x[i])) * (Fx[i+1] - Fx[i])) - (3/(x[i] - x[i-1]) * (Fx[i] - Fx[i-1]))

print(vecB)

#Vector X calculation
vecX = np.linalg.solve(matA, vecB)
print(vecX)

