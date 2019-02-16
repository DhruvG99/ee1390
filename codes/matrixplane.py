from __future__ import print_function  
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n1 = np.array([2,1,-1])
n2 = np.array([1,2,1])
d1 = 3
d2 = 2
#n = np.array([[1,1,1]])
#print(np.matmul(n1.T,n.T))

drs_loi = np.cross(n1,n2)
#A)
#check if direction ratios are same
r = np.zeros(3)
givendr = np.array([1,2,-1])
for i in range(3):
	r[i] = drs_loi[i]/givendr[i]

if((r[0]==r[1])&(r[1]==r[2])&(r[2]==r[0])):
	print("A)Direction ratios are correct")
else:
	print("A)Direction ratios are wrong")

#B)
given_drs = np.array([9,9,3])
dot_prod  = np.dot(drs_loi,given_drs)

if(dot_prod==0):
	print("B)The given line is perpendicular to the Line of intersection of P1 and P2")
else:
	print("B)The given line is not perpendicular to the Line of intersection of P1 and P2")

#C)
normn1 = np.linalg.norm(n1)
normn2 = np.linalg.norm(n2)

angle = np.rad2deg(np.arccos(np.absolute(np.dot(n1,n2)/(normn1*normn2))))
print("C)The acute angle between P1 and P2 is ",angle)

#D)
A = np.array([4,2,-2])
P = np.array([2,1,1])
d = np.matmul(drs_loi.T,A)
d3 = d
n3 = drs_loi
dist = np.absolute(np.dot(drs_loi,P)-d)/(np.linalg.norm(drs_loi))
print(dist)
print(2/np.sqrt(3))


#plotting
t = np.linspace(-50,50,50)
x_loi = t+(4.0/3.0)
y_loi = (1.0/3.0)-t
z_loi = t
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-50, 50, 50)
y = np.linspace(-50, 50, 50)
X, Y = np.meshgrid(x, y)
Z1 = (d1 - n1[0] * X - n1[1] * Y) / n1[2]
Z2 = (d2 - n2[0] * X - n2[1] * Y) / n2[2]
Z3 = (d3 - n3[0] * X - n3[1] * Y) / n3[2]

ax.plot(X.flatten(),Y.flatten(),Z1.flatten(), 'mo',label = 'Plane 1')
ax.plot(X.flatten(),Y.flatten(),Z2.flatten(), 'bo',label = 'Plane 2')
ax.plot(x_loi,y_loi,z_loi,'go', label = 'Line of intersection')
ax.plot(X.flatten(),Y.flatten(),Z3.flatten(), 'yo',label = 'Plane 3')
ax.plot([2.], [1.], [1.], marker='o', markersize=8,color = 'red')

# adjust the view so we can see the point/plane alignment
ax.view_init(0,20)
#plt.tight_layout()

plt.legend()
plt.show()