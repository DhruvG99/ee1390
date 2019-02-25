import numpy as np 
import xlrd
import matplotlib.pyplot as plt 

no_of_data = 106
def optimiser(numberofdatavalues):
	filename = 'resistence data.xlsx'
	workbook = xlrd.open_workbook(filename)
	worksheet = workbook.sheet_by_name("Sheet1")

	num_rows = worksheet.nrows #Number of Rows
	num_cols = worksheet.ncols #Number of Columns

	#print(num_cols,num_rows)
	n = numberofdatavalues
	Temp1 = np.array(worksheet.col_values(0,1,n+1),dtype = 'float64')
	Res1  = np.array(worksheet.col_values(1,1,n+1))

	Temp = np.reshape(Temp1,(n,1))
	Res  = np.reshape(Res1,(n,1))
	conv = np.array(np.ones((n,1))*273.0)
	#print conv.shape,'&',conv.dtype
	#print Temp.shape,'&',Temp.dtype

	Temp = Temp+conv #temperature in Kelvin
	Y = np.zeros([n,1])
	Y = 1/Temp

	X = np.zeros((n,3))
	#print(X.shape)

	for i in range(n):
		X[i] = np.array([1,np.log(Res[i]),np.power(np.log(Res[i]),3)])

	w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
	#print(w)
	R1 = 175.86
	x1 = np.array([1,np.log(R1),np.power(np.log(R1),3)])
	x1 = np.reshape(x1,(1,3)).T

	y1  = np.dot(x1.T,w)
	t1 = 1/y1
	#print(t1,w)
	return t1
	

vals = np.zeros((no_of_data-4,))
#print vals.shape
for i in range(4,no_of_data):
	vals[i-4] = optimiser(i)
print(optimiser(50))
w_best = np.array([[0.04222115,-0.01077005,0.00011271]])
R = np.linspace(1,200,200)
y = w_best[0,0] + w_best[0,1] * (np.log(R)) + w_best[0,2] * np.power(np.log(R),3)

filename = 'resistence data.xlsx'
workbook = xlrd.open_workbook(filename)
worksheet = workbook.sheet_by_name("Sheet1")
n = no_of_data
Temps = np.array(worksheet.col_values(0,1,n+1),dtype = 'float64')
Reses = np.array(worksheet.col_values(1,1,n+1),dtype = 'float64')
invtemps = np.zeros((no_of_data,))
for i in range(no_of_data-1):
	invtemps[i] = 1.0/(Temps[i]+273)

#print invtemps
#print vals
x_axis = np.linspace(4,no_of_data,no_of_data-4)
#print(x_axis.shape)
#fig,ax = plt.subplots(figsize = (8,7),dpi = 0)

#plot these 2 together: this is Value vs no. of samples taken
plt.plot(x_axis,vals)
plt.plot(x_axis,473*np.ones(no_of_data-4)) #straight line y=200

#plot these 2 together and zoom in a lot to see how the points are different from the optimum curve
#plt.scatter(Reses,invtemps,color = 'red')  #plot directly from table
#plt.plot(R,y) # plot of 1/T =y vs R , this is with ideal w
plt.grid()
plt.show()
