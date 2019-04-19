import numpy as np
from library import *
import typing

import cmath as m
def check_MU(s1:np.matrix,s2:np.matrix):

	inner =np.transpose( s1).dot( s2)
	inner =np.multiply(np.conj(inner),inner)
	print(inner)
def trace_inner_product(a:np.matrix,b:np.matrix):
	a = np.transpose((np.conj(a))).dot(b)
	return np.trace(a)
def translate_num_to_char(i):
	if i==0:
		return "x"
	elif i==1:
		return "y"
	elif i==2:
		return "z"
def construct_X_mat(d):
	xmat = np.asmatrix(np.zeros((d,d),dtype=np.complex))
	for i in range(xmat.shape[0]):
		xmat[i,(i+1)%d]=1

	return xmat
def construct_Z_mat(d):
	xmat = np.asmatrix(np.zeros((d,d),dtype=np.complex))
	for i in range(xmat.shape[0]):
		xmat[i,i]= m.exp(i*2*m.pi *1j /d)
	return xmat
def construct_M(x,z,a,b):
	M = np.eye(x.shape[0],dtype=np.complex)
	for i in range(a):
		M = M.dot( x)
	for i in range(b):
		M = M.dot( z)
	return M

if __name__ == '__main__':
	d = int(input("Please Enter the dimension:"))
	X = construct_X_mat(d)
	Z = construct_Z_mat(d)

	Mcollection = []
	for i in range(d):
		for j in range(d):
			Mcollection += [construct_M(X,Z,i,j)]
	sz = Mcollection.__len__()
	projectionmat = np.asmatrix(np.zeros((sz,sz),dtype=np.complex))
	np.set_printoptions(suppress=True,)
	for i in range(sz):
		for j in range(sz):
			projectionmat[i,j] = trace_inner_product(Mcollection[i],Mcollection[j])

	print("The followin matrix contains the inner products of the bases")
	print("i,j position represnet the inner product of the ith and jth bases")
	matprint(np.abs(projectionmat))
	print("The matrix is diagonal, therefore the bases are orthogonal")