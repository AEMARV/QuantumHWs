import numpy as np
import typing
from library import *
def innerproductsqrd(s1:np.matrix,s2:np.matrix):
	''' The inner product squared of two matrices containing orthornormal bases in columns'''
	inner =np.transpose( s1) * s2
	inner =np.multiply(np.conj(inner),inner)
	return inner
def translate_num_to_char(i):
	if i==0:
		return "x"
	elif i==1:
		return "y"
	elif i==2:
		return "z"
if __name__ == '__main__':
	s_x = np.array([[0,1],[1,0]])

	s_y = np.array([[0,-1j],[1j,0]])
	s_z = np.array([[1,0],[0,-1]])
	s_x = np.asmatrix(s_x)
	s_y = np.asmatrix(s_y)
	s_z  = np.asmatrix(s_z)


	_,eig_s_x = np.linalg.eig(s_x)
	_,eig_s_y = np.linalg.eig(s_y)
	_,eig_s_z = np.linalg.eig(s_z)
	printsection()
	print("\neigen bases of sigma_x in the columns of the following matrix\n")
	matprint(eig_s_x)

	printsection()
	print("\neigen bases of sigma_y in the columns of the following matrix\n")
	matprint(eig_s_y)

	printsection()
	print("\neigen bases of sigma_z in the columns of the following matrix\n")
	matprint(eig_s_z)

	eiglist = [eig_s_x, eig_s_y , eig_s_z]
	for i in [[0,1],[0,2],[1,2]]:
		first_op_char = translate_num_to_char(i[0])
		second_op_char = translate_num_to_char(i[1])
		printsection()
		print("\n The absolute value square of inner products of the eigen bases"+
		      " Corresponding to Pauli Operator sigma_{} and sigma_{} \n\n".format(first_op_char,second_op_char))
		inner = innerproductsqrd(eiglist[i[0]],eiglist[i[1]])
		matprint(inner)
