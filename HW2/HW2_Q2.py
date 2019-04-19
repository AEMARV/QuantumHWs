from library import *
import typing
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_swap_op():
	''' Creates the controlled swap operation linear operator'''
	op = np.asmatrix(np.zeros((4,4),dtype=np.complex))

	for bin0 in range(2):
		for bin1 in range(2):
			# This way of initing the matrix is inspired by the ket notation
			# I was truly amazed by the conceptual simplicity of the outer product notation
			indexin = 2*bin0 + bin1
			indexout = bin0 + 2*bin1
			op[indexout,indexin] = 1
	return op


def plot3d(mat:np.matrix,step):
	''' Function wrapper to plot 3d'''
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_xlabel("$\phi$", fontsize=20, rotation=150)
	ax.set_ylabel(r"$\theta$", fontsize=20, rotation=150)

	ax.set_zlabel('$\mathbb{P}(C=1)$', fontsize=20, rotation=150)
	plt.title(r"$|\psi_1\rangle= |0\rangle,\quad|\psi_2\rangle=sin(2\pi\theta)|0\rangle + e^{2\pi i \phi}cos(2\pi \theta)|1\rangle$")
	X = np.linspace(0, 1, step)
	Y = np.linspace(0, 1, step)
	X, Y = np.meshgrid(X, Y)
	mat = np.asarray(mat)
	surf = ax.plot_surface(X, Y, mat, cmap=mpl.cm.coolwarm,
	                       linewidth=0, antialiased=False)
	plt.show()


def plot_polar_parameterized_state(step=100):
	state1 = usualkets(0)
	ket0 = usualkets(0)
	ket1 = usualkets(1)
	probout = np.zeros((step,step))
	for i,theta in enumerate(np.linspace(0,1,step)):
		for j, phi in enumerate(np.linspace(0,1,step)):
			state2 = np.sin(2*np.pi*theta)*ket0 + np.exp(2j*np.pi*phi)*np.cos(2*np.pi*theta)*ket1
			prob1 = swaptest(state1,state2)
			probout[i,j]= prob1
	plot3d(probout,step)


def swaptest(state1, state2):
	''' Gets qstate1 and qstate 2 and performs the swap test,
		the output is the probability of observing 1 in the control gate
	'''
	swap = create_swap_op()
	full_state = np.kron(state1,state2)
	rotated_state = hadamard_test(full_state,swap)
	prob1 =measure(rotated_state,1)
	return np.absolute(prob1)

if __name__ == '__main__':
	state1 = usualkets(0)
	state2 = usualkets(0)
	prob1 = swaptest(state1, state2)
	printsection()
	print("q-state 1 : \n")
	matprint(state1)

	print("\nq-state 1 : \n")
	matprint(state2)
	print("\n Probability of measuring 1 in the control is : \n")
	print(prob1)

	printsection()

	state1 = usualkets(0)
	state2 = usualkets(1)
	prob1 = swaptest(state1, state2)

	print("\nq-state 1 : \n")
	matprint(state1)

	print("\nq-state 1 : \n")
	matprint(state2)
	print("\n Probability of measuring 1 in the control is : \n")
	print(prob1)
	plot_polar_parameterized_state()


