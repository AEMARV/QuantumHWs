import numpy as np
import math as m
import matplotlib as mpl
from matplotlib import pyplot as plt

from library import *

def create_randn_map(bitnum):
	''' Creates a random complex unitary operator
		The real and imaginary parts are sampled with normal distriubtion
	'''
	dim= 2**bitnum
	a = np.random.randn(dim,dim)
	a =a +  1j*np.random.randn(dim,dim)
	u,v = np.linalg.qr(a)
	return u


def hadamard(state):
	''' Applies a Hadamard operation to the q-state
		Note that the operation naturally operates on the q-state of all the q-bits

	'''
	H = np.asmatrix(np.array([[1,1],[1,-1]])/np.sqrt(2))
	sh = int(state.shape[0]/2)
	id = np.asmatrix(np.eye(sh,dtype=np.complex))
	H = np.kron(H,id)
	return H.dot(state)

# THIS IS THE HADAMARD TEST FUNCTION REQUESTED FROM ASSIGNMENT
def hadamard_test(state, U):
	''' Applies the Hadamard test with Unitary operation U.
		The operation Automatically concatenates a |0> state.
	'''
	g = usualkets(0)
	state = np.kron(g,state)
	state = hadamard(state)
	state = controlled_U_operator(U).dot(state)
	state = hadamard(state)

	return state


def controlled_U_operator(U):
	''' Creates the linear operator of the controlled operation'''
	ket0 = usualkets(0)
	ket1 = usualkets(1)
	k0 = np.outer(ket0,ket0)
	k1 = np.outer(ket1,ket1)
	id = np.asmatrix(np.eye(U.shape[0],U.shape[1],dtype=np.complex))
	CU = np.kron(k1,U) +np.kron(k0,id)
	return CU


def usualkets(i):
	''' Constructs |0> and |1>'''
	ket = np.asmatrix(np.zeros((2,1),dtype=np.complex))
	ket[i,0]= 1
	return ket


def ketplus():
	''' Constructs the |+> states'''
	ket = np.asmatrix(np.zeros((2,1),dtype=np.complex))
	ket[0,0] = 1
	ket[1, 0] = 1
	return normalize_state(ket)


def ketminus():
	''' Constructs the |-> states'''
	ket = np.asmatrix(np.zeros((2, 1), dtype=np.complex))
	ket[0, 0] = 1
	ket[1, 0] = -1
	return normalize_state(ket)


def randomstate(numbit):
	""" Creates a random q-state on numbit bits"""
	statec = 1j*np.asmatrix(np.random.randn(2**numbit,1))
	state = np.asmatrix(np.random.randn(2**numbit,1)) + statec
	state = normalize_state(state)
	return state


def normalize_state(state):
	''' Normalizes a q-state'''
	norm = np.absolute(state)
	norm = np.multiply(norm,norm)
	norm = np.sqrt(np.sum(norm))
	state= state/norm
	return state


def prob_amplitude(state):
	''' Calculates the probability amps of a state'''
	prob = np.multiply(np.conj(state),state)
	return prob


def printsection():
	print("____________________________________________\n\n")


def bitflip_experiment():
	bitflipper = np.asmatrix(np.array([[0, 1], [1, 0]], dtype=complex))
	state = ketplus()
	state2 = hadamard_test(state, bitflipper)
	printsection()
	print("Init State(|+> state):\n")
	matprint(state)

	print("\nState after Hadamard Test with bitflip(|+> state):\n")
	matprint(abs((prob_amplitude(state2))))

	state = ketminus()
	state2 = hadamard_test(state, bitflipper)

	printsection()
	print("Init State(|- > state):\n")
	matprint(state)

	print("\nState after Hadamard Test with bitflip(|- > state):\n")
	matprint(abs((prob_amplitude(state2))))

def measure(state,bit_state):
	''' Measures the first bit
	bit_state is an integer determining whether to measure probability
	 state 1 or 0 of the first bit
	'''
	projector = np.outer(usualkets(bit_state),usualkets(bit_state))
	projector = np.kron(projector,np.asmatrix(np.eye(2)))
	state = projector.dot(state)
	prob = (state.T).dot(np.conj(state))
	return prob[0,0]

def phase_experiment():
	precision = 100
	probout = np.zeros((precision))
	for i, phi in enumerate(np.linspace(0,1,precision)):
		U = np.diag(np.array([1,np.exp(2j*np.pi*phi)]))
		state = usualkets(1)
		state_updated = hadamard_test(state,U)
		prob1 = measure(state_updated,1)
		prob0 = measure(state_updated, 0)
		probout[i] = np.absolute(prob1)
	fig = plt.figure()
	ax = fig.gca()
	plt.title("Probability of measurement of 1 in control\n"
	          +r"$U=diag(1,e^{2\pi i \phi})$")
	ax.set_xlabel("$\phi$")
	ax.set_ylabel("$\mathbb{P}(C=1)$")
	plt.plot(np.linspace(0,1,precision),probout)
	plt.show()

if __name__ == '__main__':
	bitflip_experiment()
	phase_experiment()



