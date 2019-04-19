from library import *
from matplotlib import pyplot as plt
def phased_register(phi,nbits):
	''' Constructs the outcome phase kickbacks of the controlled gates'''
	for i in range(nbits):
		temp_state = (usualkets(0) + np.exp(2j * np.pi * (2 ** i) * phi) * usualkets(1)) / np.sqrt(2)
		if i==0:
			reg = temp_state
		else:
			reg = np.kron(temp_state,reg)
	return reg
def ifourier_op(nbits):
	''' Instantiates an inverse fourier operator over 2^nbits  states'''
	ifour = create_randn_map(nbits)
	states = 2**nbits
	for i in range(states):
		for j in range(states):
			ifour[i,j] = np.exp(-2j * np.pi * i * j / (states))/np.sqrt(states)
	return ifour


if __name__ == '__main__':
	printsection()
	printsection()
	print("\nThe phase is going to be plotted between 0 and 1")
	print("\nI felt that it is a better representation of what is happening.")
	print("\nIt can be easily change by commenting Line36 and uncommenting Line37")
	printsection()
	printsection()
	bitnum = int(input("Please Enter Bit Number: "))
	phase = float(input("Please Enter the Phase: "))
	inv_fourier_op= ifourier_op(bitnum)
	state = phased_register(phase,bitnum)
	phase_state = inv_fourier_op.dot(state)
	probs = prob_amplitude(phase_state)
	ax = plt.figure().gca()
	ax.set_xlabel("$discrete phase$", fontsize=10)
	ax.set_ylabel(r"$\mathbb{P}rob$", fontsize=10)
	plt.title("The phase's real valued representation [0,1).")
	plt.stem(np.arange(2**bitnum)/(2**bitnum),probs,'-.')
	#plt.stem(np.arange(2 ** bitnum), probs, '-.')
	#plt.plot(np.arange(2 ** bitnum) / 2 ** bitnum, probs)
	plt.show()

	pass