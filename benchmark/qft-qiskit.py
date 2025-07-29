from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import Aer
import math

num_of_qubits = 18

state = QuantumRegister(num_of_qubits)
qc = QuantumCircuit(state)

for i in range(num_of_qubits):
    qc.h(state[-1-i])
    for j in range(i+1, num_of_qubits):
        qc.cp(2 * math.pi / 2**(j+1-i), state[-1-j], state[-1-i])

sim = Aer.get_backend('statevector_simulator')
job = sim.run(qc)
print(job.result().get_statevector())