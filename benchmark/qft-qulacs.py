from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import X, RY, DenseMatrix
import numpy as np

num_of_qubits = 18

state = QuantumState(num_of_qubits)
qc = QuantumCircuit(num_of_qubits)

for i in range(num_of_qubits):
    qc.add_H_gate(num_of_qubits - 1 - i)
    for j in range(i + 1, num_of_qubits):
        gate = DenseMatrix(num_of_qubits - 1 - j, [[1 , 0], [0, np.exp(1j * 2.0*np.pi / 2.0**(j + 1))]])
        gate.add_control_qubit(j, 1)
        qc.add_gate(gate)

qc.update_quantum_state(state)
print (state.get_vector())  # Print the state vector
