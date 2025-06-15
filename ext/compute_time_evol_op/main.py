# This is copied from https://dojo.qulacs.org/ja/latest/notebooks/5.2_Quantum_Circuit_Learning.html#1.-%E6%A8%AA%E7%A3%81%E5%A0%B4%E3%82%A4%E3%82%B8%E3%83%B3%E3%82%B0%E3%83%8F%E3%83%9F%E3%83%AB%E3%83%88%E3%83%8B%E3%82%A2%E3%83%B3%E4%BD%9C%E6%88%90

import numpy as np
from functools import reduce

nqubit = 3
time_step = 0.77  ## ランダムハミルトニアンによる時間発展の経過時間

I_mat = np.eye(2, dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)


## fullsizeのgateをつくる関数.
def make_fullgate(list_SiteAndOperator, nqubit):
    """
    list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...] を受け取り,
    関係ないqubitにIdentityを挿入して
    I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
    という(2**nqubit, 2**nqubit)行列をつくる.
    """
    list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = []  ## 1-qubit gateを並べてnp.kronでreduceする
    cnt = 0
    for i in range(nqubit):
        if i in list_Site:
            list_SingleGates.append(list_SiteAndOperator[cnt][1])
            cnt += 1
        else:  ## 何もないsiteはidentity
            list_SingleGates.append(I_mat)

    return reduce(np.kron, list_SingleGates)


#### ランダム磁場・ランダム結合イジングハミルトニアンをつくって時間発展演算子をつくる
ham = np.zeros((2**nqubit, 2**nqubit), dtype=complex)
for i in range(nqubit):  ## i runs 0 to nqubit-1
    Jx = -1.0 + 2.0 * np.random.rand()  ## -1~1の乱数
    ham += Jx * make_fullgate([[i, X_mat]], nqubit)
    for j in range(i + 1, nqubit):
        J_ij = -1.0 + 2.0 * np.random.rand()
        ham += J_ij * make_fullgate([[i, Z_mat], [j, Z_mat]], nqubit)

## 対角化して時間発展演算子をつくる. H*P = P*D <-> H = P*D*P^dagger
diag, eigen_vecs = np.linalg.eigh(ham)
time_evol_op = np.dot(
    np.dot(eigen_vecs, np.diag(np.exp(-1j * time_step * diag))), eigen_vecs.T.conj()
)  # e^-iHT

print(time_evol_op)
result = []
for val in time_evol_op.flatten():
    result.append(f"Complex::new({val.real}, {val.imag})")
print("[\n    " + ",\n    ".join(result) + "\n]")
