import tensorflow as tf
import numpy as np
import scipy.sparse as sp

def convert_scipy_CRS_space_to_tensor(sparce_mat: sp.csr.csr_matrix):
    sparce_mat_coo = sparce_mat.tocoo()
    indices = np.transpose(np.array([sparce_mat_coo.row, sparce_mat_coo.col]))
    return tf.SparseTensor(indices, sparce_mat_coo.data, sparce_mat_coo.shape)


def normalize_adjencency_mat(adj_mat: sp.csr.csr_matrix):
    assert len(adj_mat.shape) == 2
    
    a = adj_mat + sp.eye(adj_mat.shape[0], dtype=adj_mat.dtype)
    d = np.array(a.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt).tocsr()
    
    return d_mat @ a @ d_mat