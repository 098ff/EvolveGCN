import tensorflow as tf
from typing import Tuple

class GCNSkipLayer(tf.keras.layers.Layer):
    def __init__(self, units: int, activation=None, kernel_initializer="glorot_uniform", dtype=tf.float32):
        super(GCNSkipLayer, self).__init__(dtype=dtype)

        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        last_dim_nodes = tf.TensorShape(input_shape[1])[-1]
        last_dim_skip = tf.TensorShape(input_shape[2])[-1]

        self.kernel_nodes = self.add_weight(
            name='kernel_nodes', 
            shape=[last_dim_nodes, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        
        self.kernel_skip = self.add_weight(
            name='kernel_skip', 
            shape=[last_dim_skip, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        
        # 🟢 สิ่งที่เพิ่มเข้ามา: Learnable Gating Parameter
        self.alpha = self.add_weight(
            name='alpha_gate',
            shape=[1], # เป็นตัวเลข 1 ตัวสำหรับบาลานซ์น้ำหนักทั้งเลเยอร์
            initializer=tf.keras.initializers.Constant(0.5), # เริ่มต้นที่ 0.5 (ให้น้ำหนัก 50/50 เท่ากัน)
            dtype=self.dtype,
            trainable=True)
            
        self.built = True

    def call(self, inputs: Tuple[tf.SparseTensor,tf.Tensor,tf.Tensor], training=None, mask=None):
        adj, nodes, skip_input = inputs
        kernel_branch = tf.matmul(tf.sparse.sparse_dense_matmul(adj,nodes),self.kernel_nodes)
        skip_branch = tf.matmul(skip_input,self.kernel_skip)
        
        # 🟢 สิ่งที่เปลี่ยนไป: คุมน้ำหนักการบวกด้วย Gate
        # ใช้ Sigmoid เพื่อบีบค่า alpha ให้อยู่ในช่วง 0.0 ถึง 1.0 เสมอ
        gate = tf.nn.sigmoid(self.alpha) 
        
        # ถ่วงน้ำหนัก (ถ้า gate เข้าใกล้ 1 จะเน้น GCN, ถ้าเข้าใกล้ 0 จะเน้น Skip)
        combined = (gate * kernel_branch) + ((1.0 - gate) * skip_branch)

        return self.activation(combined)