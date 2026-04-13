import numpy as np
import tensorflow as tf

from data_loader import EllipticDatasetLoader
from models import GCNTwoLayersSkipConnection
from models.layers import GCNLayer, GCNSkipLayer


DATADIR = "elliptic_bitcoin_dataset"
FILTER_UNKNOWN = False
ONLY_LOCAL_FEATURE = False
CLASS_WEIGTHS = [0.7,0.29,0.01]
TEST_SHARE = 0.3
# NUM_EPOCH = 10
NUM_EPOCH = 500
LEARNING_RATE = 1e-3

def reset_metrics(list_of_metrics):
    for m in list_of_metrics:
        m.reset_state()


dl = EllipticDatasetLoader(DATADIR, TEST_SHARE, FILTER_UNKNOWN,local_features_only=ONLY_LOCAL_FEATURE)

model = GCNTwoLayersSkipConnection(
    layer_gcn=GCNLayer(64,"relu"),
    dropout=tf.keras.layers.Dropout(0.3),
    layer_gcn_skip=GCNSkipLayer(dl.num_classes)
)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def run_model(adj,nodes,targets,training=False):
    weigths = tf.reduce_sum(CLASS_WEIGTHS * targets, axis=-1)
    logits = model([adj, nodes],training=training)
    loss = loss_func(targets, logits,sample_weight=weigths)
    return logits, loss, weigths


train_loss_metric = tf.keras.metrics.Mean()
train_accuracy_metric = tf.keras.metrics.Accuracy()
train_precision_metric = tf.keras.metrics.Precision()
train_recall_metric = tf.keras.metrics.Recall()
train_f1_class = tf.keras.metrics.F1Score(average=None, name='train_f1_class')
train_f1_micro = tf.keras.metrics.F1Score(average='micro', name='train_f1_micro')
test_loss_metric = tf.keras.metrics.Mean()
test_accuracy_metric = tf.keras.metrics.Accuracy()
test_precision_metric = tf.keras.metrics.Precision()
test_recall_metric = tf.keras.metrics.Recall()
test_f1_class = tf.keras.metrics.F1Score(average=None, name='test_f1_class')
test_f1_micro = tf.keras.metrics.F1Score(average='micro', name='test_f1_micro')
metrics = [train_loss_metric,train_accuracy_metric,train_precision_metric,train_recall_metric
            ,test_loss_metric,test_accuracy_metric,test_precision_metric,test_recall_metric,
            train_f1_class, train_f1_micro, test_f1_class, test_f1_micro]

for epoch in range(NUM_EPOCH):
    reset_metrics(metrics)

    for _, n, t, adj in dl.train_batch_iterator():
        with tf.GradientTape() as tape:
            logits, loss, weigths = run_model(adj, n, t, training=True)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads,model.trainable_weights))

        y_true = tf.cast(tf.argmax(t,axis=-1) == 0,tf.float32)
        y_pred = tf.cast(tf.argmax(logits,axis=-1) == 0,tf.float32)
        train_loss_metric(loss)
        train_accuracy_metric(tf.argmax(t,axis=-1), tf.argmax(logits,axis=-1), sample_weight=weigths)
        train_precision_metric(y_true, y_pred)
        train_recall_metric(y_true, y_pred)
        train_f1_class.update_state(t, logits)
        train_f1_micro.update_state(t, logits)

    for _, n, t, adj in dl.test_batch_iterator():
        logits, loss, weigths = run_model(adj, n, t)

        y_true = tf.cast(tf.argmax(t, axis=-1) == 0, tf.float32)
        y_pred = tf.cast(tf.argmax(logits, axis=-1) == 0, tf.float32)
        test_loss_metric(loss)
        test_accuracy_metric(tf.argmax(t,axis=-1), tf.argmax(logits,axis=-1), sample_weight=weigths)
        test_precision_metric(y_true, y_pred)
        test_recall_metric(y_true, y_pred)
        test_f1_class.update_state(t, logits)
        test_f1_micro.update_state(t, logits)

    illicit_idx = 0 # กำหนด index ของคลาส Illicit

    print(f"Epoch: {epoch}")
    
    # Print for Train
    print(f"TRAIN Loss: {train_loss_metric.result():.5f} | "
          f"Micro F1: {train_f1_micro.result():.4f} | "
          f"Illicit F1: {train_f1_class.result()[illicit_idx]:.4f}")
    
    # Print for Test
    print(f"TEST Loss: {test_loss_metric.result():.5f} | "
          f"Micro F1: {test_f1_micro.result():.4f} | "
          f"Illicit F1: {test_f1_class.result()[illicit_idx]:.4f}")
