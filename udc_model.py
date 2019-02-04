import tensorflow as tf
import sys

def get_id_feature(features, key, len_key, max_len):
    ids = features[key]
    ids_len = tf.squeeze(features[len_key], [1])
    ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=int64))
    return ids, ids_len

def create_train_op(loss, hparams):
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=hparams.learning_rate,
        clip_greadients=10.0,
        optimizer=hparams.optimizer)
    return train_op


def create_model_fn(hparams, model_impl):
    def model_fn(features, targets, mode):
        context, context_len = get_id_feature(
            features, "context", "context_len", hparams.max_context_len)

        utterance, utterance_len = get_id_feature(
            features, "utterance", "utterance_len", hparams.max_utterance_len)

        batch_size = targets.get_shape().as_list()[0]


