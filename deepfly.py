import os
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import sqlalchemy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler, LabelBinarizer

from utils import DataFrameSelector, CategoricalEncoder

# "Global" variables
verbose = True
n_entries = 10000
n_hidden1 = 300
n_hidden2 = 100

# import data with sqlalchemy
db = "raw2017.db"
table = "statcast"

engine = sqlalchemy.create_engine("sqlite:///%s" % db, execution_options={"sqlite_raw_colnames": True})
data = pd.read_sql_table(table, engine)
if verbose:
    print("Connected to database.")


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_log_dir = "dnn_model/tb"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_log_dir, name)


if verbose:
    print("Cleaning data...")
# Begin cleaning out bad attributes
data = data[[col for col in data.columns if 'deprecated' not in col]]
bad_attrs = ['des', 'at_bat_number', 'pitch_number', 'spin_dir', 'umpire', 'type', 'game_year', 'pos2_person_id.1',
             'game_pk', 'batter', 'pitcher', 'babip_value', 'index', 'iso_value', 'woba_denom', 'woba_value',
             'launch_speed_angle', 'zone', 'description', 'estimated_ba_using_speedangle',
             'estimated_woba_using_speedangle', 'hit_location', 'bb_type']
for attr in bad_attrs:
    del data[attr]
# We're going to make balls and strikes discrete so the algorithm doesn't define a threshold
# that will lose a lot of information
data['balls'] = data['balls'].astype(str)
data['strikes'] = data['strikes'].astype(str)
# Discretize runners on base as well as outs_when_up, id's
for feature in [col for col in data.columns if col.endswith('id') or col.startswith('on')] + ['outs_when_up']:
    data[feature] = data[feature].astype(str)
# Further discretization
data['inning'] = data['inning'].astype(str)
hit_types = {'single', 'double', 'triple', 'home_run'}
# Turn date attribute into month
data['game_date'] = data['game_date'].apply(lambda s: s[5:7])

# Get continuous and discrete attributes
continuous_attrs = np.array(data._get_numeric_data().columns)
discrete_attrs = np.array([attr for attr in data.columns if attr not in continuous_attrs and attr != 'events'])
# Double check...
for attr in discrete_attrs:
    data[attr] = data[attr].astype(str)

# To ensure the prepared data matrix is accurate
if os.path.exists("entries.npy"):
    entries = np.load("entries.npy")
else:
    entries = np.random.choice(data.shape[0], n_entries, replace=False)
    np.save("entries.npy", entries)
labels = np.array(data['events'])
del data['events']
data_small = data.iloc[entries]
labels_small = labels[entries]

if verbose:
    print("Data cleaned.")

# Begin data preparation!
continuous_pipeline = Pipeline([
    ('selector', DataFrameSelector(continuous_attrs)),
    ('imputer', Imputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

discrete_pipeline = Pipeline([
    ('selector', DataFrameSelector(discrete_attrs)),
    ('encoder', CategoricalEncoder(encoding='onehot-dense'))
])

full_pipeline = FeatureUnion([
    ('c_pipeline', continuous_pipeline),
    ('d_pipeline', discrete_pipeline)
])

# Prepare the data!
if verbose:
    print("Preparing data with pipeline...")

if os.path.exists("prepared.npy"):
    data_prepared = np.load("prepared.npy")
else:
    data_prepared = full_pipeline.fit_transform(data_small)
    np.save("prepared.npy", data_prepared)

if verbose:
    print("Data prepared.")

labels_small_onehot = LabelBinarizer().fit_transform(labels_small)
# Map string labels to encodings
labels_small_cat = np.array([np.argmax(encoding) for encoding in labels_small_onehot])
label_map = {}
for label, encoding in zip(labels_small, labels_small_onehot):
    label_map[np.argmax(encoding)] = label


# Deep Learning hyperparameters
n_inputs = len(data_prepared[0])
n_outputs = len(labels_small_onehot[0])

# Set up train/test set
train_indices, test_indices = train_test_split(list(range(len(data_prepared))), test_size=0.2)
train_data, train_labels = data_prepared[train_indices], labels_small_cat[train_indices]
test_data, test_labels = data_prepared[test_indices], labels_small_cat[test_indices]

# Input tensor
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')

# label tensor
y = tf.placeholder(tf.int32, shape=None, name='y')

# This variable will tell batch norm if we're training or not
training = tf.placeholder_with_default(False, shape=(), name='is_training')

# We'll use He initialization for our weights
he_init = tf.variance_scaling_initializer()

# Primary architecture
with tf.name_scope("FlyNet"):
    bn_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)
    dense = partial(tf.layers.dense, kernel_initializer=he_init)
    hidden_1 = dense(X, n_hidden1, name='Hidden1')
    act_1 = tf.nn.selu(bn_layer(hidden_1))
    hidden_2 = dense(act_1, n_hidden2, name='Hidden2')
    act_2 = tf.nn.selu(bn_layer(hidden_2))
    logits = bn_layer(dense(act_2, n_outputs, name='Output'))

# Defining loss function
with tf.name_scope('loss'):
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(x_entropy, name='loss')
    loss_summary = tf.summary.scalar('log_loss', loss)

# Defining back-prop function
learning_rate = 0.001
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# Define evaluation function
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

# This is a utility to save all of the above information
saver = tf.train.Saver()

# Define training parameters
m, n = train_data.shape
n_epochs = 10000
batch_size = 200
n_batches = int(np.ceil(m / batch_size))

# Some logging paths
checkpoint_path = "/tmp/deep_fly_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./deep_fly_model"
file_writer = tf.summary.FileWriter(log_dir('mlb_dnn'), tf.get_default_graph())

# Starting values for early stopping
best_loss = np.inf
epochs_without_progress = 0
max_epochs_without_progress = 50

# Explicit ops for batch normalization
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# Now let's run it!
with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        # if the checkpoint file exists, restore the model and load the epoch number
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        tf.global_variables_initializer().run()

    X_batch, y_batch = None, None
    for epoch in range(n_epochs):
        # Iterate through a number of mini-batches (enough to cover the whole training set)
        for iteration in range(n_batches):
            # Get our next batch of training data and targets
            indices = np.random.choice(train_data.shape[0], batch_size, replace=False)
            X_batch, y_batch = train_data[indices], train_labels[indices]
            # Run the network on the batch
            sess.run([training_op, extra_update_ops], feed_dict={X: X_batch, y: y_batch, training: True})
            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
                [accuracy, loss, accuracy_summary, loss_summary],
                feed_dict={X: test_data, y: test_labels})
            file_writer.add_summary(accuracy_summary_str, epoch)
            file_writer.add_summary(loss_summary_str, epoch)

            if epoch % 5 == 0:
                print("Epoch:", epoch,
                      "\tTest accuracy: {:.3f}%".format(accuracy_val * 100),
                      "\tLoss: {:.5f}".format(loss_val))
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))
                if loss_val < best_loss:
                    saver.save(sess, final_model_path)
                    best_loss = loss_val
                else:
                    epochs_without_progress += 5
                    if epochs_without_progress > max_epochs_without_progress:
                        print("Early stopping")
                        break

os.remove(checkpoint_epoch_path)
with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: test_data, y: test_labels, training: False})
print('Best model accuracy:', accuracy_val)
file_writer.close()
