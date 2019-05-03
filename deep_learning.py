import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from DataStream import Data_Stream
from scipy import integrate
from scipy import interpolate
import math
import sys
from scipy import signal
import pandas as pd
import tensorflow as tf
from numba import cuda
import numba
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras import backend as K 
from IPython.display import clear_output
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, Bidirectional, LSTM
from tensorflow.python.keras.optimizers import RMSprop, Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def measure_accuracy(ground_truth, test):
    diff_vectors = ground_truth - test
    accuracy = np.mean(np.linalg.norm(diff_vectors, ord = 2, axis = 1))
    return accuracy

def create_input_and_output(data, just_acc=False, higher_freq=True):
    ## Needed Data
    gps = data.gps[:,1:3] 
    if(higher_freq):
        acc = data.acc_with_grav_ERC[:, 1:4]
    else:
        acc = data.acc_ERC[:, 1:4]
        
    mag = data.mag[:, 1:4]
    gyro = data.gyro[:, 1:4]

    time_series = data.acc_with_grav_ERC[:, 0]
    ground_truth = data.ground_truth.dis[:, 1:3]
    delta_time = np.diff(time_series, axis=0)
    delta_time = np.concatenate(([[0]], delta_time))
    
    
    # Choose which data to include in input
    if (just_acc):
        input_data = np.concatenate((gps, acc, delta_time), axis=1)
    else:
        input_data = np.concatenate((gps, acc, gyro, mag, delta_time), axis=1) ## Feature Vector Length = 11
    return input_data, ground_truth

def load_datasets(files, higher_freq=True, no_cache=False):
    training_datasets = []
    for file in files:
        data = Data_Stream(file, load_truth=True, higher_freq=higher_freq, no_cache=no_cache)
        input_data, ground_truth = create_input_and_output(data, higher_freq=higher_freq)
        training_datasets.append([input_data, ground_truth])
    return training_datasets

def scale_dataset(training_dataset, test_dataset):
    scaled_training_dataset = []
    scaled_test_dataset = []
    
    for activity in training_dataset:
        scaled_training_dataset.append([x_scaler.transform(activity[0]), 
                                        y_scaler.transform(activity[1])])
    for activity in test_dataset:
        scaled_test_dataset.append([x_scaler.transform(activity[0]), 
                                    y_scaler.transform(activity[1])])
    
    return scaled_training_dataset, scaled_test_dataset
    
def get_seqs(sequence_length, dataset, offset):
    
    x_seqs = []
    y_seqs = []
    for activity in dataset:
        
        ##Create all sequencest successes is that almost none of these suc-cesses were achieved with a vanilla recurrent neural network. Rat
        for i in range(0, len(activity[0]) - sequence_length, offset):
            x_seqs.append(activity[0][i:i+sequence_length])
            y_seqs.append(activity[1][i:i+sequence_length])
            
    x_seqs = np.asarray(x_seqs)
    y_seqs = np.asarray(y_seqs)
    return x_seqs, y_seqs

def batch_generator(batch_size, x_seqs, y_seqs):  
    
    x_seqs, y_seqs = shuffle(x_seqs, y_seqs)
    #Print number of batches required to pass all sequences in an epoch
    while True:
        #For each batch in an epoch, given n sequences that are in shuffled order
        for i in range(int(len(x_seqs)/batch_size + 1)):
            i1 = i+1
            yield (x_seqs[i*batch_size:i1*batch_size], y_seqs[i*batch_size:i1*batch_size])
        #Shuffle sequences between epochs
        x_seqs, y_seqs = shuffle(x_seqs, y_seqs)
        

def custom_loss(y_true, y_pred):
    
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    
    
	#     ts = tf.reshape(y_true_slice, (-1, 2))
    
	#     ps = tf.reshape(y_pred_slice, (-1, 2))
    
	#     def get_min_dist(pi):
	# 	#         print(pi.get_shape(), ts.get_shape())
	#         eu_dists = tf.norm(ts-pi, ord='euclidean', axis=1)
	#         min_dist = tf.reduce_min(eu_dists)
	#         return min_dist


	#     min_dists = tf.map_fn(get_min_dist, ps, dtype=tf.float32)
	#     print("Minimum distances to Ground Truth Points Shape", min_dists.get_shape())
	#     mean_dist = tf.reduce_mean(min_dists)
	#     print("Mean Mimimum Distance Shape", mean_dist.get_shape())
    
	#     min_dists_tot = 0.0
	#     print(ps.shape)
	#     for pi in ps:
	#         min_t = sys.float_info.max
	#         for ti in ts:
	#             t = tf.norm(ti-pi, ord='euclidean')
	#             if(t < min_t):
	#                 min_t = t
	#         min_dists_tot += min_t
        
	#     mean_dist = min_dists_tot/len(ps)
    
    eu_dists = tf.norm(y_true_slice-y_pred_slice, ord='euclidean')
    loss_mean = tf.reduce_mean(eu_dists)
    
    return loss_mean

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.figure(figsize=(9, 8))
        plt.plot(self.x, self.losses, label="Training Loss", c='r')
        plt.plot(self.x, self.val_losses, label="Validation Loss", c='b')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=2, frameon=False)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig("Loss.pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.ioff()
        plt.close()
        # plt.show();

def plot_dataset(dataset, seq_len=100, filenames=[]):
    count = 0
    for activity in dataset:
        print("Plotting ", filenames[count])
        input_data = activity[0]
        ground_truth = activity[1]
        orig_gps = activity[0][:, 0:2]
        
        
        #Pad input
        padding = np.zeros((seq_len, input_data.shape[1]))
        input_data = np.concatenate((padding, input_data))

        ##Scale down the trial input, get predicted output
        input_data = x_scaler.transform(input_data)
        input_data = np.expand_dims(input_data, axis=0)
        predicted_output = model.predict(input_data)

        
        ## Remove data upto the start point
    #         ground_truth = ground_truth[:]
    #         orig_gps = orig_gps[:]
        predicted_output = y_scaler.inverse_transform(predicted_output[0])[seq_len:]
        warm_up=30
        ##Print Graphs of X against Y
        plt.figure(figsize=(8,5))
        plt.plot(ground_truth[warm_up:-warm_up, 0], ground_truth[warm_up:-warm_up, 1], label='ground truth', color = 'g')
        plt.plot(predicted_output[warm_up:-warm_up, 0], predicted_output[warm_up:-warm_up, 1], label='seen training data', color = 'b')
        plt.plot(orig_gps[warm_up:-warm_up, 0], orig_gps[warm_up:-warm_up,1], label='original gps', color = 'r')
        plt.legend()
        plt.xlabel('Position X (Metres)')
        plt.ylabel('Position Y (Metres)')
        plt.savefig(str(filenames[count])+".pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.show()

        print("Accuracy of GPS: ", measure_accuracy(ground_truth[warm_up:-warm_up], orig_gps[warm_up:-warm_up]))
        print("Accuracy of RNN: ", measure_accuracy(ground_truth[warm_up:-warm_up], predicted_output[warm_up:-warm_up]))
        count+=1








x_dim      = 12
y_dim      = 2
gps_bound  = 3000.0 #Actual bound (3000)
acc_bound  = 3000.0 #Actual bound (30)
gyro_bound = 3000.0 #Actual bound (2)
mag_bound  = 3000.0 #Actual bound (30)
dt_bound   = 3000.0 #Actual bound (0.1)
custom_scale_matrix = np.asmatrix([gps_bound, gps_bound, acc_bound, acc_bound, acc_bound,  gyro_bound, gyro_bound, gyro_bound, mag_bound, mag_bound,mag_bound, dt_bound])
custom_scale_matrix = np.concatenate((custom_scale_matrix, -custom_scale_matrix))
x_scaler = MinMaxScaler()
x_scaler = x_scaler.fit(custom_scale_matrix)
y_scaler = MinMaxScaler()
y_scaler = y_scaler.fit(custom_scale_matrix[:, 0:2])


data = Data_Stream('tut0', load_truth=True, higher_freq=False, no_cache=True)
plt.plot(data.gps[:, 1], data.gps[:, 2])
plt.show()
data = Data_Stream('tut-rev0', load_truth=True, higher_freq=False, no_cache=True)
plt.plot(data.gps[:, 1], data.gps[:, 2])
plt.show()
# data = Data_Stream('run-harbour-rev0', load_truth=True, higher_freq=False, no_cache=True)
# plt.plot(data.gps[:, 1], data.gps[:, 2])
# plt.show()

###########################################################
################## CHOOSE TRAINING AND TESTING FILES
###########################################################
print("\n###########################################################")
print("######### Loading Data")
print("###########################################################\n")
cycling_files = ['cyc-asda0', 'cyc-asda-rev0', 'cyc-bro0', 'cyc-bro-rev0',  'cyc-tuto0', 'cyc-tuto-rev0', 'cyc-tuto1', 'cyc-tuto-rev1', 'cyc-tuto2', 'cyc-tuto-rev2']

running_files = ['run-harbour0','run-john0']
walking_files = ['uni', 'uni1','uni2','uni3', 'mb0', 'tutoring0', 'dog0', 'train0']


training_files = ['uni','uni2', 'tutoring0', 'dog0', 'train0', 'mb0']
testing_files  = ['uni3', 'uni1']
training_dataset = load_datasets(training_files, higher_freq=False, no_cache=False)
testing_dataset = load_datasets(testing_files, higher_freq=False, no_cache=False)
scaled_training_dataset, scaled_testing_dataset = scale_dataset(training_dataset, testing_dataset)


# print("\n###########################################################")
# print("######### Scaling Data")
# print("###########################################################\n")
# print("Scale")

###########################################################
################## HYPER-PARAMETERS
###########################################################
print("\n###########################################################")
print("######### Hyper-Parameters")
print("###########################################################\n")

seq_len         = 300
seq_offset      = int(seq_len/20)
warmup_steps    = 5
batch_size      = 32
print("Sequence Length: ", seq_len)
print("Sequence Offset: ", seq_offset)

###########################################################
################## MAKE SEQUENCES
###########################################################
print("\n###########################################################")
print("######### Training Data")
print("###########################################################\n")
PRINT_DEBUG = False
training_length = 0
testing_length  = 0
print("Training Activities: ", len(training_dataset))
print(training_files)
for i in range(len(training_dataset)):
    if(PRINT_DEBUG): print("    ",training_files[i],  "Activity ", i, " Length: ", len(training_dataset[i][0]))
    training_length += len(training_dataset[i][0])
if(PRINT_DEBUG): print("Total Training Length: ", training_length)

print("Number of Testing Activities: ", len(testing_dataset))
print(testing_files)
for i in range(len(testing_dataset)):
    if(PRINT_DEBUG): print("    ", testing_files[i], "Activity ", i, " Length: ", len(testing_dataset[i][0]))
    testing_length += len(testing_dataset[i][0])
if(PRINT_DEBUG): print("Total Training Length: ", testing_length)

print("\n###########################################################")
print("######### Training Sequences and Batches")
print("###########################################################\n")
x_train_seqs, y_train_seqs = get_seqs(sequence_length=seq_len, dataset=scaled_training_dataset, offset=seq_offset)
print("Training Data Consists of ", x_train_seqs.shape[0], " Unique Sequences of Length ", seq_len)
x_test_seqs, y_test_seqs = get_seqs(sequence_length=seq_len, dataset=scaled_testing_dataset, offset=seq_offset)
print("Testing Data Consists of  ", x_test_seqs.shape[0], " Unique Sequences of Length ", seq_len)


sequences = x_train_seqs.shape[0]
batch_size = min(sequences, batch_size)
print("\nTraining Batch Size: ", batch_size)
batches_per_epoch = int(sequences/batch_size)+1
# batches_per_epoch = 1+int(training_length/(seq_len*batch_size))
print("Batches per Epoch: ", str(batches_per_epoch))


print("\n###########################################################")
print("######### Validation Data")
print("###########################################################\n")
val_batch_size = int(0.75*x_test_seqs.shape[0])
print("\nNumber of Validation Sequences: ", val_batch_size)
val_generator = batch_generator(batch_size=val_batch_size, x_seqs=x_test_seqs, y_seqs=y_test_seqs)
val_batch_x, val_batch_y = next(val_generator)
validation_data = (val_batch_x, val_batch_y)


###########################################################
################## COMPILE MODEL
###########################################################
print("\n###########################################################")
print("######### Model")
print("###########################################################\n")
sess = tf.InteractiveSession()

model = Sequential()
model.add(GRU(units=128, return_sequences=True, input_shape=(None, x_dim,)))
model.add(GRU(units=128, return_sequences=True, input_shape=(None, x_dim,)))
# model.add(Bidirectional(GRU(units=128, return_sequences=True), merge_mode='ave', input_shape=(None, x_dim,)))
# model.add(Bidirectional(GRU(units=128, return_sequences=True), merge_mode='ave', input_shape=(None, x_dim,)))
model.add(Dense(y_dim, activation='linear'))
optimizer = Adam(lr=1e-4)
model.compile(loss=custom_loss, optimizer=optimizer)
model.summary()


###########################################################
################## CREATE CALLBACK and TRAIN MODEL
###########################################################
# path_checkpoint = 'RNN.checkpoint'
# callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
# callback_tensorboard = TensorBoard(log_dir='./23_logs/', histogram_freq=0, write_graph=False)

callback_early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, min_lr=1e-9, patience=2, verbose=1)      
plot_losses = PlotLosses()
callbacks = [callback_early_stopping, plot_losses, callback_reduce_lr]
generator = batch_generator(batch_size=batch_size, x_seqs=x_train_seqs, y_seqs=y_train_seqs)
model.fit_generator(generator=generator, verbose=2, epochs=40, steps_per_epoch=batches_per_epoch, validation_data=validation_data, callbacks=callbacks)



###########################################################
################## TESTING DATA
###########################################################
print("\n###########################################################")
print("######### Testing Data")
print("###########################################################\n")
for activity in scaled_testing_dataset:
    print("Test Data")
    result = model.evaluate(x=np.expand_dims(activity[0], axis=0),
                            y=np.expand_dims(activity[1], axis=0))
    print("loss (test-set):", result)

###########################################################
################## PLOT TEST AND TRAINING DATA
###########################################################
plot_dataset(testing_dataset,  seq_len=seq_len, filenames=testing_files)
print("\n###########################################################")
print("######### Training Data")
print("###########################################################\n")
plot_dataset(training_dataset, seq_len=seq_len, filenames=training_files)

tf.reset_default_graph()
sess.close()