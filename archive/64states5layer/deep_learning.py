import numpy as np
import matplotlib
from matplotlib import pyplot as plt
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

class GroundTruthGPX:
	latlngs = np.asmatrix([0.0, 0.0])
	def __init__(self, filename, gps, load_cache=False):
	    if(load_cache):
	        self.read_from_cache(filename)
	        return
	    
	    directory = 'Data/ground_truth/' + filename + '.gpx'
	    ##Parse GT file
	    f = open(directory, 'r')
	    latlngs = []
	    #Read Data from CSV
	    for line in f:
	        tags = [tag[6:] for tag in line.split("><") if tag[0]=='r']
	        for tag in tags:
	            split_tag = tag.split('"')
	            if(len(split_tag)>1):
	                latlngs.append([float(split_tag[3]), float(split_tag[1])])
	    self.latlngs= np.asmatrix(latlngs)
	    self.dis = self.convert_longlat_to_dis(self.latlngs)

	    ## Add  linear time to GPS, interpolated from start time to end time
	    
	    time_vector = []
	    indexes = []
	    for position in self.dis:
	        distances = gps[:, 1:3] - position
	        magnitudes = np.linalg.norm(distances, ord = 2, axis = 1)
	        indexes.append(np.argmin(magnitudes))
	        time_vector.append([gps[indexes[-1], 0]])

	    self.dis = np.concatenate((np.asmatrix(time_vector), self.dis), axis=1)
	    self.dis[-1] = gps[-1, 0:3]  
	    indexes[-1] = len(gps)
	    
	    previous_index = indexes[0]
	    interpolated_positions = []
	    for i in range(1, len(indexes)):
	        n = indexes[i] - previous_index 
	        a = self.dis[i-1]            
	        b = self.dis[i]
	        interpolated_positions.extend(self.interpolate(a, b, n))
	        previous_index = indexes[i]

	    self.dis = np.squeeze(np.asarray(interpolated_positions))
	    latlngs = np.asmatrix([0.0, 0.0])
	#         if(load_cache):
	#             self.read_from_cache(filename)
	#             return
	    
	#         directory = 'Data/ground_truth/' + filename + '.gpx'
	#         ##Parse GT file
	#         f = open(directory, 'r')
	#         latlngs = []

	#         for line in f:
	#             tags = [tag[6:] for tag in line.split("><") if tag[0]=='r']
	#             for tag in tags:
	#                 split_tag = tag.split('"')
	#                 if(len(split_tag)>1):
	#                     latlngs.append([float(split_tag[3]), float(split_tag[1])])
	#         self.latlngs= np.asmatrix(latlngs)
	    
	#         lin_time = np.asmatrix(np.linspace(0, gps[-1, 0], len(self.latlngs))).T
	#         self.latlngs = np.concatenate((lin_time, self.latlngs), axis=1)
	#         irreg_var = self.latlngs
	#         reg_varX = np.asmatrix(np.interp(gps[:, 0], np.ravel(irreg_var[:,0]), np.ravel(irreg_var[:,1])))
	#         reg_varY = np.asmatrix(np.interp(gps[:, 0], np.ravel(irreg_var[:,0]), np.ravel(irreg_var[:,2])))
	#         self.latlngs = np.concatenate((reg_varX, reg_varY), axis=1)
	    
	#         self.dis = self.convert_longlat_to_dis(self.latlngs)
	#         self.dis = np.concatenate((gps[:, 0], self.dis), axis=1)
	    
	    print("Aligned Ground Truth")
	    
	def interpolate(self, a, b, n):
	    interp_points = []
	    for i in range (n):
	        interp_points.append(a + ((b - a) * (float(i))) / (n-1))
	    return interp_points
	      
	def convert_longlat_to_dis(self, gps):
	    dis_list = [[0.0, 0.0]]
	    start_gps = gps[0]
	    for i in range(1, gps.shape[0]):
	        dis_list.append([self.get_arc_len(start_gps[0, 0], gps[i, 0]),
	                         self.get_arc_len(start_gps[0, 1], gps[i, 1])])
	    return np.asmatrix(dis_list)

	def get_arc_len(self, deg1, deg2):
	    delta_theta = deg2 - deg1
	    delta_theta = ((delta_theta+180)%360)-180
	    earth_R = 6378100
	    return 2.0*math.pi*earth_R*delta_theta/360.0

	def correct_ground_truth(self, ground_truth, gps):
	    ## Function that identifies the closest ground truth point for a given sensor reading
	    def closest_ground_truth_point(x):
	        ## Check every 200 steps
	#             check_step = 300
	        ground_truth_steps = ground_truth
	        ## Get a vector, that represents the distance to all ground truth points
	        distance_vectors = ground_truth_steps - x
	        ## Get the magnitude, and thus identify the index with the minimum distance
	        magnitudes = np.linalg.norm(distance_vectors, ord = 2, axis = 1)
	        min_step_index = np.argmin(magnitudes)
	        
	        
	#             ## Check points around close point
	#             lower_bound = max(0,                   int(min_step_index*check_step - (check_step/2)))
	#             upper_bound = min(len(ground_truth)-1, int(min_step_index*check_step + (check_step/2)))
	#             neighbours = ground_truth[lower_bound : upper_bound]
	        
	#             ##Create distance vector, get magintudes and get closest two points
	#             distance_vectors = neighbours - x
	#             magnitudes = np.linalg.norm(distance_vectors, ord = 2, axis = 1)
	#             minimum_neighbour = np.argmin(magnitudes)
	    
	        return ground_truth_steps[min_step_index]
	    
	    for i in range(len(gps)):
	        ground_truth[i] = closest_ground_truth_point(gps[i])
	    self.dis = ground_truth
	    
	def read_from_cache(self, filename):
	    self.dis = self.read_var_from_cache(filename, "dis")
	    # self.latlngs = self.read_var_from_cache(filename, "latlng")
	    print("READ GROUND TRUTH DATA FROM CACHE")
	    
	def read_var_from_cache(self, filename, var_name):
	    f=open('Data/ground_truth/cache/'+var_name+'/'+filename+'.csv',"r")
	    var=[]
	    for line in f:
	        split = line.split(',')
	        var.append([float(split[0]), float(split[1]), float(split[2])])
	    f.close()
	    return np.asmatrix(var)

	def write_to_cache(self, filename):
	    self.write_var_to_cache(filename, "dis", self.dis)
	    # self.write_var_to_cache(filename, "latlng", self.latlngs)
	    print("WRITTEN GROUND TRUTH DATA TO CACHE")

	def write_var_to_cache(self, filename, var_name, data):
	    f=open('Data/ground_truth/cache/'+var_name+'/'+filename+'.csv',"w+")
	    for entry in data:
	#             line = str(entry[0, 0])+','+str(entry[0, 1])+','+str(entry[0, 2])+'\n'            
	        line = str(entry[0])+','+str(entry[1])+','+str(entry[2])+'\n'

	        f.write(line)
	    f.close()

class Data_Stream:
    gps_latlng = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    gps = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    kal_dis = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    kal_latlng = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    
    rot_vec = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    mag = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    gyro = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    acc_with_grav = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    
    acc_DRC = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    vel_DRC = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    dis_DRC = np.asmatrix([0.0, 0.0, 0.0, 0.0])

    acc_ERC = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    vel_ERC = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    dis_ERC = np.asmatrix([0.0, 0.0, 0.0, 0.0])
    
        
    def __init__(self, filename, invert=False, load_truth=False, higher_freq=False, no_cache=False):
        print("Parsing Data "+ filename)
        
        if((not no_cache) and os.path.exists('Data/streams/cache/acc/'+filename+'.csv')):
            self.read_from_cache(filename)
            self.ground_truth = GroundTruthGPX(filename, self.gps, load_cache=True)
            return
        
        
        self.var_codes = {1.0 : [], 82.0 : [], 84.0 : [], 3.0 : [], 4.0 : [], 5.0 : []}
        directory = 'Data/streams/' + filename +".csv"
        ##Parse file
        f=open(directory, "r")
        start_time = False
        for line in f:
            line = line.split(',')
            if(start_time == False):
                start_time = float(line[0])
            self.process_csv_line(start_time, line)
        self.var_codes[1.0] = np.delete(self.var_codes.get(1.0), (0), axis=0) #GPS cant be (0 0 0 0) at init
        for key, value in self.var_codes.items():
            self.var_codes[key] = np.asmatrix(value)
        
        self.var_codes[13.0] = self.var_codes[1.0]
        self.var_codes[1.0]  = self.swap_xy(self.var_codes[1.0])
        self.var_codes[13.0] = self.swap_xy(self.var_codes[13.0])
        
        ##Convert longitude and latitutde of GPS sensor to meters
        self.var_codes[1.0] = self.convert_longlat_to_dis(self.var_codes.get(1.0))
        
        ##Print Frequency of Acceleration, and Lin. Acceleration
        time_period = np.mean(np.diff(self.var_codes[3.0].T))
        frequency = 1/time_periodyc-bro-rev0
        print("Freq. of Acceleration", frequency)
        time_period = np.mean(np.diff(self.var_codes[82.0].T))
        frequency = 1/time_period
        print("Freq. of Lin. Acceleration", frequency)        
        
        
        ##Interpolate rotation and acceleration so that they occur at the same time step
        if(higher_freq):
            new_time = self.var_codes.get(3.0)[:, 0] # Set timesteps to be that of the acceleration, as it has most readings
            
        else:
            new_time = self.var_codes.get(82.0)[:, 0] # Set timesteps to be that of the acceleration, as it has most readings
            
        
        for key, value in self.var_codes.items():
            irreg_var = value
            reg_varX = np.asmatrix(np.interp(new_time, np.ravel(irreg_var[:,0]), np.ravel(irreg_var[:,1])))
            reg_varY = np.asmatrix(np.interp(new_time, np.ravel(irreg_var[:,0]), np.ravel(irreg_var[:,2])))
            reg_varZ = np.asmatrix(np.interp(new_time, np.ravel(irreg_var[:,0]), np.ravel(irreg_var[:,3])))
            self.var_codes[key] = np.concatenate((new_time, reg_varX, reg_varY, reg_varZ), axis=1)
        print("Interpolated Samples")
        
        
        
        
        
        ##Set class members to matrices read from csv
        self.acc_DRC = self.var_codes.get(82.0)
        self.rot_vec = self.var_codes.get(84.0)
        self.gyro = self.var_codes.get(4.0)
        self.mag = self.var_codes.get(5.0)
        self.acc_with_grav = self.var_codes.get(3.0)
        self.gps = self.var_codes.get(1.0)
        self.gps_latlng = self.var_codes.get(13.0)
        
        # If device axis is wrong, invert data
        self.invert_acceleration()
        
        ##Use rotation vectors to achieve acceleration in ERC
        self.acc_with_grav_ERC = self.rotate_acceleration(self.rot_vec, self.acc_with_grav)
        self.acc_ERC = self.rotate_acceleration(self.rot_vec, self.acc_DRC)

        
        print("Rotated Acceleration")
        
        if(not higher_freq):
            self.integrate_variables()
            print("Integrated Acceleration")

        
        ##Load ground truth if there is one
        if(load_truth):
            self.ground_truth = GroundTruthGPX(filename, self.var_codes.get(1.0)[:, 0:3])
            print("Loaded Ground Truth")
            
        print("Finished Dataset "+filename+"\n")
        
        self.write_to_cache(filename)
        self.ground_truth.write_to_cache(filename)
        
        
    def read_from_cache(self, filename):
        self.acc_with_grav_ERC = self.read_var_from_cache(filename, "acc")
        self.acc_ERC = self.read_var_from_cache(filename, "lin-acc")
        self.gyro = self.read_var_from_cache(filename, "gyro")
        self.mag = self.read_var_from_cache(filename, "mag")
        self.gps = self.read_var_from_cache(filename, "dis")
        self.gps_latlng = self.read_var_from_cache(filename, "latlng")
        print("READ DATA FROM CACHE")
        
    def read_var_from_cache(self, filename, var_name):
        f=open('Data/streams/cache/'+var_name+'/'+filename+'.csv',"r")
        var=[]
        for line in f:
            split = line.split(',')
            var.append([float(split[0]), float(split[1]), float(split[2]), float(split[3])])
        f.close()
        return np.asmatrix(var)
    
    def write_to_cache(self, filename):
        self.write_var_to_cache(filename, "acc", self.acc_with_grav_ERC)
        self.write_var_to_cache(filename, "lin-acc", self.acc_ERC)
        self.write_var_to_cache(filename, "gyro", self.gyro)
        self.write_var_to_cache(filename, "mag", self.mag)
        self.write_var_to_cache(filename, "dis", self.gps)
        self.write_var_to_cache(filename, "latlng", self.gps_latlng)
        print("WRITTEN DATA TO CACHE")
    
    def write_var_to_cache(self, filename, var_name, data):
        f=open('Data/streams/cache/'+var_name+'/'+filename+'.csv',"w+")
        for entry in data:
            line = str(entry[0, 0])+','+str(entry[0,1])+','+str(entry[0, 2])+','+str(entry[0, 3])+'\n'
            f.write(line)
        f.close() 
        
        
        
    def invert_acceleration(self):
        self.acc_DRC[:, 1:3] *= -1
        self.acc_with_grav[:, 1:3] *= -1
    
    def process_csv_line(self, start_time, line):
        i = 1
        while(i < len(line)):
            if(float(line[i]) in self.var_codes):
                self.var_codes[float(line[i])].append([float(line[0])-start_time, 
                                                       float(line[i+1]), float(line[i+2]), float(line[i+3])])
            if(float(line[i]) == 8.0):
                i+=2
            else :
                i+=4
                
    def integrate_variable(self, var):
        return np.concatenate((self.acc_DRC[:,0], integrate.cumtrapz(var[:,1:4], initial=0, axis=0)), axis=1)
    
    def integrate_variables(self):
        self.vel_DRC = self.integrate_variable(self.acc_DRC)
        self.dis_DRC = self.integrate_variable(self.vel_DRC)
        
        self.vel_ERC = self.integrate_variable(self.acc_ERC)
        self.dis_ERC = self.integrate_variable(self.vel_ERC)
        
    def rotate_acceleration(self, rot_vectors, acc_vectors):
        acc_ERC = acc_vectors[:, 0]
        acc_vectors = acc_vectors[:, 1:4].T
        acc_ERC_list=[]
        for i in range(len(rot_vectors[:, 0])):
            rot_matrix_inv = self.get_rotation_matrix(rot_vectors[i, 1:4]) #Orthogonal so transpose is inverse
            acc_ERC_list.append(np.matmul(rot_matrix_inv, acc_vectors[:, i]))        

        return np.concatenate((acc_ERC, np.concatenate(acc_ERC_list, axis=1).T), axis=1)
    
    def get_rotation_matrix(self, rot_vec):
        qx = rot_vec[0, 0]
        qy = rot_vec[0, 1]
        qz = rot_vec[0, 2]
        qw = 1 - qx**2 - qy**2 - qz**2
        rot_matrix = [[1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw]]
        rot_matrix.append([2*qx*qy+2*qz*qw,  1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw])
        rot_matrix.append([2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2])
        return rot_matrix
    
    def swap_xy(self, gps):
        swapped = []
        for i in range(0, gps.shape[0]-1):
            swapped.append(np.asmatrix([
                                gps[i, 0],
                                gps[i, 2],
                                gps[i, 1],
                                gps[i, 3]]))
        return np.concatenate(swapped)
    
    def convert_longlat_to_dis(self, gps):
        dis_list = [[gps[0, 0], 0.0, 0.0, gps[0, 3]]]
        start_gps = gps[0]
        for i in range(1, gps.shape[0]-1):
            dis_list.append([gps[i, 0],
                            self.get_arc_len(start_gps[0, 1], gps[i, 1]),
                            self.get_arc_len(start_gps[0, 2], gps[i, 2]),
                            gps[i, 3]])
        return np.asmatrix(dis_list)
    
    def get_arc_len(self, deg1, deg2):
        delta_theta = deg2 - deg1
        delta_theta = ((delta_theta+180)%360)-180
        earth_R = 6378100
        return 2.0*math.pi*earth_R*delta_theta/360.0
    
    def init_kalman(self, xks, reverse=False):
        latlng_list= []
        earth_R = 6378100
        s_lat = self.gps_latlng[0, 2]
        s_lng = self.gps_latlng[0, 1]
        
        lngs = 360.0*xks[:, 0]/(2.0*math.pi*earth_R)+s_lng
        lats = 360.0*xks[:, 1]/(2.0*math.pi*earth_R)+s_lat
        kal_latlng = np.concatenate((self.gps[:, 0], lngs, lats, self.gps[:, 3]), axis=1)
        if(reverse):
            self.kal_latlng_reverse = kal_latlng[::-1]
            self.kal_dis_reverse = np.concatenate((self.gps[:, 0], xks[:, 0:2], self.gps[:, 3]), axis=1)[::-1]
            
        else:
            self.kal_latlng = kal_latlng
            self.kal_dis = np.concatenate((self.gps[:, 0], xks[:, 0:2], self.gps[:, 3]), axis=1)
            
    
    def plot(self):
        # Graph the variables.
        plt.figure(figsize=(9, 20))
        ax=plt.subplot(521)
        plt.plot(self.acc_DRC[:, 0], self.acc_DRC[:, 1], 'r-', lw=1, label='X')
        plt.plot(self.acc_DRC[:, 0], self.acc_DRC[:, 2], 'b-', lw=1, label='Y')
        plt.plot(self.acc_DRC[:, 0], self.acc_DRC[:, 3], 'g-', lw=1, label='Z')
        plt.title("Acceleration - DRC")
        ax.legend()

        ax=plt.subplot(522)
        plt.plot(self.acc_ERC[:, 0], self.acc_ERC[:, 1], 'r-', lw=1, label='X')
        plt.plot(self.acc_ERC[:, 0], self.acc_ERC[:, 2], 'b-', lw=1, label='Y')
        plt.plot(self.acc_ERC[:, 0], self.acc_ERC[:, 3], 'g-', lw=1, label='Z')
        plt.title("Acceleration - ERC")
        ax.legend()

        ax=plt.subplot(523)
        plt.plot(self.vel_DRC[:, 0], self.vel_DRC[:, 1], 'r-', lw=1, label='X')
        plt.plot(self.vel_DRC[:, 0], self.vel_DRC[:, 2], 'b-', lw=1, label='Y')
        plt.plot(self.vel_DRC[:, 0], self.vel_DRC[:, 3], 'g-', lw=1, label='Z')
        plt.title("Velocity - DRC")
        ax.legend()

        ax=plt.subplot(524)
        plt.plot(self.vel_ERC[:, 0], self.vel_ERC[:, 1], 'r-', lw=1, label='X')
        plt.plot(self.vel_ERC[:, 0], self.vel_ERC[:, 2], 'b-', lw=1, label='Y')
        plt.plot(self.vel_ERC[:, 0], self.vel_ERC[:, 3], 'g-', lw=1, label='Z')
        plt.title("Velocity - ERC")
        ax.legend()

        ax=plt.subplot(525)
        plt.plot(self.dis_DRC[:, 0], self.dis_DRC[:, 1], 'r-', lw=1, label='X')
        plt.plot(self.dis_DRC[:, 0], self.dis_DRC[:, 2], 'b-', lw=1, label='Y')
        plt.plot(self.dis_DRC[:, 0], self.dis_DRC[:, 3], 'g-', lw=1, label='Z')
        plt.title("Displacement - DRC")
        ax.legend()

        ax=plt.subplot(526)
        plt.plot(self.dis_ERC[:, 0], self.dis_ERC[:, 1], 'r-', lw=1, label='X')
        plt.plot(self.dis_ERC[:, 0], self.dis_ERC[:, 2], 'b-', lw=1, label='Y')
        plt.plot(self.dis_ERC[:, 0], self.dis_ERC[:, 3], 'g-', lw=1, label='Z')
        plt.title("Displacement - ERC")
        ax.legend()

        ax=plt.subplot(527)
        plt.plot(self.gyro[:, 0], self.gyro[:, 1], 'r-', lw=1, label='X')
        plt.plot(self.gyro[:, 0], self.gyro[:, 2], 'b-', lw=1, label='Y')
        plt.plot(self.gyro[:, 0], self.gyro[:, 3], 'g-', lw=1, label='Z')
        plt.title("Gyroscope")
        ax.legend()

        ax=plt.subplot(528)
        plt.plot(self.mag[:, 0], self.mag[:, 1], 'r-', lw=1, label='X')
        plt.plot(self.mag[:, 0], self.mag[:, 2], 'b-', lw=1, label='Y')
        plt.plot(self.mag[:, 0], self.mag[:, 3], 'g-', lw=1, label='Z')
        plt.title("Magnetometer")
        ax.legend()

        ax=plt.subplot(529)
        plt.plot(self.gps_latlng[:, 1], self.gps_latlng[:, 2], 'r-', lw=1, label='X')
        plt.title("GPS Lat Lng") ##Proof that values are reversed, should be -9.8 its 9.8
        ax.legend()

        ax=plt.subplot(5, 2, 10)
        plt.plot(self.gps[:, 1], self.gps[:, 2], 'r-', lw=1, label='X')
        plt.title("GPS Displacements") ##Proof that values are reversed, should be -9.8 its 9.8
        ax.legend()
    

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
        
        ##Create all sequences
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

        ##Print Graphs of X against Y
        plt.figure(figsize=(8,5))
        plt.plot(ground_truth[:, 0], ground_truth[:, 1], label='ground truth', color = 'g')
        plt.plot(predicted_output[:, 0], predicted_output[:, 1], label='seen training data', color = 'b')
        plt.plot(orig_gps[:, 0], orig_gps[:,1], label='original gps', color = 'r')
        plt.legend()
        plt.xlabel('Position X (Metres)')
        plt.ylabel('Position Y (Metres)')
        plt.savefig(str(filenames[count])+".pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.show()

        print("Accuracy of GPS: ", measure_accuracy(ground_truth, orig_gps))
        print("Accuracy of RNN: ", measure_accuracy(ground_truth, predicted_output))
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


# data = Data_Stream('cyc-tuto2', load_truth=True, higher_freq=False, no_cache=True)
# plt.plot(data.gps[:, 1], data.gps[:, 2])
# plt.show()
# data = Data_Stream('cyc-tuto-rev2', load_truth=True, higher_freq=False, no_cache=True)
# plt.plot(data.gps[:, 1], data.gps[:, 2])
# plt.show()
# data = Data_Stream('run-john0', load_truth=True, higher_freq=False, no_cache=True)
# plt.plot(data.gps[:, 1], data.gps[:, 2])
# plt.show()

###########################################################
################## CHOOSE TRAINING AND TESTING FILES
###########################################################
print("\n###########################################################")
print("######### Loading Data")
print("###########################################################\n")
training_files = ['uni', 'tutoring0', 'uni2', 'dog0', 'uni3', 'train0']
testing_files  = ['mb0', 'uni1']
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
seq_offset      = int(seq_len/10)
warmup_steps    = 5
batch_size      = 512
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
val_batch_size = int(0.45*x_test_seqs.shape[0])
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
model.add(GRU(units=64, return_sequences=True, input_shape=(None, x_dim,)))
model.add(GRU(units=64, return_sequences=True, input_shape=(None, x_dim,)))
model.add(GRU(units=64, return_sequences=True, input_shape=(None, x_dim,)))
model.add(GRU(units=64, return_sequences=True, input_shape=(None, x_dim,)))
model.add(GRU(units=64, return_sequences=True, input_shape=(None, x_dim,)))
model.add(Dense(y_dim, activation='linear'))
optimizer = Adam(lr=1e-3)
model.compile(loss=custom_loss, optimizer=optimizer)
model.summary()


###########################################################
################## CREATE CALLBACK and TRAIN MODEL
###########################################################
# path_checkpoint = 'RNN.checkpoint'
# callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
# callback_tensorboard = TensorBoard(log_dir='./23_logs/', histogram_freq=0, write_graph=False)
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-9, patience=2, verbose=1)      
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
plot_dataset(training_dataset, seq_len=seq_len, filenames=training_files)

tf.reset_default_graph()
sess.close()