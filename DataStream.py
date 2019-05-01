import numpy as np
import matplotlib
from GroundTruth import GroundTruthGPX
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import interpolate
import math
from scipy import signal
import copy
import os

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
        frequency = 1/time_period
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
    
def __main__():
	print("Loaded")