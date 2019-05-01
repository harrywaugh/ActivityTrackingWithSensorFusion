import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import interpolate
import math
from scipy import signal
import copy
import os

class GroundTruthGPX:
    latlngs = np.asmatrix([0.0, 0.0])
    def __init__(self, filename, gps, load_cache=False):
#         if(load_cache):
#             self.read_from_cache(filename)
#             return
        
#         directory = 'Data/ground_truth/' + filename + '.gpx'
#         ##Parse GT file
#         f = open(directory, 'r')
#         latlngs = []
#         #Read Data from CSV
#         for line in f:
#             tags = [tag[6:] for tag in line.split("><") if tag[0]=='r']
#             for tag in tags:
#                 split_tag = tag.split('"')
#                 if(len(split_tag)>1):
#                     latlngs.append([float(split_tag[3]), float(split_tag[1])])
#         self.latlngs= np.asmatrix(latlngs)
#         self.dis = self.convert_longlat_to_dis(self.latlngs)

#         ## Add  linear time to GPS, interpolated from start time to end time
        
#         time_vector = []
#         indexes = []
#         for position in self.dis:
#             distances = gps[:, 1:3] - position
#             magnitudes = np.linalg.norm(distances, ord = 2, axis = 1)
#             indexes.append(np.argmin(magnitudes))
#             time_vector.append([gps[indexes[-1], 0]])

#         self.dis = np.concatenate((np.asmatrix(time_vector), self.dis), axis=1)
#         self.dis[-1] = gps[-1, 0:3]  
#         indexes[-1] = len(gps)
        
#         indexes.sort()
#         previous_index = indexes[0]
#         interpolated_positions = []
#         for i in range(1, len(indexes)):
#             n = indexes[i] - previous_index
            
#             if (n > 1):
#                 a = self.dis[i-1]            
#                 b = self.dis[i]
#                 interpolated_positions.extend(self.interpolate(a, b, n))
#             elif(n == 1):
#                 interpolated_positions.append(self.dis[i])
#             previous_index = indexes[i]
        
#         self.dis = np.squeeze(np.asarray(interpolated_positions))
#         latlngs = np.asmatrix([0.0, 0.0])

        
#         print(len(gps))
#         print(len(self.dis))
        if(load_cache):
            self.read_from_cache(filename)
            return
        
        directory = 'Data/ground_truth/' + filename + '.gpx'
        ##Parse GT file
        f = open(directory, 'r')
        latlngs = []

        for line in f:
            tags = [tag[6:] for tag in line.split("><") if tag[0]=='r']
            for tag in tags:
                split_tag = tag.split('"')
                if(len(split_tag)>1):
                    latlngs.append([float(split_tag[3]), float(split_tag[1])])
        self.latlngs= np.asmatrix(latlngs)
        
        lin_time = np.asmatrix(np.linspace(0, gps[-1, 0], len(self.latlngs))).T
        self.latlngs = np.concatenate((lin_time, self.latlngs), axis=1)
        irreg_var = self.latlngs
        reg_varX = np.asmatrix(np.interp(gps[:, 0], np.ravel(irreg_var[:,0]), np.ravel(irreg_var[:,1])))
        reg_varY = np.asmatrix(np.interp(gps[:, 0], np.ravel(irreg_var[:,0]), np.ravel(irreg_var[:,2])))
        self.latlngs = np.concatenate((reg_varX, reg_varY), axis=1)
        
        
        
        self.dis = self.convert_longlat_to_dis(self.latlngs)
        self.dis = np.concatenate((gps[:, 0], self.dis), axis=1)
        
        self.correct_ground_truth(self.dis[:, 1:3], gps[:,1:3])
        print(len(self.dis))
        self.dis = np.concatenate((gps[:, 0], self.dis), axis=1)
        
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
#         self.latlngs = self.read_var_from_cache(filename, "latlng")
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
#         self.write_var_to_cache(filename, "latlng", self.latlngs)
        print("WRITTEN GROUND TRUTH DATA TO CACHE")
    
    def write_var_to_cache(self, filename, var_name, data):
        f=open('Data/ground_truth/cache/'+var_name+'/'+filename+'.csv',"w+")
        for entry in data:
            line = str(entry[0, 0])+','+str(entry[0, 1])+','+str(entry[0, 2])+'\n'            
#             line = str(entry[0])+','+str(entry[1])+','+str(entry[2])+'\n'

            f.write(line)
        f.close()
        
    