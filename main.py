import numpy as np
from utils import *
import matplotlib.pyplot as plt
from scipy.linalg import expm,inv


filename = "./data/0042.npz"
t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)


	# (a) IMU Localization via EKF Prediction
tt = t.shape[1] -1
noise_pose = 1

sai_hat = np.zeros((4,4))
sai_sharp = np.zeros((6,6))
W = noise_pose * np.identity(6)
car_mu = np.zeros([4, 4, tt])
out_mu = np.zeros([4, 4, tt])
sigma = np.zeros([6, 6, tt])

car_mu[:,:,0] = np.identity(4)
out_mu[:,:,0] = np.identity(4)
sigma[:,:,0] = np.identity(6)
for i in range(tt-1):
    w = rotational_velocity[:,i]
    v = linear_velocity[:,i]
    ##### sai_hat #####
    sai_hat[0:3, 0:3]= hat(w)
    sai_hat[0:3, 3] = v
    ##### sai_sharp ######
    sai_sharp[0:3, 0:3] = hat(w)
    sai_sharp[0:3, 3:6] = hat(v)
    sai_sharp[3:6, 3:6] = hat(w)
    
    tau = t[0, i+1] - t[0,i]
    car_mu[:,:,i+1] = expm(-tau * sai_hat).dot(car_mu[:,:,i])
    sigma[:,:,i+1] = expm(-tau * sai_sharp).dot(sigma[:,:,i]) @ np.transpose(expm(-tau * sai_sharp)) +tau**2*W
    out_mu[:,:,i+1] = inv(car_mu[:,:,i+1])

fig,ax = visualize_trajectory_2d( out_mu, path_name=filename, show_ori=False)




	# (b) Landmark Mapping via EKF Update
fsu = K[0,0]
fsv = K[1,1]
cu = K[0,2]
cv = K[1,2]

points = np.zeros([4, features.shape[1]])
for i in range(tt):
    fea = features[:,:,i]
    index = get_num(fea)
    fea = fea[:, index]
    
    z = fsu*b/(fea[0,:] - fea[2,:])
    x = (fea[0,:] - cu)/ fsu * z
    y = (fea[1,:] - cv)/ fsv * z
    
    res = np.ones([4,fea.shape[1]])
    res[0,:] = x
    res[1,:] = y
    res[2,:] = z
    m = inv(car_mu[:,:,i]) @ inv(cam_T_imu) @ res
    points[:,index] = m


#######################################
M = np.zeros([4,4])
M[0:3,0:3] = K 
M[2,:] = np.array([K[0,0],K[0,1],K[0,2],-K[0,0]*b])
M[3,:] = M[1,:]
#######################################
D = np.zeros([4,3]) 
D[0:3,0:3] = np.identity(3)

noise = 1   ######### changing noise can lead to change of the result ##########

follow_list = []
point_list = []
all_miu = np.zeros([4,features.shape[1]])
all_cov = np.zeros([3,3,features.shape[1]])
for k in range(tt):
    fea = features[:,:,k]
    index = get_num(fea)
    fea = fea[:,index]
    
    z = fsu*b/(fea[0,:] - fea[2,:])
    x = (fea[0,:] - cu)/ fsu * z
    y = (fea[1,:] - cv)/ fsv * z
    
    res = homo(np.vstack([x,y,z]))
    
    m = inv(car_mu[:,:,k]) @ inv(cam_T_imu) @ res
    
    for i in range(len(index)):
        pin = index[i]
        if pin in follow_list: 
            mu = all_miu[:, pin]
            cov = all_cov[:,:, pin]
            zt = features[:, pin, k]
            V = noise*np.identity(4)
            all_miu[:,pin], all_cov[:,:,pin] = calculate(cam_T_imu, zt, car_mu[:,:,k], mu, cov, V, M, D)
        else:
            all_miu[:,pin], all_cov[:,:,pin] = m[:,i], np.identity(3)
            follow_list.append(pin)
        if k == tt-1 or not check(pin, features[:,:,k+1]):
            point_list.append(all_miu[:,pin])
            follow_list.remove(pin)

update_points = point_list[0].reshape(-1,1)
for i in range(1,len(point_list)):
    update_points = np.concatenate((update_points, point_list[i].reshape(-1,1)),axis = 1)
    
fig , ax = trajectory_and_points(out_mu, points, update_points, path_name=filename)  
	# (c) Visual-Inertial SLAM (Extra Credit)

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)
