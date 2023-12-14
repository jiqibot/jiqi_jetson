#!/usr/bin/env python3
import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TwistWithCovariance
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry



class MultiSensorFusion():

    def __init__(self, state_dim, control_dim, measurement_dim):

        self.prev_pose=np.zeros(3)
        self.curr_pose=None

        # Set time step to 0.1 seconds
        self.dt = 0.1
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.measurement_dim = measurement_dim
        
        # Set starting values for state transition matrix
        self.F_ = np.identity(self.state_dim)
        self.F_[0,2] = self.dt
        self.F_[1,3] = self.dt

        # Set starting values for control input matrix
        self.B_ = np.zeros(shape=(self.state_dim, self.control_dim))
        self.B_[2,0] = self.dt
        self.B_[3,1] = self.dt

        # Set starting values for measurement matrix
        self.H_ = np.zeros(shape=(self.measurement_dim, self.state_dim))
        self.H_[0,0] = 1
        self.H_[1,1] = 1
        self.H_[2,2] = 1
        self.H_[3,3] = 1

        # Set starting values for state covariance matrix
        self.P_ = np.zeros(shape=(self.state_dim, self.state_dim))
        self.P_[0,0] = 1
        self.P_[1,1] = 1
        self.P_[2,2] = 1
        self.P_[3,3] = 1

        # Set starting values for process noise covariance matrix
        self.Q_ = np.zeros(shape=(self.state_dim, self.state_dim))
        self.Q_[0,0] = 0.1
        self.Q_[1,1] = 0.1
        self.Q_[2,2] = 0.1
        self.Q_[3,3] = 0.1

        # Set starting values for measurement noise covariance matrix
        self.R_ = np.zeros(shape=(self.measurement_dim, self.measurement_dim))
        self.R_[0,0] = 1
        self.R_[1,1] = 1
        self.R_[2,2] = 1
        self.R_[3,3] = 1

        # Set initial state estimate
        self.state = np.zeros(self.state_dim)
        self.zed_pose = None
        self.zed_ang_vel = None
        self.zed_orietation = None
        self.lidar_pose = None
        self.gps_pose = None
        rospy.Subscriber("/zed_odom", Odometry, self.zed_pose_callback)
        rospy.Subscriber("/lidar_pose", PoseWithCovarianceStamped, self.lidar_pose_callback)
        rospy.Subscriber("/gps_pose", NavSatFix, self.gps_pose_callback)

        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=10) #move_base subscibes to the odom topic

        
    def zed_pose_callback(self, msg):
        self.zed_pose = np.array([[msg.pose.pose.position.x], [msg.pose.pose.position.y], [msg.pose.pose.position.z],[0]])
        self.zed_orietation = np.array([[msg.pose.pose.orientation.x],[msg.pose.pose.orientation.y],[msg.pose.pose.orientation.z]])
        self.zed_ang_vel = np.array([[msg.twist.twist.angular.x],[msg.twist.twist.angular.y],[msg.twist.twist.angular.z]])
        
    def lidar_pose_callback(self, msg):
        self.lidar_pose = np.array([[msg.pose.pose.position.x], [msg.pose.pose.position.y], [msg.pose.pose.position.z],[0]])
        
    def gps_pose_callback(self, msg):
        self.gps_pose = np.array([[msg.latitude], [msg.longitude], [msg.altitude]])
    
    def calc_velocity(self):
        vel_x = (self.curr_pose[0] - self.prev_pose[0])/self.dt
        vel_y = (self.curr_pose[1] - self.prev_pose[1])/self.dt
        vel_z = (self.curr_pose[2] - self.prev_pose[2])/self.dt
        return np.array([vel_x,vel_y,vel_z])

        
    def update(self):
        if self.zed_pose is not None and self.lidar_pose is not None:

            self.state = np.dot(self.F_, self.state_dim) + np.random.multivariate_normal(np.zeros(4), self.Q_) #np.dot(self.transition_matrix, self.state) + np.random.multivariate_normal(np.zeros(3), self.process_covariance)
            measurement = np.array([self.zed_pose, self.lidar_pose])
            average_measurement = np.mean(measurement, axis=0)
            self.covariance = np.dot(self.F_, np.dot(self.P_, self.F_.T)) + self.Q_ #np.dot(self.transition_matrix, np.dot(self.covariance, self.transition_matrix.T)) + self.process_covariance
            innovation = average_measurement - np.dot(self.H_, self.state) #np.dot(self.measurement_matrix, self.state)
            innovation_covariance = np.dot(self.R_, np.dot(self.P_, self.H_.T)) + self.R_ #np.dot(self.measurement_matrix, np.dot(self.covariance, self.measurement_matrix.T)) + self.measurement_covariance
            kalman_gain = np.dot(self.P_, np.dot(self.H_.T, np.linalg.inv(innovation_covariance))) #np.dot(self.covariance, np.dot(self.measurement_matrix.T, np.linalg.inv(innovation_covariance)))
            self.state = self.state + np.dot(kalman_gain, innovation)
            self.covariance = np.dot((np.identity(4) - np.dot(kalman_gain, self.H_)), self.P_)

            self.curr_pose=np.array([self.state[0][0],self.state[0][1],0])

            estimated_pose = Odometry()
            estimated_pose.header.stamp = rospy.Time.now()
            estimated_pose.header.frame_id = "odom"
            estimated_pose.pose.pose.position.x = self.curr_pose[0]
            estimated_pose.pose.pose.position.y = self.curr_pose[1]
            estimated_pose.pose.pose.position.z = 0
            estimated_pose.pose.pose.orientation.x = self.zed_orietation[0]
            estimated_pose.pose.pose.orientation.y = self.zed_orietation[1]
            estimated_pose.pose.pose.orientation.z = self.zed_orietation[2]

            self.curr_pose=np.array([self.state[0][0],self.state[0][1],0])

            estimated_velocity = self.calc_velocity()
            estimated_pose.twist.twist.linear.x= estimated_velocity[0]
            estimated_pose.twist.twist.linear.y= estimated_velocity[1]
            estimated_pose.twist.twist.linear.z= estimated_velocity[2]
            estimated_pose.twist.twist.angular.x=self.zed_ang_vel[0]
            estimated_pose.twist.twist.angular.y=self.zed_ang_vel[1]
            estimated_pose.twist.twist.angular.z=self.zed_ang_vel[2]
            self.odom_pub.publish(estimated_pose)

            self.prev_pose = self.curr_pose


if __name__ == '__main__':
    rospy.init_node('multi_sensor_fusion')
    fusion = MultiSensorFusion(4,2,4)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        fusion.update()
        rate.sleep()
