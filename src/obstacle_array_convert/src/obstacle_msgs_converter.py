#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import PolygonStamped, Point32

import numpy as np


def callback(marker_array):
    pub = rospy.Publisher('/test_optim_node/obstacles', ObstacleArrayMsg, queue_size=1)
    obstacle_msg = ObstacleArrayMsg() 
    obstacle_msg.header.stamp = rospy.Time.now()
    obstacle_msg.header.frame_id = "odom"
    i =0
    markers = marker_array.markers
    for marker in markers:
        centroid = np.array([marker.pose.position.x, marker.pose.position.y])
        scale = np.array([marker.scale.x, marker.scale.y])

        obstacle_msg.obstacles.append(ObstacleMsg())
        obstacle_msg.obstacles[i].id = 2
        v1 = Point32()
        v1.x = centroid[0]-(scale[0]/2)
        v1.y = centroid[1]-(scale[1]/2)
        v2 = Point32()
        v2.x = centroid[0]+(scale[0]/2)
        v2.y = centroid[1]+(scale[1]/2)
        v3 = Point32()
        v3.x = centroid[0]+(scale[0]/2)
        v3.y = centroid[1]-(scale[1]/2)
        v4 = Point32()
        v4.x = centroid[0]-(scale[0]/2)
        v4.y = centroid[1]+(scale[1]/2)
        obstacle_msg.obstacles[i].polygon.points = [v1, v2, v3, v4]
        i+=1

    pub.publish(obstacle_msg)

def markerArrayConverter():
    rospy.init_node('obstacle_msgs_converter', anonymous=True)
    rospy.Subscriber("/object_bounding_boxes", MarkerArray, callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        markerArrayConverter()
    except rospy.ROSInterruptException:
        pass