#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import sys, select, termios, tty, time

settings = termios.tcgetattr(sys.stdin)
msg = """How to Control JIQI!
--------------------------------------
Moving Around:
        w
    a   s   d
        
w/s - Increase/Decrease Linear Velocity
a/d - Increase/Decrease Angular Velocity

Space Key - Force Stop
q to Quit\n\n
"""


def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _= select.select([sys.stdin],[],[],0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key= ' '
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def vels(speed, turn):
    return "Currently: \tspeed %s \tturn %s" %(speed, turn)

def takeover_callback(data):
    if msg.data =="start":
        takeover=True
    elif msg.data=="stop":
        takeover=False

def teleop():
    global takeover
    takeover = False
    manual_override = False
    takeover_message_printed = False

    rospy.init_node("keyboard_teleop")
    pub_cmd = rospy.Publisher("cmd_vel", Twist, queue_size=5)
    pub_autonomous = rospy.Publisher("Autonmous_Status", String, queue_size=5)

    rospy.Subscriber("takeover", String, takeover_callback)
    rate=rospy.Rate(100)
    twist = Twist()

    while not rospy.is_shutdown():
        key = getKey()
        if key =='t':
            manual_override=True
            pub_autonomous.publish("auton stop") #autonomous driving stopped
            print("Manual Operation Activated")
            print(msg)
            takeover_message_printed = True

        if takeover or manual_override:
            if not takeover_message_printed:
                print("JIQI is in a sticky, manual operation required")
                print(msg)
                takeover_message_printed= True

            key = getKey()
            if key =='w':
                twist.linear.x=1.0
                twist.angular.z=0.0

            if key =='s':
                twist.linear.x=-1.0
                twist.angular.z=0.0   

            if key =='a':
                twist.linear.x=0.0
                twist.angular.z=1.0 

            if key =='d':
                twist.linear.x=0.0
                twist.angular.z=-1.0 

            if key ==' ':
                twist.linear.x=0.0
                twist.angular.z=0.0 
            
            if key =='q':
                twist.linear.x=0.0
                twist.angular.z=0.0
                print("Are you sure you want to exit manual operation? Type 'yes' to confirm")
                confirm=input()
                if confirm=='yes':
                    pub_autonomous.publish("auton resume") #the autonomous drving node will then set takeover to stop/False resuming autonomous navigation
                    manual_override=False
                    print("\n\nReturning to Autonomous Navigation in...")
                    time.sleep(1)
                    print("3...")
                    time.sleep(1)
                    print("2...")
                    time.sleep(1)
                    print("1...")
                    time.sleep(1)
                    print("Autonomous Navigation Activated")
                    print("\nPress t to take manual control over JIQI\n")
                    

            else:
                twist.linear.x=0.0
                twist.angular.z=0.0 

            pub_cmd.publish(twist)
        rate.sleep()

if __name__ =="__main__":

    print("\nPress t to take manual control over JIQI\n")

    try:
        teleop()
    except rospy.ROSInterruptException:
        pass
