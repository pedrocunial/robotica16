#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from math import fabs
from std_msgs.msg import String
# import roslib; roslib.load_manifest('mini_max_tutorials')


def notificacao(data):
    """
        Codigo de notificacao executado sempre que chega uma leitura da odometria

        Esta leitura chega na variavel data e e'  um objeto do tipo odometria
    """

def talker():
	pub = rospy.Publisher('chatter', String, queue_size=10)
	rospy.init_node('talker', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	while not rospy.is_shutdown():
		hello_str = "hello world %s" % rospy.get_time()
		rospy.loginfo(hello_str)
		pub.publish(hello_str)
		rate.sleep()

def controle():
	# coisas do miranda
	rospy.init_node("exemplo_python")
	velocidade_objetivo = None;
	pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)	
	rospy.Subscriber('/odom', Odometry, notificacao)
	# Initial movement.
	# velocidade_objetivo = 
	pub.publish(velocidade_objetivo)
	rospy.spin() # Faz um loop infinito para o ROS nao retornar

def move_exemplo():
	x_speed = 0.1 # 0.1 m/s

	# first thing, init a node!
	rospy.init_node('move')

	# publish to cmd_vel
	p = rospy.Publisher('cmd_vel', Twist)

	# create a twist message, fill in the details
	twist = Twist()
	twist.linear.x = x_speed;                   # our forward speed
	twist.linear.y = 0; twist.linear.z = 0;     # we can't use these!        
	twist.angular.x = 0; twist.angular.y = 0;   #          or these!
	twist.angular.z = 0;                        # no rotation

	# announce move, and publish the message
	rospy.loginfo("About to be moving forward!")
	for i in range(30):
		p.publish(twist)
		rospy.sleep(0.1) # 30*0.1 = 3.0

	# create a new message
	twist = Twist()

	# note: everything defaults to 0 in twist, if we don't fill it in, we stop!
	rospy.loginfo("Stopping!")
	p.publish(twist)

class square:
	""" This example is in the form of a class. """
	
	def __init__(self):
		""" This is the constructor of our class. """
		# register this function to be called on shutdown
		rospy.on_shutdown(self.cleanup)
		rospy.init_node('square')
		# publish to cmd_vel
		self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		# give our node/publisher a bit of time to connect
		rospy.sleep(1)
		
		# use a rate to make sure the bot keeps moving
		r = rospy.Rate(5.0)
		
		# go forever!
		while not rospy.is_shutdown():
			# create a Twist message, fill it in to drive forward
			twist = Twist()
			twist.linear.x = 0.5
			for i in range(10):         # 10*5hz = 2sec
				self.pub.publish(twist)
				r.sleep()
				# create a twist message, fill it in to turn
			twist = Twist()
			twist.angular.z = 1.57/2    # 45 deg/s * 2sec = 90 degrees
			for i in range(10):         # 10*5hz = 2sec
				self.pub.publish(twist)
				r.sleep()

	def cleanup(self):
		# stop the robot!
		twist = Twist()
		self.pub.publish(twist)


if __name__ == '__main__':
    try:
        # controle()
        # talker()
		square()
    except rospy.ROSInterruptException:
        pass
