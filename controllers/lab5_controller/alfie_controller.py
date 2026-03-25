"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as pyplot

def clamp(value, min_val, max_val): #helper function
    return max(min(value, max_val), min_val)

def draw_threshold_map(threshold_map, display):
    height = display.getHeight()
    width = display.getWidth()

    rows, cols = threshold_map.shape

    cell_w = width // cols
    cell_h = height // rows

    for i in range(rows):
        for j in range(cols):
            if threshold_map[i, j]:
                display.setColor(0xFFFFFF)  # occupied (white)
            else:
                display.setColor(0x000000)  # free (black)

            display.fillRectangle(i * cell_w, j * cell_h, cell_w, cell_h)


MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### potential field for autonomous:
# CONSTANTS
gain_rho0 = 0.8 # distance boundary
gain_nu = 0.02
gain_nu_forward = 3
gain_u = 2.5
gain_v = 0.35

##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

ranger = robot.getDevice('range-finder')
ranger.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

increment_const = 5e-3

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

# map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
mode = 'autonomous' # Part 1.1: manual mode
    
# mode = 'autonomous'

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.zeros(shape=[360,360])
map_tf = np.zeros(shape=[360,360])

save_pressed = False

# Autonomous mode state
prev_error = 0  # for PD derivative term

stuck_timer         = 0   # counts remaining backup steps
stuck_check_counter = 0   # counts steps between position snapshots
STUCK_CHECK_EVERY   = 30  # check position every N steps
BACKUP_STEPS        = 30  # how many steps to back up
last_stuck_x        = 0.0
last_stuck_y        = 0.0

while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################
    
    # attractive forces which are assigned as the initial forces before being summed later
    Fx= gain_nu_forward
    Fy= 0

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]

    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]
        
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y
        # print(f"{i}: {wx}, {wy}, {rho}, {rx}, {ry}")
        
        ################ ^ [End] Do not modify ^ ##################

        # print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
        if wx >= 12:
            wx = 11.999
        if wy >= 12:
            wy = 11.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
            px_curr = 359-abs(int(wx*30))
            py_curr = abs(int(wy*30))
            if px_curr > 359 or py_curr > 359:
                continue;
            elif map[px_curr,py_curr] > (1 - increment_const):
                map[px_curr,py_curr] = 1
                g = 1
                # display.setColor(int(0xEEA24A))
                # display.fillOval(360-abs(int(wx*30)),abs(int(wy*30)), 8,8)
                
            else:
                map[px_curr,py_curr] += increment_const;
                g = map[px_curr,py_curr]
            cmap = (g*256**2 + g*256 + g)*255
            display.setColor(int(cmap))
            map_tf = map>0.5;
            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            # display.setColor(int(0X0000FF))
            display.drawPixel(360-abs(int(wx*30)),abs(int(wy*30)))
        #calculation of repelling forces
        #      note: theyre negative because we want the repelling forces to be acting opposite of where they are
        #      note: force formula aquired from Pranav Bhounsule: https://www.youtube.com/watch?v=bgeI_wMT67Q
        if rho > gain_rho0:
            f_repel = 0
        else:
            f_repel = gain_nu*(1/rho - 1/gain_rho0)/rho**2
                
           
        fx_repel = -math.cos(alpha)*f_repel
        fy_repel = math.sin(alpha)*f_repel
        # sum to the net forces
        Fx += fx_repel
        Fy += fy_repel
        #NATE: display the robots sensor with account for size of robot
        
            
        print("\n")
        # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-abs(int(pose_x*30)), abs(int(pose_y*30)))
    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        ranges = np.array(lidar_sensor_readings)
        ranges[ranges == np.inf] = LIDAR_SENSOR_MAX_RANGE
        right_side = np.min(ranges[490: 501]) 
        front = np.min(ranges[320 : 345]) 
        print(f"front:{front:.2f} right:{right_side:.2f} vL:{vL:.2f} vR:{vR:.2f}")
        #Perform teleoperation and obtain a map of the entire environment and store the map as an npy file
        vL = 0
        vR = 0
        # for i, rho in enumerate(lidar_sensor_readings):
            # print(f"i:{i}, rho:{rho}")
        key = keyboard.getKey()
        # print(key)
        if(key == keyboard.UP):
            vL = 1 * MAX_SPEED
            vR = 1 * MAX_SPEED
        elif(key == keyboard.DOWN):
            vL = -1 * MAX_SPEED
            vR = -1 * MAX_SPEED
        elif(key == keyboard.LEFT):
            vL = -1 * MAX_SPEED
            vR = 1 * MAX_SPEED
        elif(key == keyboard.RIGHT):
            vL = 1 * MAX_SPEED
            vR = -1 * MAX_SPEED
        elif(key == ord('s') or key == ord('S')):
            if save_pressed == False:
                threshold_map = map > 0.5
                
                # np.save("map.npy", map)
                print("saved map")
                width = display.getWidth()
                height = display.getHeight()
                
                
               
                display.setColor(0x00000)
                
               
                display.fillRectangle(0, 0, width, height)
                
                
                draw_threshold_map(threshold_map, display)
                
                # break
                save_pressed = True
        elif(key == ord('a') or key == ord('A')):
            if mode_pressed == False:
                mode = 'autonomous'
                mode_pressed = True                
        else:
            save_pressed = False
            mode_pressed = False
    # issues ive noticed, when spinnissng fast the robot seems to notice the boundary of its vision as a wall
            #also ive noticed that certain objects it can see through and certain objects it cant
            #also that things with legs like chairs and tables are seen as just the legs so may be misinterpreted as passaable
            #there is also random noise but it just comes in dots and not in big packets outside of the spinning issue
            

    
    elif mode == 'autonomous':
        
        key = keyboard.getKey()
        if ( key == ord('a') or key == ord('A') ) and not mode_pressed:
            mode = 'manual'
            mode_pressed = True
        else:
            mode_pressed = False
        
        # Inside the autonomous elif:
        ranges = np.array(lidar_sensor_readings)
        ranges[ranges == np.inf] = LIDAR_SENSOR_MAX_RANGE

        # Change 1: use median instead of min to reduce noise spikes
        right_side = np.median(ranges[490:501])
        front      = np.median(ranges[320:345])

        desired_dist = 0.6
        base_speed   = 0.3 * MAX_SPEED

        # Priority 1: right wall in range — PD controller steers into corners naturally
        stuck_check_counter += 1
        if stuck_check_counter >= STUCK_CHECK_EVERY:
            dist_moved = math.sqrt((pose_x - last_stuck_x)**2 + (pose_y - last_stuck_y)**2)
            if dist_moved < 0.05:  # less than 5 cm => stuck
                stuck_timer = BACKUP_STEPS
                print("Stuck detected — backing up")
            last_stuck_x = pose_x
            last_stuck_y = pose_y
            stuck_check_counter = 0

        # Priority 0: back up if stuck
        if stuck_timer > 0:
            vL = -0.45 * MAX_SPEED
            vR = -0.45 * MAX_SPEED
            stuck_timer -= 1
            print(f"Backing up — steps left: {stuck_timer}")

        # Priority 1: front obstacle
        elif front < 0.9:
            vL = -0.4 * MAX_SPEED
            vR =  0.4 * MAX_SPEED
            print("Avoiding front wall")
            
        elif right_side < 0.85:
            error   = desired_dist - right_side
            d_error = error - prev_error
            turn    = 0.70 * error + 0.30 * d_error
            prev_error = error
            vL = clamp(base_speed + turn, -MAX_SPEED, MAX_SPEED)
            vR = clamp(base_speed - turn, -MAX_SPEED, MAX_SPEED)
            print(f"Wall following — error:{error:.2f} d_error:{d_error:.2f} turn:{turn:.2f}")

        else:
            vL = MAX_SPEED * 0.7
            vR = MAX_SPEED * 0.15
            print("Searching for right wall")

        print(f"front:{front:.2f} right:{right_side:.2f} vL:{vL:.2f} vR:{vR:.2f}")
            
        
        
        
        
        # print(f"bl:{rho_bl_mean} l:{rho_l_mean} fl:{rho_fl_mean} front:{rho_front_mean} fr:{rho_fr_mean} r:{rho_r_mean} br:{rho_br_meaan}a")
        # #instead of doing if else case: sum different values by weight until i turn the right direction 
        

        # if rho_front_mean > 4:
            # print("rho_front_mean case")
            # vL = -0.5*MAX_SPEED
            # vR = 0.5*MAX_SPEED
        # elif rho_fl_mean > 4:
            # print("rho_lr_mean case")
            # vL = -0.5*MAX_SPEED
            # vR = 0.5*MAX_SPEED
        # elif rho_l_mean > 4:
            # print("rho_l_mean case")
            # vL = 0.5*MAX_SPEED
            # vR = 0.2*MAX_SPEED
        # elif rho_r_mean > 4:
            # print("rho_r_mean case")
            # vL = 0.2*MAX_SPEED
            # VR = 0.5*MAX_SPEED
        # else:
            # print("forward case")
            # vL = 0.5*MAX_SPEED
            # vR = 0.5*MAX_SPEED
    
    # GOAL: to solve this i want the rboot to fllow the path that is set by the radius of the oval fil
    # I have set points on a grid and 
        #solution, using the display, if the point in gpus is originally orange and then changed to red then im too close and need to turn left, I could also calculate the distance from the 
    # take all lidar sensor data and subtract by current position, if that distance is less than radius then turn left by weight ammount

        
        #Autonomously explore the entire environment, generate a map, and store the map as an npy file
    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    
while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass