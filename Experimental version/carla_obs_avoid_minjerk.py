import carla
import random
import time
import sys
import glob
import os
import weakref
import math
import random
from collections import deque

from numpy import empty

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

try:
    import pygame
except ImportError:
    raise RuntimeError('Unable to import PyGame library. Ensure the PyGame package is installed.')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/carla')
except IndexError:
    pass

import carla

from agents.tools.modified_misc import draw_coordinates, draw_waypoints, get_speed
from agents.navigation.modified_controller import ModVehiclePIDController, MinJerkPlanner
from agents.navigation.controller import VehiclePIDController

VIEW_WIDTH = 1920 #Dimension of player window view
VIEW_HEIGHT = 1080

NUM_OF_OBSTACLES = 60 #Number of obstacles to be spawned on the map

DIST_OFFSET = 2 #Offset distance for selecting valid waypoints for path planning

#String that specifies the obstacle formation desired.
#Options are: 'overtake', 'long_overtake', 'full_block', 'for_part_block'
# 'rev_part_block', 
OBSTACLE_FORMATION = 'overtake'

formation_spawn = False #If True, obstacles spawn in a specific formation, else, they spawn randomly

SPAWN_INDEX = 40 #Specific index to select Ego vehicle spawn from list of available points

reset = True #Bool to check whether simulation should be reset
overtake = False #Bool to indicate if Ego should overtake

obstacle_list = [] #List to track all spawned obstacles on map
actor_list = [] #List to track ego vehicle and sensor attached to it
spawn_list = [] #List to track all available spawn points on map

min_distance = 10 #Minimum vehicle spawn distance from junctions and other vehicles

sample_resolution = 1 #Sampling resolution for waypoint search around Ego 
                        #the smaller it is, the better the resolution.

safe_distance = 6 #A buffer distance the Ego must maintain with obstacles

time_delta = 0.05 #Simulation time step in synchronous mode operation

DEST_CLEARANCE = 1.25*safe_distance #Clearance from obstacle for selecting destination waypoint
OBS_CLEARANCE = 0.65*safe_distance
WP_CLEARANCE = 1.1*safe_distance
VER_CLEARANCE = 4

#List of lane types in Carla that are considered to be roads
lanetype_list = [carla.LaneType.Driving,
                 carla.LaneType.OnRamp, 
                 carla.LaneType.OffRamp,
                 carla.LaneType.Stop,
                 carla.LaneType.Shoulder,
                 carla.LaneType.Parking,
                 carla.LaneType.Tram]


#Vehicle class
class Vehicle():
    def __init__(self, world, spawn_index):

        self._waypoints_queue = deque(maxlen=10000)
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = 2.0
        self._args_lateral_dict = {'K_P': 1.5, 'K_I': 0.05, 'K_D': 1000, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 0.1, 'K_I': 0.05, 'K_D': 1000.0, 'dt': self._dt}
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8
        self._offset = 0
        self._base_min_distance = 2.0
        self._distance_ratio = 0.5
        self._follow_speed_limits = False
        self.wmap = world.get_map()
      
        self.world = world

        global spawn_list
        self.start_point = spawn_list[spawn_index]
        vehicle_bp = self.world.get_blueprint_library().find('vehicle.dodge.charger_2020')
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.start_point)

        #Add actor to tracking list
        global actor_list
        actor_list.append(self.vehicle)
        #This spawn point is now occupied and must be removed from available spawns
        spawn_list.remove(self.start_point)

        self.init_controller()

        return None

    def init_controller(self):
        #PID controller
        self.vehicle_controller = VehiclePIDController(self.vehicle,
                                                        args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)

        #Minimum jerk trajectory planner
        self.vehicle_path_planner = MinJerkPlanner(time_delta = time_delta, 
                                                   avg_speed = self._target_speed)


        # Compute the current vehicle waypoint
        current_waypoint = self.get_waypoint(self.wmap)
        self.target_waypoint = current_waypoint
        self._waypoints_queue.append(self.target_waypoint)

    def path_plan(self, waypoint_sequence):

        trajectory = self.vehicle_path_planner.generate_path(waypoint_sequence)

        for coord in trajectory[0]:
            x_coord, y_coord = coord
            wp = carla.Transform(carla.Location(x = x_coord, y = y_coord), carla.Rotation())


            if wp:
                self._waypoints_queue.append(wp)
            else:
                print('\nNo waypoint returned!')

        return self._waypoints_queue


    def run_step(self, debug=False):

        if self._follow_speed_limits:
            self._target_speed = self.vehicle.get_speed_limit()

        # Purge the queue of obsolete waypoints
        veh_location = self.get_transform().location
        vehicle_speed = get_speed(self.vehicle) / 3.6
        self._min_distance = self._base_min_distance + self._distance_ratio * vehicle_speed

        num_waypoint_removed = 0

        for waypoint in self._waypoints_queue:

            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._min_distance

            print(waypoint)
            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        # Get the target waypoint and move using the PID controllers. Stop if no target waypoint
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            self.target_waypoint = self._waypoints_queue[0]
            control = self.vehicle_controller.run_step(self._target_speed, self.target_waypoint)

        if debug:
            draw_coordinates(self.vehicle.get_world(), [self.target_waypoint], 1.0)

        self.vehicle.apply_control(control)

        return control

    #Returns current transform (location & rotation) 
    def get_transform(self):

        vehicle_transform = self.vehicle.get_transform()

        return vehicle_transform
    
    #Returns current waypoint 
    def get_waypoint(self, wmap):

        vehicle_transform = self.vehicle.get_transform()
        vehicle_waypoint = wmap.get_waypoint(vehicle_transform.location)

        return vehicle_waypoint

#Collision sensor class
class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, transform):
        """Constructor method"""
        self.transform = transform
        self.sensor = None
        self._parent = parent_actor
        self.world = self._parent.vehicle.get_world()
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = self.world.spawn_actor(blueprint, self.transform, attach_to=self._parent.vehicle)

        global actor_list
        actor_list.append(self.sensor)

        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        #self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

        return None

#RGB camera class
class CameraSensor():

    def __init__(self, parent_actor, transform, display):
        self.display = display
        self.transform = transform
        self.surface = None
        self.sensor = None
        self._parent = parent_actor
        self.world = self._parent.vehicle.get_world()
        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', '1920')
        blueprint.set_attribute('image_size_y', '1080')
        blueprint.set_attribute('fov', '90')
        self.sensor = self.world.spawn_actor(blueprint, self.transform, attach_to = self._parent.vehicle)

        global actor_list
        actor_list.append(self.sensor)

        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor._on_capture(weak_self, image))
    
    #Renders image captured by camera on game window
    def render_image(self):

        if self.surface is not None:
            self.display.blit(self.surface, (0, 0))

        return None

    @staticmethod
    def _on_capture(weak_self, image):
        """Image processing method"""

        self = weak_self()
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype = np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        self.render_image()
        pygame.display.flip()

#Class for obstacle detection sensor
class ObstacleSensor():
    
    def __init__(self, parent_actor, transform):
        """Constructor method"""
        self.transform = transform
        self.sensor = None
        self._parent = parent_actor
        self.world = self._parent.vehicle.get_world()
        obstacle_bp = self.world.get_blueprint_library().find('sensor.other.obstacle')
        obstacle_bp.set_attribute('distance', '5.0')# Detection distance in meters
        obstacle_bp.set_attribute('hit_radius', '1.5')  # Radius of the detection area
        obstacle_bp.set_attribute('only_dynamics', 'True')  # To detect only dynamic objects like vehicles
        obstacle_bp.set_attribute('debug_linetrace', 'True')  # Set to 'True' for visualization
        obstacle_bp.set_attribute('sensor_tick', '0.0') 

        self.sensor = self.world.spawn_actor(obstacle_bp, self.transform, attach_to=self._parent.vehicle)

        global actor_list
        actor_list.append(self.sensor)


class Obstacle():

    def __init__(self, world, spawn_point):

        self.vehicle = None
        self.spawn_point = spawn_point
        self.world = world
        obstacle_bp = self.world.get_blueprint_library().find('vehicle.dodge.charger_2020')
        self.obstacle = self.world.spawn_actor(obstacle_bp, self.spawn_point)

        if self.obstacle:
            global obstacle_list
            obstacle_list.append(self.obstacle)

              

    def get_transform(self):
        if self.obstacle:
            return self.obstacle.get_transform()
        
    def get_waypoint(self, wmap):

            obs_transform = self.get_transform()
            obs_waypoint = wmap.get_waypoint(obs_transform.location)

            return obs_waypoint

#Establish connection to server and configure it
def client_connect():
    #Creates a client to the server
    client = carla.Client('127.0.0.1', 2000)

    #Checks if connection to server is established within a time frame
    client.set_timeout(20.0)


    world = client.get_world()

    #Reload world to remove unwanted actors
    world = client.load_world('Town01')

    tm = client.get_trafficmanager(8000)

    settings = world.get_settings()
            
    #This script run on synchronous mode, so Server AND Traffic Manager are set to synchronous mode
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = time_delta
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)


    #Return World, Client, and Traffic Manager object for use in code
    return client, world, tm

#Updates spectator location and orientation
def spawn_spec(world, ego_vehicle):

    spectator = world.get_spectator()

    #Spectator is positioned where the Ego spawns
    spec_spawn_point = ego_vehicle.start_point
    spec_spawn_point.location.z = spec_spawn_point.location.z + 5
    # spec_spawn_point.location.y = spec_spawn_point.location.y + 1
    # spec_spawn_point.location.x = spec_spawn_point.location.x + 0
    # spec_spawn_point.rotation.pitch = spec_spawn_point.rotation.pitch - 15
    # spec_spawn_point.rotation.yaw = spec_spawn_point.rotation.yaw - 180
    spectator.set_transform(spec_spawn_point)

    return spectator

#Function to spawn an obstacle at a random location
def spawn_obstacle_rand(world):

    global spawn_list

    #Find number of spawn points
    spawn_len = len(spawn_list)
    wmap = world.get_map()
    wp_list = wmap.generate_waypoints(2.0)

    junction_list = []

    #Create a list of all waypoints at junctions
    for wp in wp_list:
        if wp.is_junction:
            junction_list.append(wp)

    spawn = False

    if spawn_len > 0:

        while True:
            spawn = True
            rand_int = random.randint(0, spawn_len - 1)
            random_spawn = spawn_list[rand_int]

            vehicles = obstacle_list
            target_location = random_spawn.location       

            #Next sections of code check if a potential spawn is close to a vehicle or junction 
            for waypoint in junction_list:

                distance = target_location.distance(waypoint.transform.location)

                #Target location is too close to junction, do not spawn obstacle
                if distance < min_distance:
                    spawn = False
                    break
 
            #Check if spawn is not already False
            if spawn:
                for vehicle in vehicles:           

                    distance = target_location.distance(vehicle.get_transform().location)
                    #Target location is too close to other vehicles, do not spawn obstacle
                    if distance < min_distance:
                        spawn = False
                        break

            if spawn:

            
                obs = Obstacle(world, random_spawn)

                #This spawn point is now occupied and must be removed from list of available spawns
                spawn_list.remove(random_spawn)
                break

    else:
        print('\nNo available spawn points left!')

  
    return None

#Function that spawns an obstacle in a specific formation in front of the Ego
#Used only to test A* overtake algorithm
def spawn_obstacle_form(world, ego, form):

    world.tick()

    wmap = world.get_map()
    ego_location = ego.get_transform().location

    obs1_loc = carla.Location()
    obs2_loc = carla.Location()

    obs3_loc = carla.Location()
    obs4_loc = carla.Location()

    
    if form == 'full_block':
        obs1_loc.x = ego_location.x + 60
        obs1_loc.y = ego_location.y
        obs1_loc.z = ego_location.z

        obs2_loc.x = ego_location.x + 60
        obs2_loc.y = ego_location.y - 4
        obs2_loc.z = ego_location.z

        obs1 = Obstacle(world, carla.Transform(obs1_loc, carla.Rotation()))
        obs2 = Obstacle(world, carla.Transform(obs2_loc, carla.Rotation()))

    elif form == 'for_part_block':

        obs1_loc.x = ego_location.x + 60
        obs1_loc.y = ego_location.y
        obs1_loc.z = ego_location.z

        obs2_loc.x = ego_location.x + 67.5
        obs2_loc.y = ego_location.y - 4
        obs2_loc.z = ego_location.z

        obs1 = Obstacle(world, carla.Transform(obs1_loc, carla.Rotation()))
        obs2 = Obstacle(world, carla.Transform(obs2_loc, carla.Rotation()))

    elif form == 'rev_part_block':

        obs1_loc.x = ego_location.x + 60
        obs1_loc.y = ego_location.y
        obs1_loc.z = ego_location.z

        obs2_loc.x = ego_location.x + 52.5
        obs2_loc.y = ego_location.y - 4
        obs2_loc.z = ego_location.z

        obs1 = Obstacle(world, carla.Transform(obs1_loc, carla.Rotation()))
        obs2 = Obstacle(world, carla.Transform(obs2_loc, carla.Rotation()))

    elif form == 'overtake':

        obs1_loc.x = ego_location.x + 60
        obs1_loc.y = ego_location.y
        obs1_loc.z = ego_location.z

        obs1 = Obstacle(world, carla.Transform(obs1_loc, carla.Rotation()))

    elif form == 'long_overtake':

        obs1_loc.x = ego_location.x + 60
        obs1_loc.y = ego_location.y
        obs1_loc.z = ego_location.z

        obs2_loc.x = ego_location.x + 76
        obs2_loc.y = ego_location.y
        obs2_loc.z = ego_location.z

        # obs3_loc.x = ego_location.x + 80
        # obs3_loc.y = ego_location.y
        # obs3_loc.z = ego_location.z

        # obs4_loc.x = ego_location.x + 52.5
        # obs4_loc.y = ego_location.y - 4
        # obs4_loc.z = ego_location.z

        obs1 = Obstacle(world, carla.Transform(obs1_loc, carla.Rotation()))
        obs2 = Obstacle(world, carla.Transform(obs2_loc, carla.Rotation()))
        # obs3 = Obstacle(world, carla.Transform(obs3_loc, carla.Rotation()))
        # obs4 = Obstacle(world, carla.Transform(obs4_loc, carla.Rotation()))

    elif form == 'zig_zag':

        obs1_loc.x = ego_location.x + 60
        obs1_loc.y = ego_location.y
        obs1_loc.z = ego_location.z

        obs2_loc.x = ego_location.x + 67.5
        obs2_loc.y = ego_location.y - 4
        obs2_loc.z = ego_location.z

        obs3_loc.x = ego_location.x + 80
        obs3_loc.y = ego_location.y
        obs3_loc.z = ego_location.z

        obs4_loc.x = ego_location.x + 87.5
        obs4_loc.y = ego_location.y - 4
        obs4_loc.z = ego_location.z

        obs1 = Obstacle(world, carla.Transform(obs1_loc, carla.Rotation()))
        obs2 = Obstacle(world, carla.Transform(obs2_loc, carla.Rotation()))
        obs3 = Obstacle(world, carla.Transform(obs3_loc, carla.Rotation()))
        obs4 = Obstacle(world, carla.Transform(obs4_loc, carla.Rotation()))
    
    elif form == 'corner':

        obs1 = Obstacle(world, carla.Transform(carla.Location(x=10, y=10, z=1.0), carla.Rotation()))

    else:
        print('Failed to spawn formation.')


    return None

#Function to execute all commands to overtake and safely manouvre around obstacle
def handle_overtake(sensor, ego, world, event):

    wmap = world.get_map()

    sensor.sensor.stop()

    # wp_list = []
    # wp = ego.get_waypoint(wmap)
    # for i in range(5):
    #     wp_list.append(wp)
    #     next_wp = wp.next(4)[0]
    #     wp = next_wp

    # draw_waypoints(world, wp_list, 1.0)

    ego.vehicle.set_autopilot(False)
    print('Overtake handling...')
    print('Detected: ' + str(event.other_actor))
      
    ego_wp = ego.get_waypoint(wmap)
    obs_shortlist = shortlist_obstacles(wmap, event)
    obs_check, obs_start = verify_obs(ego, wmap, obs_shortlist)

    print('\nObs_check = ' + str(obs_check))

    if obs_check == True:
        destination = find_destination(wmap, obs_start, obs_shortlist)
        #draw_waypoints(world, [destination], 1.0)
        graph = select_waypoints(ego_wp, obs_start, wmap, destination, obs_shortlist)

        draw_waypoints(world, graph, 1.0)

        if graph:
            # route = ego.path_plan(graph)
            # draw_coordinates(world, route, 1.0)

            ego._waypoints_queue.extend(graph)

            while ego._waypoints_queue:
                ego.run_step()

            sensor.sensor.listen(lambda event: handle_overtake(sensor, ego, world, event))
        else:
            print('Since no valid route is available, no overtaking will occur')

    ego.vehicle.set_autopilot(True)

    if obs_check == False:
        time.sleep(2.0)
        sensor.sensor.listen(lambda event: handle_overtake(sensor, ego, world, event))

    print('\nhandle_overtake done!')
    return

    #Pass returned list of waypoints into VehiclePID controller
    #When destination reached, let Traffic Manager take over again

#Function that shortlists obstacles on the same road as the Ego
def shortlist_obstacles(wmap, event):

    #Find waypoint of first obstacle in front of Ego
    obs_waypoint = wmap.get_waypoint(event.other_actor.get_location())
    obs_location = obs_waypoint.transform.location

    # Rounding off to avoid floating point imprecision
    x1, y1, z1, x2, y2, z2 = np.round([obs_location.x, obs_location.y, obs_location.z, obs_location.x, obs_location.y, obs_location.z], 0)
    obs_waypoint.transform.location = obs_location

    road_id = obs_waypoint.road_id

    global obstacle_list
    #Compile a list of obstacles in the same road as Ego
    obs_list = []
    for obs in obstacle_list:
        obs_loc = obs.get_location()
        obs_wp = wmap.get_waypoint(obs_loc)
        if obs_wp.road_id == road_id:
            obs_list.append(obs_wp)

    return obs_list

#Cross checks with waypoints ahead of Ego to identify false positives
def verify_obs(ego, wmap, obs_list):

    wp_num = 10

    wp = ego.get_waypoint(wmap)
    for i in range(wp_num):
        for obs in obs_list:
            obs_loc = obs.transform.location
            if wp.transform.location.distance(obs_loc) < VER_CLEARANCE:
                return True, obs
        next_wp = wp.next(sample_resolution)[0]
        wp = next_wp

    return False, None
            
#Function to find suitable destination waypoint on the same lane that is unoccupied
def find_destination(wmap, obs_start, obs_list):

    #Find waypoint of first obstacle in front of Ego
    obs_waypoint = obs_start
    obs_location = obs_waypoint.transform.location

    # Rounding off to avoid floating point imprecision
    x1, y1, z1, x2, y2, z2 = np.round([obs_location.x, obs_location.y, obs_location.z, obs_location.x, obs_location.y, obs_location.z], 0)
    obs_waypoint.transform.location = obs_location

    road_id, section_id, lane_id = obs_waypoint.road_id, obs_waypoint.section_id, obs_waypoint.lane_id

    #Create a list of obstacles in the same road, section, and lane as Ego
    obstacle_list = []
    for obs in obs_list:
        obs_wp = obs
        if obs_wp.road_id == road_id and obs_wp.section_id == section_id and obs_wp.lane_id == lane_id:
            obstacle_list.append(obs_wp)

    destination_found = False
    #Iterate through waypoints in front of first obstacle
    #to find a viable destination, in the same lane, unoccupied by obstacles
    wp = obs_waypoint.next(sample_resolution)[0]
    while not destination_found and wp:
        for obs_wp in obstacle_list:
            obs_loc = obs_wp.transform.location
            #If too close to any one obstacle, it fails
            if wp.transform.location.distance(obs_loc) < DEST_CLEARANCE:
                destination_found = False
                next_wp = wp.next(sample_resolution)[0]
                wp = next_wp
                break
            else:
                destination_found = True

    if destination_found:
        print('\nDestination waypoint found!')

        return wp
    else:
        print('\nNo destination found. Debug simulation')
        return None

#Function to get all available waypoints within a certain radius around the Ego
def select_waypoints(ego_waypoint, obs_wp, wmap, destination, obs_list):

    route = []
    obs_waypoint = obs_wp
    obs_loc = obs_waypoint.transform.location

    # Rounding off to avoid floating point imprecision
    x1, y1, z1, x2, y2, z2 = np.round([obs_loc.x, obs_loc.y, obs_loc.z, obs_loc.x, obs_loc.y, obs_loc.z], 0)
    obs_waypoint.transform.location = obs_loc

    #Lane ID of Ego's current lane, which also indicates direction using signed integers
    obs_lane_dir = obs_waypoint.lane_id

    #Select a waypoint offset to the front of the Ego
    offset_waypoint = ego_waypoint.next(5)[0]

    #Get waypoints on the left and right lanes right next to Ego
    left_start_wp = offset_waypoint.get_left_lane()

    side_lane_wp = obs_waypoint.get_left_lane()

    #print('Lane ID: ' + str(left_start_wp.lane_id))

    if left_start_wp:
        side_lane_dir = side_lane_wp.lane_id
        print(obs_lane_dir)
        print(side_lane_dir)
        #Check if lane potentially used to overtake is a road
        for lanetype in lanetype_list:
            if left_start_wp.lane_type == lanetype:
                lanetype_valid = True
                print('Valid lane type')
                break
            else:
                print(str(left_start_wp.lane_type))

        if lanetype_valid:

            var1 = abs(obs_lane_dir + side_lane_dir)
            var2 = max(abs(obs_lane_dir), abs(side_lane_dir))

            print('var1 = ' + str(var1))
            print('var2 =  ' + str(var2))

            if obs_lane_dir == side_lane_dir:
                left_wps = []
            elif var1 > var2:
                left_wps = path_find(left_start_wp, wmap, destination, obs_list, ego_waypoint, 'forward')
            elif var1 < var2:
                left_wps = path_find(left_start_wp, wmap, destination, obs_list, ego_waypoint, 'reverse')
        else:
            left_wps = []
    else:
    #Otherwise, return an empty list
        left_wps = []


    #right_start_wp = offset_waypoint.get_right_lane()

    # if right_start_wp.lane_type == carla.LaneType.Driving:
    #     right_wps = path_find(right_start_wp, wmap, destination, obs_list, ego_waypoint, 'forward')
    # else:
    #     right_wps = []

   

    if left_wps == []: #and right_wps == []
        print('\nNo valid route available')
        return []
    else:
        route.extend(left_wps)
        #route.extend(right_wps)
        route.append(destination)
        return route

#Selects available waypoints on a single lane. Direction is needed to correct for for built-in waypoint directin.
def path_find(start_wp, wmap, destination, obs_list, ego_wp, direction = 'forward'):

    road_id, section_id, lane_id = start_wp.road_id, start_wp.section_id, start_wp.lane_id

    #Create a list of obstacles in the same road, section, and lane as left lane
    obstacle_list = []
    for obs in obs_list:
        obs_wp = obs
        if obs_wp.road_id == road_id and obs_wp.section_id == section_id and obs_wp.lane_id == lane_id:
            obstacle_list.append(obs_wp)

    #Bool to indicate if seach should be stopped
    stop_search = False

    wp_list = []

    wp = start_wp

    while not stop_search and wp:

        #Bool to indicate if waypoint is too near to an obstacle
        obstacle_hit = False

        if obstacle_list:
            for obs_wp in obstacle_list:
                obs_loc = obs_wp.transform.location
                #If too close to any one obstacle, it fails
                if wp.transform.location.distance(obs_loc) < OBS_CLEARANCE:
                    obstacle_hit = True
                    break

        #Distance between current waypoint and destination and Ego, respectively 
        dist_to_dest = destination.transform.location.distance(wp.transform.location)
        dist_to_ego = ego_wp.transform.location.distance(wp.transform.location)

        #Distance between destination and Ego
        dest_to_ego = destination.transform.location.distance(ego_wp.transform.location)

        #If current waypoint is close enough to destination, check if it's too close to obstacle
        #If it is, ignore it and stop searching.
        #Otherwise, append it to the route
        #If it encounters obstacle before nearing destination, delete the route as it's not valid
        #Otherwise, append current waypoint to the route
        if dist_to_dest < WP_CLEARANCE:
            if not obstacle_hit:
                wp_list.append(wp)
            else:
                wp_list = []
            break
        elif obstacle_hit:
            wp_list = []
        else:
            wp_list.append(wp)



        if direction == 'reverse':
            next_wp = wp.previous(sample_resolution)[0]
        elif direction == 'forward':
            next_wp = wp.next(sample_resolution)[0]
        else:
            print('Invalid direction indicator.')
     
        
        wp = next_wp
        #print(stop_search)

    return wp_list

#Resets the simulation environment to before by deleting spawned actors
def destroy_actors(client, world): 

    global actor_list
    global obstacle_list

    stat1 = client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    stat2 = client.apply_batch([carla.command.DestroyActor(x) for x in obstacle_list])

    if stat1:
        actor_list = []

    if stat2:
        obstacle_list = []

    print('Obstacle list length: ' + str(len(obstacle_list)))
    print(len(obstacle_list))


    return None


# def game_loop(client, world, tm, display):
def game_loop(client, world, tm, display):
    try:
        print('\nStart of new loop-------------------------')
        #Initialize pygame modules
        pygame.init() 
        
        #Delete all leftover actors from previous simulation
        destroy_actors(client, world)

        #Set weather of map
        weather = carla.WeatherParameters(
            cloudiness = 0.0,
            precipitation = 0.0,
            sun_altitude_angle = 70.0
            )

        
        world.set_weather(weather)

        #Create a list of all available spawn points on the map
        global spawn_list
        spawn_list = world.get_map().get_spawn_points()

        #Spawn Ego vehicle; spawn position is returned
        ego_vehicle = Vehicle(world, SPAWN_INDEX)

        #Spawn spectator where the Ego is
        spectator = spawn_spec(world, ego_vehicle)

        obstacle_sens1 = ObstacleSensor(ego_vehicle, carla.Transform(carla.Location(x=2.0, z=1.0)))

        #Spawn collision sensor(s)
        collision_sens1 = CollisionSensor(ego_vehicle, carla.Transform())

        #Spawn camera for first-person player view
        rgb_cam1 = CameraSensor(ego_vehicle, carla.Transform(carla.Location(x=-15.0,y=-0.37, z=5.0), carla.Rotation()), display)
        
        if formation_spawn:
            spawn_obstacle_form(world, ego_vehicle, OBSTACLE_FORMATION)
        else:
            #Spawn obstacles
            for obstacle in range(NUM_OF_OBSTACLES):
                world.tick()
                spawn_obstacle_rand(world)


        tm.ignore_lights_percentage(ego_vehicle.vehicle, 100)
        tm.ignore_signs_percentage(ego_vehicle.vehicle, 100)
        tm.update_vehicle_lights(ego_vehicle.vehicle, True)
        tm.set_desired_speed(ego_vehicle.vehicle, 30)


        #Vehicle initialized to obey Traffic Manager
        ego_vehicle.vehicle.set_autopilot(True)

        obstacle_sens1.sensor.listen(lambda event: handle_overtake(obstacle_sens1, ego_vehicle, world, event))

        clock = pygame.time.Clock()


        while True:
            clock.tick()
            world.tick()


            #Reset bool to check whether simulation should be reset
            global reset
            reset = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        reset = True
            if reset:
                print("Resetting Simulation")
                break

    finally:

        if world is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            tm.set_synchronous_mode(True)

        print('\nEnd of game loop-------------------------')

            
def main():

    try:

        #Create separate game window to enter player view
        display = pygame.display.set_mode(
        (VIEW_WIDTH, VIEW_HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

        #Connect to server
        client, world, tm = client_connect()
        
        #Check if simulation should be reset or terminated
        global reset
        while reset:

            print('\nSimulation reset!. Running again.')
            game_loop(client, world, tm, display)
            
    except KeyboardInterrupt:
        
        print('\nManual termination by user!') 
        
    finally:
        destroy_actors(client, world)

        if world is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            tm.set_synchronous_mode(True)

if __name__ == '__main__':
    main()
