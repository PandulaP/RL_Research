"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CustomCartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson.
        Pendulum dynamics are adjusted based on nonlinear dynamics defined by Wang et al., 1996
    
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4800                   4800
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -90 deg                 90 deg
        3       Pole Angular Velocity     -Inf                    Inf
    
    Actions:
        Type: Box(1)
        Action                      Action value Range       Action effect (forece)
        Push cart to the left       [-1,0)                   action_value * 50N
        Do nothing                  0                        -
        Push cart to the right      (0,1] * 50N              action_value * 50N 
        
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    
    Reward:
        Reward is 1 for every step taken, including the termination step
    
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
        Possible to pass any initial state when required
    
    Episode Termination:
        Pole Angle is more than 90 degrees.
        Cart Position is more than 2400.
        Episode length is greater than 1,500.
        
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        1,000 over 100 consecutive trials.
        
    
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }       
            
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 9.0 # UPDATED
        self.masspole = 2.0 # UPDATED
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 50.0 # UPDATED
        self.tau = 0.1  # seconds between state updates - UPDATED (original: 0.02)
        self.kinematics_integrator = 'semi-implicit euler' # - UPDATED (original: euler) | semi-implicit is more stable: https://github.com/openai/gym/pull/1019

        # Angle at which to fail the episode
        self.theta_threshold_radians = 90 * 2 * math.pi / 360 # UPDATED -  (original: 12 degrees)
        self.x_threshold = 2.4 * 100 # UPDATED -  (original: 2.4)

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        #self.action_space = spaces.Discrete(2)
        self.action_space = spaces.Box(low=-1, high = 1, shape = (1,1), dtype=np.float32) # UPDATED - (original: spaces.Discrete(2))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        #force = self.force_mag if action == 1 else -self.force_mag # original 'force'
        
        # continuous 'force' | UPDATED - (original: self.force_mag if action == 1 else -self.force_mag)
        if action > 0:
            force = self.force_mag * action  
        elif action < 0:
            force = self.force_mag * action 
        else:
            force = 0 

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        
        # ORIGINAL nonlinear dynamics of the system
        #thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)) # original equation
        
        # UPDATED nonlinear dynamics of the system by Wang et al. 1996 (based on Dimitrakakis and Lagoudakis 2008)
        sin2theta = math.sin(2*theta)
        alpha = 1/(self.total_mass)
        thetaacc = ((self.gravity * sintheta) - (alpha * self.polemass_length * (theta_dot ** 2) * (sin2theta/2)) - (alpha * costheta * force)) / ((self.length * (4.0 / 3.0)) - (alpha * self.polemass_length * (costheta ** 2)))
                
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self, init_state=None):
        if init_state is not None:
            self.state = init_state # accept an initial state
        else:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600 * 2 # UPDATED - since x_threshold is increased
        screen_height = 400 * 2 # UPDATED - since x_threshold is increased

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0 / (2) # UPDATED - since x_threshold is increased
        polelen = scale * (2 * self.length) * (20) # UPDATED - since x_threshold is increased
        cartwidth = 50.0 / (2) # UPDATED - since x_threshold is increased
        cartheight = 30.0 / (2) # UPDATED - since x_threshold is increased

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            cart.set_color(0,0,10) # UPDATED 
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None