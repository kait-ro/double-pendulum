import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import g

class Pendulum():
    def __init__(self, length1, length2, mass1, mass2, theta1, theta2, omega1=0.0, omega2=0.0):
        self.length1 = length1
        self.length2 = length2
        self.mass1 = mass1
        self.mass2 = mass2
        self.theta1 = np.deg2rad(theta1)
        self.theta2 = np.deg2rad(theta2)
        self.omega1 = omega1
        self.omega2 = omega2
        self.initial_state = np.array([self.theta1, self.theta2, self.omega1, self.omega2])

    def coordinates(self):
        x1 = self.length1*np.sin(self.theta1)
        y1 = -self.length1*np.cos(self.theta1)
        x2 = self.length1*np.sin(self.theta1) + self.length2*np.sin(self.theta2)
        y2 = -(self.length1*np.cos(self.theta1) + self.length2*np.cos(self.theta2))
        coordinates = np.array([[x1,y1], [x2,y2]])
        return coordinates

    def equations_of_motions(self, t, state):
        theta1, theta2, omega1, omega2 = state #unpacking tuple
        alpha1 = (
            -g*(self.mass1+self.mass2)*np.sin(theta1) - 
            self.mass2*g*np.sin(theta1-2*theta2) - 
            2*self.mass2*np.sin(theta1-theta2)*(
                self.length2*(omega2**2) + self.length1*(omega1**2)*np.cos(theta1-theta2)
            )
        )/(
            self.length1*(
                self.length2*(
                    2*self.mass1+self.mass2-self.mass2*np.cos(2*theta1-2*theta2)
                ))
        )
        alpha2 = (2*np.sin(theta1-theta2)*( 
            self.length1*(omega1**2)*(self.mass1+self.mass2)
            +
            g*(self.mass1+self.mass2)*np.cos(theta1)
            +
            self.length2*(omega2**2)*self.mass2*np.cos(theta1-theta2)
            ))/(
                self.length2*(
                    2*self.mass1+self.mass2-self.mass2*np.cos(2*theta1-2*theta2)
                ))
        sol = np.array([omega1, omega2, alpha1, alpha2])
        return sol
    
    def simulate(self, time_span, time_eval):
        sol = solve_ivp(self.equations_of_motions, time_span, self.initial_state, t_eval = time_eval)
        return sol

initial_pendulum  = Pendulum(1.0, 1.5, 1.0, 1.0, 15, 30)
print(initial_pendulum.coordinates())
teval = np.arange(0,10, 0.5)
print(initial_pendulum.simulate((0,10), teval))