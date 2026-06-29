import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import g
from matplotlib.animation import FuncAnimation



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

    def coordinates(self, theta_one, theta_two):
        x1 = self.length1*np.sin(theta_one)
        y1 = -self.length1*np.cos(theta_one)
        x2 = self.length1*np.sin(theta_one) + self.length2*np.sin(theta_two)
        y2 = -(self.length1*np.cos(theta_one) + self.length2*np.cos(theta_two))
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
        result = np.array([omega1, omega2, alpha1, alpha2])
        return result
    
    def simulate(self, time_span, time_eval):
        sol = solve_ivp(self.equations_of_motions, time_span, self.initial_state, t_eval = time_eval)
        return sol


initial_pendulum  = Pendulum(2, 2, 100, 50, 75, 15)
teval = np.arange(0,400, 0.5)

fig, axis = plt.subplots()
axis.set_xlim(-4,4)
axis.set_ylim(-4,4)
animated_plot, = plt.plot([],[])
bobs, = plt.plot([], [], marker='o', linestyle='none')

store_state = initial_pendulum.simulate((0,400), teval)

def update(frame):
    theta1, theta2 = store_state.y[0][frame], store_state.y[1][frame]
    coordinates_at_frame = initial_pendulum.coordinates(theta1, theta2)
    y2 = coordinates_at_frame[1,1]
    y1 = coordinates_at_frame[0,1]
    x2 = coordinates_at_frame[1,0]
    x1 = coordinates_at_frame[0,0]
    animated_plot.set_data([0, x1,x2], [0, y1,y2])
    bobs.set_data([x1, x2], [y1, y2])
    pass

animation = FuncAnimation(
    fig=fig,
    func=update,
    frames=len(teval),
    interval = 66,
)
plt.show()