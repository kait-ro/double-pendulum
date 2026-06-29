import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import g
from matplotlib.animation import FuncAnimation


class Pendulum:
    def __init__(
        self, length1, length2, mass1, mass2, theta1, theta2, omega1=0.0, omega2=0.0
    ):
        self.length1 = length1
        self.length2 = length2
        self.mass1 = mass1
        self.mass2 = mass2
        self.theta1 = np.deg2rad(theta1)
        self.theta2 = np.deg2rad(theta2)
        self.omega1 = omega1
        self.omega2 = omega2
        self.initial_state = np.array(
            [self.theta1, self.theta2, self.omega1, self.omega2]
        )

    def coordinates(self, theta_one, theta_two):
        x1 = self.length1 * np.sin(theta_one)
        y1 = -self.length1 * np.cos(theta_one)
        x2 = self.length1 * np.sin(theta_one) + self.length2 * np.sin(theta_two)
        y2 = -(self.length1 * np.cos(theta_one) + self.length2 * np.cos(theta_two))
        coordinates = np.array([[x1, y1], [x2, y2]])
        return coordinates

    def equations_of_motions(self, t, state):
        theta1, theta2, omega1, omega2 = state  # unpacking tuple
        alpha1 = (
            -g * (self.mass1 + self.mass2) * np.sin(theta1)
            - self.mass2 * g * np.sin(theta1 - 2 * theta2)
            - 2
            * self.mass2
            * np.sin(theta1 - theta2)
            * (
                self.length2 * (omega2**2)
                + self.length1 * (omega1**2) * np.cos(theta1 - theta2)
            )
        ) / (
            self.length1
            * (
                self.length2
                * (
                    2 * self.mass1
                    + self.mass2
                    - self.mass2 * np.cos(2 * theta1 - 2 * theta2)
                )
            )
        )
        alpha2 = (
            2
            * np.sin(theta1 - theta2)
            * (
                self.length1 * (omega1**2) * (self.mass1 + self.mass2)
                + g * (self.mass1 + self.mass2) * np.cos(theta1)
                + self.length2 * (omega2**2) * self.mass2 * np.cos(theta1 - theta2)
            )
        ) / (
            self.length2
            * (
                2 * self.mass1
                + self.mass2
                - self.mass2 * np.cos(2 * theta1 - 2 * theta2)
            )
        )
        result = np.array([omega1, omega2, alpha1, alpha2])
        return result

    def simulate(self, time_span, time_eval):
        sol = solve_ivp(
            self.equations_of_motions, time_span, self.initial_state, t_eval=time_eval
        )
        return sol


def main():
    fig, axis = plt.subplots(2, 2)
    max_reach = max(
        initial_pendulum.length1 + initial_pendulum.length2,
        second_pendulum.length1 + second_pendulum.length2,
    )
    axis[1, 1].set_xlim(-max_reach * 1.1, max_reach * 1.1)
    axis[1, 1].set_ylim(-max_reach * 1.1, max_reach * 1.1)
    (animated_plot,) = axis[1, 1].plot([], [])
    (bobs,) = axis[1, 1].plot([], [], marker="o", linestyle="none")

    (animated_plot_two,) = axis[1, 1].plot([], [])
    (bobs_two,) = axis[1, 1].plot([], [], marker="o", linestyle="none")

    store_state = initial_pendulum.simulate((0, 400), teval)
    store_state_two = second_pendulum.simulate((0, 400), teval)

    # Phase Portraits

    axis[0, 0].plot(store_state.y[0], store_state.y[2])  # theta1 vs omega1
    axis[0, 0].set_title("Phase Portrait of Theta1 vs Omega1")
    axis[1, 0].plot(store_state.y[1], store_state.y[3])  # theta2 vs omega2
    axis[1, 0].set_title("Phase Portrait of Theta2 vs Omega2")

    # Lyapunov Divergence

    divergence = np.abs(store_state.y[0] - store_state_two.y[0])
    axis[0, 1].plot(teval, divergence)
    axis[0, 1].set_yscale("log")
    axis[0, 1].set_title("Lyapunov Divergence (Theta1)")

    coords_all = initial_pendulum.coordinates(store_state.y[0], store_state.y[1])
    x2_trail = coords_all[1, 0]
    y2_trail = coords_all[1, 1]
    (trail,) = axis[1, 1].plot([], [], linewidth=0.5, alpha=0.5, color="purple")
    coords_all_two = second_pendulum.coordinates(
        store_state_two.y[0], store_state_two.y[1]
    )
    x2_trail_two = coords_all_two[1, 0]
    y2_trail_two = coords_all_two[1, 1]
    (trail_two,) = axis[1, 1].plot([], [], linewidth=0.5, alpha=0.5, color="green")

    def update(frame):
        theta1, theta2 = store_state.y[0][frame], store_state.y[1][frame]
        coordinates_at_frame = initial_pendulum.coordinates(theta1, theta2)
        y2 = coordinates_at_frame[1, 1]
        y1 = coordinates_at_frame[0, 1]
        x2 = coordinates_at_frame[1, 0]
        x1 = coordinates_at_frame[0, 0]
        animated_plot.set_data([0, x1, x2], [0, y1, y2])
        bobs.set_data([x1, x2], [y1, y2])

        theta1_two, theta2_two = (
            store_state_two.y[0][frame],
            store_state_two.y[1][frame],
        )
        coordinates_at_frame_two = second_pendulum.coordinates(theta1_two, theta2_two)
        y4 = coordinates_at_frame_two[1, 1]
        y3 = coordinates_at_frame_two[0, 1]
        x4 = coordinates_at_frame_two[1, 0]
        x3 = coordinates_at_frame_two[0, 0]
        animated_plot_two.set_data([0, x3, x4], [0, y3, y4])
        bobs_two.set_data([x3, x4], [y3, y4])

        trail.set_data(x2_trail[:frame], y2_trail[:frame])
        trail_two.set_data(x2_trail_two[:frame], y2_trail_two[:frame])

    animation = FuncAnimation(fig=fig, func=update, frames=len(teval), interval=25)

    plt.show()


if __name__ == "__main__":
    initial_pendulum = Pendulum(2, 1, 2, 1, 76, 90)
    teval = np.arange(0, 400, 0.5)
    second_pendulum = Pendulum(2, 1, 2, 1, 76.1, 90)
    main()
