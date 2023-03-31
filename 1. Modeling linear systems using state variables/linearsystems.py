import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import os

# for MacOs
if os.name == 'posix':
    matplotlib.use('macosx')

# Define a function named model that takes in two arguments: time (t) and the state of the system (y).
def model(t, y):
    # Set the proportional gain (Kp_gain) to 3.
    Kp_gain = 3

    # Set the time constant (T) to 2.
    T = 2

    # Calculate the A and B matrices for the system.
    A_matrix = -1 / T
    B_matrix = Kp_gain / T

    # Set the input signal (u_signal) to 1.
    u_signal = 1

    # Calculate the derivative of the state variable (y) using the A and B matrices and the input signal (u_signal).
    dydt = A_matrix * y + B_matrix * u_signal

    # Store the derivative in an array.
    result = np.array([dydt], dtype=object)

    # Convert the array to a list and return it.
    return np.ndarray.tolist(result)


# Define a function named plot_response that takes in five arguments: the name of the plot (name),
# the time vector (time_vec), the values to plot (value), an optional title for the plot (title),
# and labels for the x-axis (x_label) and y-axis (y_label).
def plot_response(name, time_vec, value, title=None, x_label='Time', y_label='Value'):

    # If no title is provided, set it to the name of the plot
    if title is None:
        title = name

    # Create a new figure
    plt.figure(name)
    plt.plot(time_vec, value)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()

    # Display the plot
    plt.show()


def example_1():

    # Set the initial condition (y_0) for the system to 0.
    y_0 = 0
    # Set the proportional gain (Kp_gain) to 3.
    Kp_gain = 3
    # Set the time constant (T) to 2.
    T = 2
    # Calculate the A, B, C, D matrices for the system.
    A_matrix = -1 / T
    B_matrix = Kp_gain / T
    C_matrix = 1
    D_matrix = 0

    # Create a transfer function (sys_tf) using the proportional gain (Kp_gain) and time constant (T) values.
    # The numerator of the transfer function is an array containing Kp_gain, and the denominator is an array
    # containing T and 1.
    sys_tf = scipy.signal.TransferFunction(np.array([Kp_gain]), np.array([T, 1]))

    # Calculate the step response of the transfer function (step_response_tf) using the step() function from the
    # scipy.signal module.
    step_response_tf = scipy.signal.step(sys_tf)

    if True:
        plot_response('Transfer Function Step response', step_response_tf[0], step_response_tf[1])

    # Create a state space model (sys_ss) using the A, B, C, and D matrices.
    # The state space model is defined using these matrices in the order A, B, C, D.
    sys_ss = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    # Calculate the step response of the state space model (step_response_ss) using the step() function from the
    # scipy.signal module.
    step_response_ss = scipy.signal.step(sys_ss)

    if True:
        plot_response('StateSpace Step response', step_response_ss[0], step_response_ss[1])

    # Set the maximum time for the simulation (time_max) to 15.
    # Set the time step for the simulation (time_step) to 0.01.
    # Create an array of time values (time_vec) from 0 to time_max with a step of time_step.
    # Define the time span for the solver (time_span) as [0, time_max].
    time_max = 15
    time_step = 0.01
    time_vec = np.arange(0, time_max, time_step)
    time_span = [0, time_max]

    # Set the initial condition (y_0) for the system to [0].
    y_0 = [0]

    # Use the solve_ivp() function from the scipy.integrate module to solve the initial value problem (IVP) defined by
    # the model function, time_span, and y_0. The method used is 'LSODA', and the time values at which the solution
    # will be evaluated are given by time_vec. The relative tolerance is set to 1e-10.
    y_response = scipy.integrate.solve_ivp(model, t_span=time_span, y0=y_0,
                                           method='LSODA',
                                           t_eval=time_vec,
                                           rtol=1e-10)
    if True:
        plot_response('Model Step response', y_response.t, y_response.y[0])

    return 0


def main():
    example_1()

    return 0


if __name__ == "__main__":
    main()
