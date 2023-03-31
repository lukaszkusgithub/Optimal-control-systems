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
def plot_response(name, time_vec, value, title=None, y_label='Value', x_label='Time'):

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
        plot_response('State Space model Step response', step_response_ss[0], step_response_ss[1])

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

    return


# RLC circuit
def example_2():
    # Setting up the circuit parameters
    # resistance in ohms
    R = 12
    # inductance in henries
    L = 1
    # capacitance in farads
    C = 0.0001

    # Creating the transfer function of the circuit
    sys_tf = scipy.signal.TransferFunction(np.array([1, 0]), np.array([L, R, 1/C]))

    # Computing the step response of the circuit using the transfer function
    step_response_tf = scipy.signal.step(sys_tf)

    # Computing the impulse response of the circuit using the transfer function
    impulse_response_tf = scipy.signal.impulse(sys_tf)

    if True:
        plot_response('Transfer Function step response', step_response_tf[0], step_response_tf[1])
    if True:
        plot_response('Transfer Function impulse response', impulse_response_tf[0], impulse_response_tf[1])

    # Define the state space matrices based on the given circuit parameters
    A_matrix = np.array([[0, 1], [(-1/(L*C)), (-R/L)]])
    B_matrix = np.array([[0], [1/L]])
    C_matrix = np.array([[0, 1]])
    D_matrix = 0

    # Create a state space system using the matrices
    sys_ss = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    # Obtain the step response of the state space system
    step_response_ss = scipy.signal.step(sys_ss)

    # Obtain the impulse response of the state space system
    impulse_response_ss = scipy.signal.impulse(sys_ss)

    if True:
        plot_response('State space model step response', step_response_ss[0], step_response_ss[1])
    if True:
        plot_response('State space model impulse response', impulse_response_ss[0], impulse_response_ss[1])

    # Convert state-space system to transfer function representation
    sys_ss2tf = scipy.signal.ss2tf(A_matrix, B_matrix, C_matrix, D_matrix)

    if True:
        print("Transfer function")
        print(sys_tf)
        print("\nState-space system to transfer function")
        print(sys_ss2tf)

    # Convert the transfer function of a second-order RLC circuit into state-space representation
    # The numerator polynomial is [1, 0], representing the transfer function 1/s
    # The denominator polynomial is [L, R, 1/C], representing the second-order characteristic equation
    sys_tf2ss = scipy.signal.tf2ss([1, 0], [L, R, 1/C])

    if True:
        print("\n\nState space model")
        print(sys_ss)
        print("\nTransfer function to State space model")
        print(sys_tf2ss)

    # inductance in henries
    L = 0.15

    sys_ss2tf = scipy.signal.ss2tf(A_matrix, B_matrix, C_matrix, D_matrix)

    if True:
        print("Transfer function")
        print(sys_tf)
        print("\nState-space system to transfer function")
        print(sys_ss2tf)
        print(f'\n{sys_ss2tf}')

    sys_tf2ss = scipy.signal.tf2ss([1, 0], [L, R, 1 / C])

    if True:
        print("\n\nState space model")
        print(sys_ss)
        print("\nTransfer function to State space model")
        print(sys_tf2ss)
        print(f'\n{sys_tf2ss}')

    return 0


# Planar manipulator with one degree of freedom.
def example_3():
    # coefficient of friction
    d = 0.1
    # mass
    m = 1
    # length
    L = 0.5
    # moment of inertia
    J = (1/3) * m * L**2

    # Define state space matrices
    A_matrix = np.array([[0, 1], [0, (-d/J)]])
    B_matrix = np.array([[0], [1/J]])
    C_matrix = np.array([[1, 0]])
    D_matrix = 0

    # Create state space system
    sys_ss = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    # Compute step response
    step_response_sys_ss = scipy.signal.step(sys_ss)

    if True:
        plot_response('Manipulator', step_response_sys_ss[0], step_response_sys_ss[1], y_label='Gain')

    # Define the time duration and time range
    t_m = 10
    t = np.linspace(0, t_m)

    # Define the input signals
    Tau_increment = np.linspace(0, 1)
    Tau_decrement = 1 - np.linspace(0, 1)

    # Obtain the system response for inputs
    Tout_i, yout_i, xout_i = scipy.signal.lsim2(sys_ss, U=Tau_increment, T=t)
    Tout_d, yout_d, xout_d = scipy.signal.lsim2(sys_ss, U=Tau_decrement, T=t)

    if True:
        plot_response('Linearly increasing signal', Tout_i, yout_i)
    if True:
        plot_response('Linearly increasing signal', Tout_d, yout_d)

    # Obtain and plot the Bode plot of the system
    w, mag, phase = scipy.signal.bode(sys_ss)
    if True:
        plt.figure('Bode')
        plt.subplot(2, 1, 1, title="Bode Magnitude")
        plt.semilogx(w, mag)
        plt.grid()
        plt.subplot(2, 1, 2, title="Bode Phase")
        plt.semilogx(w, phase)
        plt.grid()
        plt.show()

    return 0


def main():
    example_1()
    example_2()
    example_3()
    return 0


if __name__ == "__main__":
    main()
