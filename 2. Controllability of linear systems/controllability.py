import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import os
import control

# for MacOs
if os.name == 'posix':
    matplotlib.use('macosx')


# Define function to check controllability matrix
# This function checks the controllability matrix of a given system
# Inputs:
# - matrix_a: the A matrix of the system
# - matrix_b: the B matrix of the system
# Output:
# - True if the controllability matrix is of full rank, indicating the system is controllable
# - False otherwise
def check_controllability_matrix(matrix_a, matrix_b):
    # Calculate the controllability matrix using the built-in function in the control library
    controllability_matrix = control.ctrb(matrix_a, matrix_b)
    # Calculate the rank of the controllability matrix using numpy's matrix_rank function
    controllability_matrix_rank = np.linalg.matrix_rank(controllability_matrix)
    # Check if the rank of the controllability matrix is equal to the size of matrix_a
    matrix_a_size = len(matrix_a)
    if controllability_matrix_rank == matrix_a_size:
        return True
    else:
        return False


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
    # System 1
    A_matrix = np.array([[-0.5, 0], [0, -0.5]])
    B_matrix = np.array([[0.5], [0.5]])
    C_matrix = np.array([[0, 1]])
    D_matrix = 0

    sys_1 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix,
                                     D_matrix)

    # Check controllability of System 1
    sys_1_controllability = check_controllability_matrix(A_matrix, B_matrix)

    # System 2
    A_matrix = np.array([[-1, 0, 0], [0, -0.5, 0], [0, 0, -1/3]])
    B_matrix = np.array([[1], [0.5], [1/3]])
    C_matrix = np.array([[0, 0, 1]])
    D_matrix = 0

    sys_2 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix,
                                    D_matrix)

    # Check controllability of System 2
    sys_2_controllability = check_controllability_matrix(A_matrix, B_matrix)

    # System 3
    A_matrix = np.array([[-2, 0, -2], [0, 0, 1], [0.5, -0.5, -0.5]])
    B_matrix = np.array([[2], [0], [0]])
    C_matrix = np.array([[0, 0, 1]])
    D_matrix = 0

    sys_3 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix,
                                    D_matrix)

    # Check controllability of System 3
    sys_3_controllability = check_controllability_matrix(A_matrix, B_matrix)

    # Compute step response of the systems
    sys_1_step_response = scipy.signal.step(sys_1)
    sys_2_step_response = scipy.signal.step(sys_2)
    sys_3_step_response = scipy.signal.step(sys_3)

    # Generate a sinusoidal input signal and compute the response of the systems
    time = np.arange(0, 20, 0.1)
    sinus_signal = np.sin(2 * time * np.pi)

    sys_1_sin_response = scipy.signal.lsim2(sys_1, U=sinus_signal, T=time)
    sys_2_sin_response = scipy.signal.lsim2(sys_2, U=sinus_signal, T=time)
    sys_3_sin_response = scipy.signal.lsim2(sys_3, U=sinus_signal, T=time)

    # Plot response fo the systems
    if True:
        plot_response('State Space model 1 Step response', sys_1_step_response[0], sys_1_step_response[1])
    if True:
        plot_response('State Space model 2 Step response', sys_2_step_response[0], sys_2_step_response[1])
    if True:
        plot_response('State Space model 3 Step response', sys_3_step_response[0], sys_3_step_response[1])
    if True:
        plot_response('State Space model 1 Sinus response', sys_1_sin_response[0], sys_1_sin_response[1])
    if True:
        plot_response('State Space model 2 Sinus response', sys_2_sin_response[0], sys_2_sin_response[1])
    if True:
        plot_response('State Space model 3 Sinus response', sys_3_sin_response[0], sys_3_sin_response[1])

    # Define system matrices for first system, with two different output matrices
    A_matrix = np.array([[-0.5, 0], [0, -0.5]])
    B_matrix = np.array([[0.5], [0.5]])
    C_matrix = np.array([[1, 0]])
    D_matrix = 0
    sys_1_matrix_c_1 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    C_matrix = np.array([[0, 1]])
    sys_1_matrix_c_2 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    # Define system matrices for second system, with three different output matrices
    A_matrix = np.array([[-1, 0, 0], [0, -0.5, 0], [0, 0, -1 / 3]])
    B_matrix = np.array([[1], [0.5], [1 / 3]])
    C_matrix = np.array([[1, 0, 0]])
    D_matrix = 0
    sys_2_matrix_c_1 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    C_matrix = np.array([[0, 1, 0]])
    sys_2_matrix_c_2 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    C_matrix = np.array([[0, 0, 1]])
    sys_2_matrix_c_3 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    # Define system matrices for third system, with three different output matrices
    A_matrix = np.array([[-2, 0, -2], [0, 0, 1], [0.5, -0.5, -0.5]])
    B_matrix = np.array([[2], [0], [0]])
    C_matrix = np.array([[1, 0, 0]])
    D_matrix = 0
    sys_3_matrix_c_1 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    C_matrix = np.array([[0, 1, 0]])
    sys_3_matrix_c_2 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    C_matrix = np.array([[0, 0, 1]])
    sys_3_matrix_c_3 = scipy.signal.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    # Compute step responses for each system with each output matrix
    sys_1_matrix_c_1_step_response = scipy.signal.step(sys_1_matrix_c_1)
    sys_1_matrix_c_2_step_response = scipy.signal.step(sys_1_matrix_c_2)

    sys_2_matrix_c_1_step_response = scipy.signal.step(sys_2_matrix_c_1)
    sys_2_matrix_c_2_step_response = scipy.signal.step(sys_2_matrix_c_2)
    sys_2_matrix_c_3_step_response = scipy.signal.step(sys_2_matrix_c_3)

    sys_3_matrix_c_1_step_response = scipy.signal.step(sys_3_matrix_c_1)
    sys_3_matrix_c_2_step_response = scipy.signal.step(sys_3_matrix_c_2)
    sys_3_matrix_c_3_step_response = scipy.signal.step(sys_3_matrix_c_3)

    # Compute sinus responses for each system with each output matrix
    sys_1_matrix_c_1_sin_response = scipy.signal.lsim2(sys_1_matrix_c_1, U=sinus_signal, T=time)
    sys_1_matrix_c_2_sin_response = scipy.signal.lsim2(sys_1_matrix_c_2, U=sinus_signal, T=time)

    sys_2_matrix_c_1_sin_response = scipy.signal.lsim2(sys_2_matrix_c_1, U=sinus_signal, T=time)
    sys_2_matrix_c_2_sin_response = scipy.signal.lsim2(sys_2_matrix_c_2, U=sinus_signal, T=time)
    sys_2_matrix_c_3_sin_response = scipy.signal.lsim2(sys_2_matrix_c_3, U=sinus_signal, T=time)

    sys_3_matrix_c_1_sin_response = scipy.signal.lsim2(sys_3_matrix_c_1, U=sinus_signal, T=time)
    sys_3_matrix_c_2_sin_response = scipy.signal.lsim2(sys_3_matrix_c_2, U=sinus_signal, T=time)
    sys_3_matrix_c_3_sin_response = scipy.signal.lsim2(sys_3_matrix_c_3, U=sinus_signal, T=time)

    if True:
        plot_response('State Space model 1 Step response C = [1, 0]', sys_1_matrix_c_1_step_response[0],
                      sys_1_matrix_c_1_step_response[1])
    if True:
        plot_response('State Space model 1 Step response C = [0, 1]', sys_1_matrix_c_2_step_response[0],
                      sys_1_matrix_c_2_step_response[1])

    if True:
        plot_response('State Space model 2 Step response C = [1, 0, 0]', sys_2_matrix_c_1_step_response[0],
                      sys_2_matrix_c_1_step_response[1])
    if True:
        plot_response('State Space model 2 Step response C = [0, 1, 0] ', sys_2_matrix_c_2_step_response[0],
                      sys_2_matrix_c_2_step_response[1])
    if True:
        plot_response('State Space model 2 Step response C = [0, 0, 1]', sys_2_matrix_c_3_step_response[0],
                      sys_2_matrix_c_3_step_response[1])

    if True:
        plot_response('State Space model 3 Step response C = [1, 0, 0]', sys_3_matrix_c_1_step_response[0],
                      sys_3_matrix_c_1_step_response[1])
    if True:
        plot_response('State Space model 3 Step response C = [0, 1, 0]', sys_3_matrix_c_2_step_response[0],
                      sys_3_matrix_c_2_step_response[1])
    if True:
        plot_response('State Space model 3 Step response C = [0, 0, 1]', sys_3_matrix_c_3_step_response[0],
                      sys_3_matrix_c_3_step_response[1])

    if True:
        plot_response('State Space model 1 Sinus response C = [1, 0]', sys_1_matrix_c_1_sin_response[0],
                      sys_1_matrix_c_1_sin_response[1])
    if True:
        plot_response('State Space model 1 Sinus response C = [0, 1]', sys_1_matrix_c_2_sin_response[0],
                      sys_1_matrix_c_2_sin_response[1])

    if True:
        plot_response('State Space model 2 Sinus response C = [1, 0, 0]', sys_2_matrix_c_1_sin_response[0],
                      sys_2_matrix_c_1_sin_response[1])
    if True:
        plot_response('State Space model 2 Sinus response C = [0, 1, 0] ', sys_2_matrix_c_2_sin_response[0],
                      sys_2_matrix_c_2_sin_response[1])
    if True:
        plot_response('State Space model 2 Sinus response C = [0, 0, 1]', sys_2_matrix_c_3_sin_response[0],
                      sys_2_matrix_c_3_sin_response[1])

    if True:
        plot_response('State Space model 3 Sinus response C = [1, 0, 0]', sys_3_matrix_c_1_sin_response[0],
                      sys_3_matrix_c_1_sin_response[1])
    if True:
        plot_response('State Space model 3 Sinus response C = [0, 1, 0]', sys_3_matrix_c_2_sin_response[0],
                      sys_3_matrix_c_2_sin_response[1])
    if True:
        plot_response('State Space model 3 Sinus response C = [0, 0, 1]', sys_3_matrix_c_3_sin_response[0],
                      sys_3_matrix_c_3_sin_response[1])

    return 0

def example_2():
    # Define controllable system matrices
    sys_1_controllable_matrix_a = np.array([[0, 1, 0], [0, 0, 1], [-1 / 6, -1, -11 / 6]])

    sys_1_controllable_matrix_b = np.array([[0], [0], [1]])

    # Create system using state-space representation
    A_matrix = np.array([[-1, 0, 0], [0, -0.5, 0], [0, 0, -1/3]])
    B_matrix = np.array([[1], [0.5], [1/3]])
    D_matrix = 0
    C_matrix = np.array([[1, 0, 0]])
    sys_1_matrix_c_1_lti = scipy.signal.lti(A_matrix, B_matrix, C_matrix, D_matrix)
    # Create controllable system using state-space representation
    sys_1_matrix_c_1_lti_ctr = scipy.signal.lti(sys_1_controllable_matrix_a, sys_1_controllable_matrix_b, C_matrix,
                                            D_matrix)

    # Create system using state-space representation
    C_matrix  = np.array([[0, 1, 0]])
    sys_1_matrix_c_2_lti = scipy.signal.lti(A_matrix, B_matrix, C_matrix, D_matrix)
    # Create controllable system using state-space representation
    sys_1_matrix_c_2_lti_ctr = scipy.signal.lti(sys_1_controllable_matrix_a, sys_1_controllable_matrix_b, C_matrix,
                                            D_matrix)

    # Create system using state-space representation
    C_matrix = np.array([[0, 0, 1]])
    sys_1_matrix_c_3_lti = scipy.signal.lti(A_matrix, B_matrix, C_matrix, D_matrix)
    # Create controllable system using state-space representation
    sys_1_matrix_c_3_lti_ctr = scipy.signal.lti(sys_1_controllable_matrix_a, sys_1_controllable_matrix_b, C_matrix,
                                            D_matrix)

    # Define controllable system matrices
    sys_2_controllable_matrix_a = np.array([[0, 1, 0], [0, 0, 1],
                                          [-1, -5 / 2, -5 / 2]])

    sys_2_controllable_matrix_b = np.array([[0], [0], [1]])

    # Create system using state-space representation
    A_matrix = np.array([[-2, 0, -2], [0, 0, 1], [0.5, -0.5, -0.5]])
    B_matrix = np.array([[2], [0], [0]])
    D_matrix = 0
    C_matrix = np.array([[1, 0, 0]])
    sys_2_matrix_c_1_lti = scipy.signal.lti(A_matrix, B_matrix, C_matrix, D_matrix)
    # Create controllable system using state-space representation
    sys_2_matrix_c_1_lti_ctr = scipy.signal.lti(sys_2_controllable_matrix_a, sys_2_controllable_matrix_b, C_matrix,
                                            D_matrix)

    # Create system using state-space representation
    C_matrix = np.array([[0, 1, 0]])
    sys_2_matrix_c_2_lti = scipy.signal.lti(A_matrix, B_matrix, C_matrix, D_matrix)
    # Create controllable system using state-space representation
    sys_2_matrix_c_2_lti_ctr = scipy.signal.lti(sys_2_controllable_matrix_a, sys_2_controllable_matrix_b, C_matrix,
                                            D_matrix)

    # Create system using state-space representation
    C_matrix = np.array([[0, 0, 1]])
    sys_2_matrix_c_3_lti = scipy.signal.lti(A_matrix, B_matrix, C_matrix, D_matrix)
    # Create controllable system using state-space representation
    sys_2_matrix_c_3_lti_ctr = scipy.signal.lti(sys_2_controllable_matrix_a, sys_2_controllable_matrix_b, C_matrix,
                                            D_matrix)

    time = np.linspace(0, 10, 1000)

    sys_1_step_response_matrix_c_1 = scipy.signal.step(sys_1_matrix_c_1_lti, T=time)
    sys_1_step_response_matrix_c_2 = scipy.signal.step(sys_1_matrix_c_2_lti, T=time)
    sys_1_step_response_matrix_c_3 = scipy.signal.step(sys_1_matrix_c_3_lti, T=time)

    sys_1_step_response_matrix_c_1_ctr = scipy.signal.step(sys_1_matrix_c_1_lti_ctr, T=time)
    sys_1_step_response_matrix_c_2_ctr = scipy.signal.step(sys_1_matrix_c_2_lti_ctr, T=time)
    sys_1_step_response_matrix_c_3_ctr = scipy.signal.step(sys_1_matrix_c_3_lti_ctr, T=time)

    if True:
        plot_response('State Space model 1 Step response C = [1, 0, 0]', sys_1_step_response_matrix_c_1[0],
                      sys_1_step_response_matrix_c_1[1])
    if True:
        plot_response('State Space model 1 Step response C = [0, 1, 0]', sys_1_step_response_matrix_c_2[0],
                      sys_1_step_response_matrix_c_2[1])
    if True:
        plot_response('State Space model 1 Step response C = [0, 0, 1]', sys_1_step_response_matrix_c_3[0],
                      sys_1_step_response_matrix_c_3[1])

    if True:
        plot_response('State Space controllable model 1 Step response C = [1, 0, 0]', sys_1_step_response_matrix_c_1_ctr[0],
                      sys_1_step_response_matrix_c_1_ctr[1])
    if True:
        plot_response('State Space controllable model 1 Step response C = [0, 1, 0]', sys_1_step_response_matrix_c_2_ctr[0],
                      sys_1_step_response_matrix_c_2_ctr[1])
    if True:
        plot_response('State Space controllable model 1 Step response C = [0, 0, 1]', sys_1_step_response_matrix_c_3_ctr[0],
                      sys_1_step_response_matrix_c_3_ctr[1])


def example_3():
    # Define system matrices
    A_matrix = np.array([[0, 1, 0], [0, 0, 1], [-1 / 6, -1, -11 / 6]])
    B_matrix = np.array([[0], [0], [1]])
    C_matrix = np.array([0, 0, 1])
    D_matrix = 0

    # Characteristic equation of a closed-loop system with state feedback:
    # 0 = s^3 + (11/6 + k_3) * s^2 + (1+k_2) * s + (1/6 + k_1)
    # 0 = s^3 + 8s^2 + 17s + 10
    # 0 = (s + 1)(s + 2)(s + 5)
    # 8 = 11/6 + k_3 -> 37/7
    # 17 = 1 + k_2 -> k_2 = 16
    # 10 = 1/6 + k_1 -> k_1 = 59/6

    # Solve for feedback gains
    k_1 = 59 / 6
    k_2 = 16
    k_3 = 37 / 6

    k_matrix = np.array([k_1, k_2, k_3])

    # Create new system matrices with feedback gains
    A_matrix_new = np.array([[0, 1, 0], [0, 0, 1], [-1/6 - k_1, -1 - k_2, -11/6 - k_3]])
    B_matrix_new = np.array([[0], [0], [0]])
    D_matrix = np.array([[0]])

    # Define new output matrix
    C_matrix = np.array([[1, 0, 0]])
    # C_matrix = np.array([[0, 1, 0]])
    # C_matrix = np.array([[0, 0, 1]])

    sys_1 = scipy.signal.StateSpace(A_matrix_new, B_matrix_new, C_matrix, D_matrix)

    # Define time vector and input signal (zero)
    time_vec = np.linspace(0, 10, 1000)
    signal = [0] * len(time_vec)

    # Define initial conditions
    x_0 = (1, 2, 5)

    # Compute system response to input signal with initial conditions
    sys_1_response = scipy.signal.lsim2(sys_1, X0=x_0, U=signal,
                                                T=time_vec)

    # Plot the system response
    if True:
        plot_response('"Zero input response" with non-zero initial conditions. Model 1, C = [0, 0, 1]',
                      sys_1_response[0],
                      sys_1_response[2])

    return 0


def main():
    example_1()
    example_2()
    example_3()
    return 0


if __name__ == "__main__":
    main()
