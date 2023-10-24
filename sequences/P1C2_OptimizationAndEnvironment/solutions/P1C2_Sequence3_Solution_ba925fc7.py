
def compute_A_b_solve_v(p_e, p_x, p_s, gamma, tau_new, tau_eat, verbose=True):
  # Summation terms for tau_new and tau_eat
  sum_new = sum([(1-p_s)**i * gamma**i for i in range(tau_new)])
  sum_eat = sum([(1-p_s)**i * gamma**i for i in range(tau_eat)])

  # Coefficients for the matrix A and vector b
  a11 = p_e * (1-p_s)**(tau_new) * gamma**(tau_new+1) + (1-p_e) * gamma**(tau_new+1)
  a12 = p_e * p_s * gamma * sum_new
  b1  = p_e * p_s * sum_new

  a21 = (1 - p_x) * (1-p_s)**tau_eat * gamma**(tau_eat+1) + p_x * gamma**(tau_eat+1)
  a22 = (1 - p_x) * p_s * gamma * sum_eat
  b2  = (1 - p_x) * p_s * sum_eat

  A = np.array([[a11, a12],
                [a21, a22]])

  b = np.array([b1, b2])

  I = np.identity(2)

  if verbose:
    print("Matrix A:")
    print(A)
    print("\nVector b:")
    print(b)
    print("\nMatrix I - A:")
    print(I - A)

  # Solve the system of equations
  v = np.linalg.solve(I - A, b)

  return v[0], v[1]

# Parameters from earlier simulations
p_e = 0.5
p_x = 0.2
p_s = 0.9
gamma = (1 - 0.05)
tau_new = 1
tau_eat = 0

V_NP, V_SE = compute_A_b_solve_v(p_e, p_x, p_s, gamma, tau_new, tau_eat)
print("\nV(NP) =", V_NP)
print("V(SE) =", V_SE)