import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

m = [1,2,3,4,9,19]
def f(x):
    return 12*x**2 - 6*x

def test_fn(m):
    basis = np.zeros(m-1)
    for i in range(m-1):
        basis[i] = lambda x: (x**i)* (1 - x) 
    return basis

def basis_derivatives(m, x):
    basis_deriv = np.zeros(m+1)
    for i in range(m+1):
        basis_deriv[i] = lambda x : (i+1)*(x**i)

    return basis_deriv

def test_derivative(m):
    test_deriv = np.zeros(m-1)
    for i in range(m-1):
        test_deriv[i] = lambda x: (i+1)*(x**1) - (i+2)*(x**(i+1))

    return test_deriv

#test_deriv = test_derivative(m)

def lhs(m):
    stiff_matrix_A = np.zeros((m-1, m+1))
    #test_deriv = np.zeros(m-1)
    #basis_deriv = np.zeros(m+1)
    for i in range(1, m):
        
        for j in range(m+1):    
            #basis_deriv = lambda x : (i+1)*(x**i)
            #test_deriv = lambda x: (j+1)*(x**1) - (j+2)*(x**(j+1))

            #lhs_integration = lambda x : basis_deriv*test_deriv
            lhs_integration = lambda x : j*x**(j-1)*(i*x**(i-1)- (i+1)*(x**i))
            stiff_matrix_A[i-1, j], _ = quad(lhs_integration, 0, 1)

    return stiff_matrix_A


def full_stiff_matrix(m):
    stiff_matrix_A = lhs(m)
    full_matrix = np.vstack(( np.zeros((m+1)), stiff_matrix_A, np.ones(m+1)) )
    full_matrix[0,0] = 1
    return full_matrix

#full_matrix = full_stiff_matrix(m)

def rhs(m):
    f_v = np.zeros((m+1))

    for i in range(1,m):
        rhs_integration = lambda x : f(x)* x**(i)*(1-x)
        f_v[i], _ = quad(rhs_integration, 0,1)

    return f_v

#f_v = rhs(m)
#y_coeff = np.linalg.solve(full_matrix, f_v)


def y_approx(x,y):
    return sum(y[i]*x**[i] for i in range(len(y)))

x_val = np.linspace(0,1,500)

for i in m:
    full_matrix = full_stiff_matrix(i)
    f_v = rhs(i)
    y_coeff = np.linalg.solve(full_matrix, f_v)
    y_h = [y_approx(x, y_coeff) for x in x_val]
    plt.plot(x_val, y_h, label='y(x), N = '+str(i))

y_true = lambda x: -x**4 + x**3
plt.plot(x_val, y_true(x_val), label='Exact Solution', color = "black")

plt.title(' monomial basis for y'' = f with N')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
