import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy
from sympy import symbols, lambdify, solve
from sympy.physics.mechanics import ReferenceFrame, dynamicsymbols, init_vprinting
from scipy.optimize import least_squares, root
from scipy.interpolate import CubicSpline
import pandas as pd
from pymycobot.mypalletizer260 import MyPalletizer260
from pymycobot import utils
import time

def fk(alpha_val: float, beta_val: float, lengths: list)-> np.array:
    l1, l2, x, y = symbols('l1, l2, x, y')
    alpha, beta = dynamicsymbols('alpha, beta')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [alpha, N.z])
    B = A.orientnew('B', 'Axis', [-beta, A.z])

    params = {l1: lengths[0], l2: lengths[1]}

    r1 = l1 * A.x
    r2 = l2 * B.x
    rp = x * N.x + y * N.y

    loop = r1 + r2 - rp
    eqList = [loop.dot(N.x), loop.dot(N.y)]
    sol = solve(eqList, [x, y])
    x_pos, y_pos = sol[x], sol[y]

    Px_f = sympy.lambdify([alpha, beta], x_pos.subs(params))
    Py_f = sympy.lambdify([alpha, beta], y_pos.subs(params))

    x_result = Px_f(alpha_val, beta_val)
    y_result = Py_f(alpha_val, beta_val)

    return np.array([x_result, y_result])

def residuals(chi: list, lengths: list, target: list):
        return fk(chi[0], chi[1], lengths) - target

def ik_least_squares(target: list, lengths: list, initial_guess: list, bounds: tuple)-> tuple:

    target = np.asarray(target, dtype=float)
    initial_guess = np.asarray(initial_guess, dtype=float)
    sol = least_squares(residuals, x0=initial_guess, bounds=bounds, args=(lengths, target))
    if not sol.success:
       raise ValueError(f"IK no convergió para target {target}: {sol.message}")
    return sol.x

def ik_root(target: list, lengths: list, initial_guess: list):
    target = np.asarray(target, dtype=float)
    initial_guess = np.asarray(initial_guess, dtype=float)
    sol = root(residuals, x0=initial_guess, args=(lengths, target))
    if not sol.success:
        raise ValueError(f"IK no convergió para target {target}: {sol.message}")
    return sol.x

data = pd.read_csv("data star.csv")
posiciones = data[["x_m", "y_m"]]
posiciones["x_m"] = posiciones["x_m"] - 0.034
posiciones["y_m"] = posiciones["y_m"] + 0.05 + 0.03
posiciones_num = posiciones.to_numpy()
print("posiciones_num:", posiciones_num[:5])  # Muestra las primeras 5 posiciones para verificar

def posiciones_a_angulo(posiciones: np.array, lengths: list, initial_guess: list, bounds: tuple)-> np.array:
    angulos = []
    print("posiciones:", posiciones.shape)
    print("initial_guess:", np.shape(initial_guess))
    print("bounds:", bounds)
    guess = initial_guess
    for pos in posiciones:
        sol = ik_least_squares(pos, lengths, guess, bounds)
        #sol = ik_root(pos, lengths, guess)
        angulos.append(sol)
        guess = sol
    return np.array(angulos)

angulos = posiciones_a_angulo(posiciones_num, [0.13, 0.13], [np.pi/4, np.pi/4], ([0, np.deg2rad(10)], [np.deg2rad(90), np.deg2rad(110)])) #([0, np.deg2rad(30)], [np.deg2rad(90), np.deg2rad(90)])
angulos[:, 0] = np.pi/2 - angulos[:, 0]
angulos[:, 1] -= np.pi/2
angulos = np.rad2deg(angulos)
print("angulos:", angulos[:5])  # Muestra los primeros 5 ángulos para verificar

def build_trajectory(angles, duration, unwrap=True):
    """
    angles:   array (n_waypoints, n_brazos), ej. (180, 2)
    duration: duracion real del movimiento en segundos
    Devuelve: (spline, t_nodes)
    """
    angles = np.asarray(angles, dtype=float)
    if unwrap:
        angles = np.unwrap(angles, axis=0)
 
    # parametro por distancia angular acumulada (evita nodos duplicados con el max)
    d = np.maximum(np.linalg.norm(np.diff(angles, axis=0), axis=1), 1e-9)
    t = np.concatenate([[0.0], np.cumsum(d)])
    t = t / t[-1] * duration  # reescala a segundos [0, duration]
 
    spline = CubicSpline(t, angles, axis=0, bc_type="not-a-knot")
    return spline, t
 
 
def evaluate(spline, t):
    return spline(t)

pybot = MyPalletizer260('COM3')
pybot.is_power_on()
initial_angles = pybot.get_angles()
print("Initial angles:", initial_angles)
pybot.send_angles([initial_angles[0], 0, 0, initial_angles[3]], 50)

kp = float(input('Constante de error proporcional del palletizer: '))
n_points = int(input('Número de posiciones de la trayectoria: '))
duration = float(input('Duración de la trayectoria:'))

spline, t_nodes = build_trajectory(angulos, duration)
err_1, err_2 = 0, 0
t_now = 0.0
dt=duration/(n_points-1)

while t_now <= t_nodes[-1]:
    theta = evaluate(spline, t_now)
    alpha, beta = theta[0], theta[1]
    pybot.send_angles([initial_angles[0], float(alpha) + (err_1 * kp), float(beta) + (err_2 * kp), initial_angles[3]], 20)
    time.sleep(0.5)
    i_pos = pybot.get_angles()
    erf = [float(alpha) - float(i_pos[1]), float(beta) - float(i_pos[2])]
    err_1, err_2 = erf[0], erf[1]
    t_now += dt