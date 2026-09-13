import numpy as np
import pandas as pd
from pymycobot.mypalletizer260 import MyPalletizer260

l1= 0.13
l2= 0.134
l3= 0.05
l4= 0.034

#Formar lista de tuplas

def inverse_kinematics(x, y, l1, l2, l3, l4):

    X = x - l4
    Y = y + l3

    r2 = X**2 + Y**2

    sin_theta2 = (l1**2 + l2**2 - r2) / (2 * l1 * l2)

    # No hay solución
    if abs(sin_theta2) > 1:
        return []

    sin_theta2 = np.clip(sin_theta2, -1, 1)

    theta2_1 = np.arcsin(sin_theta2)
    theta2_2 = np.pi - theta2_1

    solutions = []

    alpha = np.arctan2(Y, X)

    for theta2 in [theta2_1, theta2_2]:

        A = l2 * np.cos(theta2)
        B = l1 - l2 * np.sin(theta2)

        beta = np.arctan2(B, A)

        theta1 = beta - alpha

        # Normalizar a [-pi, pi]
        theta1 = (theta1 + np.pi) % (2*np.pi) - np.pi
        theta2 = (theta2 + np.pi) % (2*np.pi) - np.pi

        solution = [theta1, theta2]

        if not any(np.allclose(solution, s) for s in solutions):
            solutions.append(solution)

    return solutions

df = pd.read_csv("data star.csv")

#Calcular lista de tuplas de los angulos

angulos = []
sol_previous = np.array([0.0, 0.0])

k=1.4
offsetx=-0.08
offsety=0.1

for _, row in df.iterrows():

    x = k*row["x_m"]+offsetx# k estira los puntos y offsetx desplaza el origen
    y = k*row["y_m"]+offsety

    solutions = inverse_kinematics(x, y, l1, l2, l3, l4)

    # Si no existe solución
    if len(solutions) == 0:
        angulos.append((np.nan, np.nan))
        print(f"No existe solución para x={x}, y={y}")
        continue

    # Si solo existe una solución
    if len(solutions) == 1:

        sol_correcta = np.array(solutions[0])

    # Si existen dos soluciones
    else:

        sol1 = np.array(solutions[0])
        sol2 = np.array(solutions[1])

        if np.linalg.norm(sol1 - sol_previous) < \
           np.linalg.norm(sol2 - sol_previous):

            sol_correcta = sol1

        else:

            sol_correcta = sol2

    # Guardar como tupla
    angulos.append(
        (float(np.rad2deg(sol_correcta[0])), float(np.rad2deg(sol_correcta[1])))
    )

    # La solución actual será la referencia para el siguiente punto
    sol_previous = sol_correcta


#Queda guardado en angulos

#Graficar angulos 



#Se lo mando al robot:

'''
import time

pybot = MyPalletizer260('COM4')  # Cambiar al puerto correcto

initial_angles = pybot.get_angles()
for angle in angulos:
    pybot.send_angles([initial_angles[0], angle[0], angle[1], initial_angles[3]], 40)
    time.sleep(0.5)  # Esperar medio segundo entre movimientos
    print(pybot.get_angles())  # Imprimir los ángulos actuales del robot

'''