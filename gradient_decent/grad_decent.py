import numpy as np
import pandas as pd
import math
def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.00001

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = [(1/n) * sum([val**2 for val in (y - y_predicted)])]
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        for c in cost:
            math.isclose(c, 0, rel_tol=1e-20 )
        if i % 100 == 0:
            print(f"Iteration {i}: m={m_curr}, b={b_curr}, cost={cost}")
    return m_curr, b_curr
if __name__ == "__main__":
    df=pd.read_csv('D:/Data Science/gradient_decent/test_scores.csv')
    x = df.math
    y = df.cs
    m, b = gradient_descent(x, y)
    print(f"Final parameters: m={m}, b={b}")