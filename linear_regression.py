import numpy as np

import math

class LinearRegression:

    def __init__(self, x, y ,alpha=0.001, iter_num=10000):
        
        self.x = x
        self.y = y
        self.alpha = alpha
        self.iter_num = iter_num

        self.m , self.n  = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.j_history = []

        self.gradient_descent()
        

    def  cost_function(self , w , b):

        m = self.m
        f_wb = np.dot(self.x, w) + b   

        total_cost = np.sum((f_wb - self.y) ** 2) / (2 * m)

        return total_cost

    def gradient(self, w, b):
        
        m = self.m
        err = np.dot(self.x, w) + b - self.y   
        
        dj_dw = np.dot(self.x.T, err) / m   
        dj_db = np.sum(err) / m 
        
        return dj_dw, dj_db
    
    def gradient_descent(self):

        for i in range(self.iter_num):
            dj_dw , dj_db = self.gradient(self.w, self.b)

            self.w -= self.alpha * dj_dw
            self.b -= self.alpha * dj_db

            self.j_history.append(self.cost_function(self.w, self.b))

            if i % math.ceil(self.iter_num / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.j_history[-1]:.4f}")

    def regression(self,x_test):

        return np.dot(x_test,self.w) + self.b
