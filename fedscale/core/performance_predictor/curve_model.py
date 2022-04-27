import numpy as np
from scipy.optimize import curve_fit
from performance_predictor.curve_functions import *



class curving_model:
    def __init__(self, y_train, endingpoint, lambda_k = 7.00e-07):
        """
        input:
        y_train: a numpt array of previous accuracy, shape (N,)
        endingpoint: the prediction point 
        """
        self.lambda_k = lambda_k
        self.y_train = y_train
        self.x_train = np.arange(y_train.shape[0])+1
        self.endingpoint = endingpoint
        self.popts = {}
        self.convergency = []
        self.expectedvalue = []
        self.lowerbound = np.max(y_train)
        self.N = y_train.shape[0]
        self.y_getting = np.zeros((10,self.N))

    def curve_fitting(self):
        ## pow3
        popt, _ = curve_fit(pow3, xdata=self.x_train, ydata=self.y_train, bounds=((self.lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
        self.popts["pow3"] = popt
        self.convergency.append(popt[0])
        self.expectedvalue.append(pow3(self.endingpoint,*popt))
        self.y_getting[0] = pow3(self.x_train,*popt)

        ## pow4
        popt, _ = curve_fit(pow4, xdata=self.x_train, ydata=self.y_train, bounds=((self.lowerbound,-np.inf,-np.inf,-np.inf),(1,np.inf,np.inf,np.inf)))
        self.popts["pow4"] = popt
        self.convergency.append(popt[0])
        self.expectedvalue.append(pow4(self.endingpoint, *popt))
        self.y_getting[1] = pow4(self.x_train, *popt)

        ## log_power
        popt, _ = curve_fit(log_power, xdata=self.x_train, ydata=self.y_train, bounds=((self.lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
        self.popts["log_power"] = popt
        self.convergency.append(popt[0])
        self.expectedvalue.append(log_power(self.endingpoint, *popt))
        self.y_getting[2] = log_power(self.x_train, *popt)

        ## weibull
        popt, _ = curve_fit(weibull, xdata=self.x_train, ydata=self.y_train, bounds=((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)))
        self.popts["weibull"] = popt
        if popt[2] < 0:
            raise ValueError("Error in weibull, kappa not legal")
        if(popt[3]) >= 1:
            self.convergency.append(popt[0])
        elif(popt[3] < 1 and popt[3] > 0):
            self.convergency.append(popt[1])
        else:
            raise ValueError("Error in weibull, delta not legal")
        self.expectedvalue.append(weibull(self.endingpoint, *popt))
        self.y_getting[3] = weibull(self.x_train,*popt)

        ## mmf
        popt, _ = curve_fit(mmf, xdata=self.x_train, data=self.y_train, bounds=((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)))
        self.popts["mmf"] = popt
        if popt[2] < 0:
            raise ValueError("Error in mmf, kappa is not legal")
        if(popt[3]) >= 1:
            self.convergency.append(popt[0])
        elif(popt[3] < 1 and popt[3] > 0):
            self.convergency.append(popt[1])
        else:
            raise ValueError("Error in mmf, delta not legal")
        self.expectedvalue.append(mmf(self.endingpoint, *popt))
        self.y_getting[4] = mmf(self.x_train, *popt)

        ## janoschek
        popt, _ = curve_fit(janoschek, xdata=self.x_train, ydata=self.y_train, bounds=((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)))
        self.popts["janoschek"] = popt
        if popt[2] < 0:
            raise ValueError("Error in janoschek, kappa not legal")
        if(popt[3]) >= 1:
            self.convergency.append(popt[0])
        elif(popt[3] < 1 and popt[3] > 0):
            self.convergency.append(popt[1])
        else:
            raise ValueError("Error in janoschek, delta not legal")
        self.expectedvalue.append(janoschek(self.endingpoint, *popt))
        self.y_getting[5] = janoschek(self.x_train, *popt)

        ## ilog2 
        popt, _ = curve_fit(ilog2, xdata=self.x_train, ydata=self.y_train, bounds=((self.lowerbound,-np.inf),(1,np.inf)))
        self.popts["ilog2"] = popt
        self.convergency.append(popt[0])
        self.expectedvalue.append(ilog2(self.endingpoint, *popt))
        self.y_getting[6] = ilog2(self.x_train, *popt)

        ## exp3
        popt, _ = curve_fit(exp3, xdata=self.x_train, ydata=self.y_train, bounds=((self.lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
        if popt[1]<=0:
            raise ValueError("Error in exp3, a is not legal")
        self.popts["exp3"] = popt
        self.convergency.append(popt[0])
        self.expectedvalue.append(exp3(self.endingpoint, *popt))
        self.y_getting[7] = exp3(self.x_train, *popt)

        ## exp4
        popt, _ = curve_fit(exp4, xdata=self.x_train, ydata=self.y_train, bounds=((self.lowerbound,-np.inf,-np.inf,-np.inf),(1,np.inf,np.inf,np.inf)))
        if popt[1] <= 0:
            raise ValueError("Error in exp4, a not legal")
        if popt[3] <= 0:
            raise ValueError("Error in exp4, alpha not legal")
        self.popts["exp4"] = popt
        self.convergency.append(popt[0])
        self.expectedvalue.append(exp4(self.endingpoint,*popt))
        self.y_getting[8] = exp4(self.x_train,*popt)

        ## dr_hill_zero_background
        popt, _ = curve_fit(dr_hill_zero_background, xdata=self.x_train, ydata=self.y_train, bounds=((self.lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
        self.popts["dr_hill"] = popt
        self.convergency.append(popt[0])
        self.expectedvalue.append(dr_hill_zero_background(self.endingpoint, *popt))
        self.y_getting[9] = dr_hill_zero_background(self.x_train, *popt)
    
    def weights_cal(self):
        A_Left = np.zeros((10,10))
        for i in range(10):
            for j in range(10):
                for k in range(self.N):
                    A_Left[i][j] += self.y_getting[i,k]*self.y_getting[j,k]
        B_right = self.y_getting.dot(self.y_train[np.newaxis].T)
        A_tmp = A_Left + np.eye(10) * self.lambda_k
        self.weights = np.linalg.solve(A_tmp, B_right)
    
    def get_expected_value(self):
        self.curve_fitting()
        self.weights_cal()
        expected_ending = 0
        for value_e in range(10):
            expected_ending += self.expectedvalue[value_e]*self.weights[value_e]
        self.expected_ending = expected_ending
        return expected_ending