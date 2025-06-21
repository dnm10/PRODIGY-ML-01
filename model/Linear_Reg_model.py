import numpy as np
# Linear Regression
class Linear_Regression():

    # Initiating the parameters (learning rate & no. of iterations) ;(Implicity) will use inside this class
    def __init__(self, learning_rate, no_of_iterations):

        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

        
    # to fit/train our model with the dataset ; will use outside of this class
    # X -> Experience, Y -> Salary
    def fit(self, X, Y):
        
        # number of training examples => Total_no_of_data_points (m)
        # number of features => Years_of_experience (n)
        
        self.m, self.n = X.shape  # number of rows(m=30) & columns(n=1)

        # initiating the weight and bias 

        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # implementing Gradient Descent

        for i in range(self.no_of_iterations):
            self.update_weights()
        

    # Update the weights ; will use inside this class
    def update_weights(self,):

        Y_prediction = self.predict(self.X)

        # calculate gradients
        # m -> No. of Training Examples, X.T -> Transpose (1row, 30cols)
        dw = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = - 2 * np.sum(self.Y - Y_prediction)/self.m

        # upadating the weights
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db


    # To Predict ; will use outside of this class
    # We will only one parameter as => if we give the value of the no. of years of experience
    # it can calculate the Salary of the person
    def predict(self, X):

        return X.dot(self.w) + self.b # formula => Y =wX + b

        