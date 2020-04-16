import numpy as np
from numpy import linalg as LA
from sympy import *
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.spatial import distance

np.random.seed(0)
random.seed(0)


class classifier(object):
    
    def __init__(self):
        self.name = "classifier"
        pass

    
    def train(self,x1,x2):
        return
        
    def predict(self,x):
        return
    
    def get_name(self):
        return self.name
    
    def test_classifier(self,x1_t,x2_t):
        
        true_prediction = 0
        if len(x1_t) > 0:
            n1 = len(x1_t[0])
        else:
            n1 = 0
        if len(x2_t) > 0:
            n2 = len(x2_t[0])
        else:
            n2 = 0

        for i in range(n1):
                classification = self.predict(x1_t[:,i])
                if classification == 0:
                        true_prediction += 1
        for j in range (n2):
                classification = self.predict(x2_t[:,j])
                if classification == 1:
                        true_prediction += 1
        return (true_prediction/(n1 + n2)) * 100
    
    
class MaximumLiklihood(classifier):
    
    def __init__(self):
        self.name = "ML"
        
    def train(self,x1,x2):
        x1 = x1.T
        x2 = x2.T
        n1 = len(x1)
        d1 = len(x1[0])
        n2 = len(x2)
        d2 = len(x2[0])
        x1 = np.reshape(x1,(n1,d1,1))
        x2 = np.reshape(x2,(n2,d2,1))
        mu_1 = np.zeros((d1,1))
        sigma_1 = np.zeros((d1,d1))
        mu_2 = np.zeros((d2,1))
        sigma_2 = np.zeros((d2,d2))
    #     estimating the mean for x1 and x2
        mu_1 = np.sum(x1,axis = 0)
        mu_1 = mu_1*1/n1
        mu_1 = np.reshape(mu_1,(d1,1))
        mu_2 = np.sum(x2,axis = 0)
        mu_2 = mu_2*1/n2
        mu_2 = np.reshape(mu_2,(d2,1))
    #     estimating sigma for x1 and x2
        for i in range(n1):
            temp = np.dot((x1[i]-mu_1),(x1[i]-mu_1).T)
            sigma_1 = np.add(sigma_1,temp)
        sigma_1 = sigma_1*1/n1
        
        for i in range(n2):
            temp = np.dot((x2[i]-mu_2),(x2[i]-mu_2).T)
            sigma_2 = np.add(sigma_2,temp)
        sigma_2 = sigma_2*1/n2
        self.sigma1 = sigma_1.T
        self.mu1 = mu_1.T[0]
        self.sigma2 = sigma_2.T
        self.mu2 = mu_2.T[0]
        self.a,self.b,self.c = generate_classifier(self.sigma1,self.mu1,self.sigma2,self.mu2,n2/n1)
        
        
    def get_mu_sigma(self):
        return self.sigma1,self.sigma2,self.mu1,self.mu2
    
        
    def predict(self,x):
        
        classification = np.dot(np.dot(np.transpose(x),self.a),x) + np.dot(self.b,x) + self.c
        if classification >= 0:
            return 0 
        else:
            return 1
            
    
        
class Baysian(classifier):
    
    
    def __init__(self,sigma1,sigma2,sigma0,mu0):
        self.name = "Bayesian"
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma0 = sigma0
        self.mu0 = mu0
        
    def train(self,x1,x2):
        x1 = x1.T
        x2 = x2.T
        n1 = len(x1)
        d1 = len(x1[0])
        n2 = len(x2)
        d2 = len(x2[0])
        self.mu_1 = 1/n1 * np.dot(np.dot(self.sigma1, np.linalg.inv(1/n1*self.sigma1 + self.sigma0)),self.mu0) + np.dot(np.dot(self.sigma0,np.linalg.inv(1/n1*self.sigma1 + self.sigma0)),1/n1*np.sum(x1,axis = 0))
        self.mu_2 = 1/n2 * np.dot(np.dot(self.sigma2, np.linalg.inv(1/n2*self.sigma2 + self.sigma0)),self.mu0) + np.dot(np.dot(self.sigma0,np.linalg.inv(1/n2*self.sigma2 + self.sigma0)),1/n2*np.sum(x2,axis = 0))
        self.a,self.b,self.c = generate_classifier(self.sigma1,self.mu_1,self.sigma2,self.mu_2,n1/n2)        
        
    def predict(self,x):    
        classification = np.dot(np.dot(np.transpose(x),self.a),x) + np.dot(self.b,x) + self.c
        if classification >= 0:
            return 0 
        else:
            return 1
        
        
    
class Parzen_window(classifier):
    
    
    def __init__(self,h):
        self.h = h
        self.name = "Parzen"
    
    def train(self,x1,x2):
        self.x1 = x1
        self.x2 = x2
    
    
    
    def predict(self,x):
        correct = 0
        p_x1 = np.zeros((len(self.x1),))
        p_x2 = np.zeros((len(self.x2),))
        n1 = len(self.x1[0])
        n2 = len(self.x2[0])
        D = len(self.x1)
        
        for d in range(D): 
                    
            p_x = 0
            for i in range(n1):

                part1 = 1 / ( ((2* np.pi)**(1/2)) * (self.h))
                part2 = (-1/2) * ((x[d]-self.x1[d,i])/(self.h))**2
                p_x += float(part1 * np.exp(part2))
            p_x1[d] = p_x/n1
                    
        for d in range(D):  
            p_x = 0
            for i in range(n2):
                part1 = 1 / ( ((2* np.pi)**(1/2)) * (self.h))
                part2 = (-1/2) * ((x[d]-self.x2[d,i])/(self.h))**2
                p_x += float(part1 * np.exp(part2))
            p_x2[d] = p_x/n2
            vote = 0    
        for d in range(D):
                if p_x1[d]>p_x2[d]:
                        vote += 1
        if vote >= D/2:
            return 0
        else:
            return 1
        
        
class KNN(classifier):
    
    def __init__(self,k):
        self.k = k
        self.name = "KNN"
    
    def train(self,x1,x2):
        self.x1 = x1
        self.x2 = x2
    
    def predict(self,x):
        second = 0
        first = 0
        k_neighbors_x1 = np.full(self.k,np.inf)
        k_neighbors_x2 = np.full(self.k,np.inf)

        for x1 in self.x1.T:
            dist = euclidean(x , x1)

            if any(dist < k_neighbors for k_neighbors in k_neighbors_x1):
                k_neighbors_x1[np.argmax(k_neighbors_x1)] = dist


        for x2 in self.x2.T:
            dist = euclidean(x , x2)
            if any(dist < k_neighbors for k_neighbors in k_neighbors_x2):
                k_neighbors_x2[np.argmax(k_neighbors_x2)] = dist

        k_neighbors_x1 = np.sort(k_neighbors_x1)
        k_neighbors_x2 = np.sort(k_neighbors_x2)

        for k_n in reversed (range(self.k)):
            if k_neighbors_x1[k_n] < k_neighbors_x2[k_n]:
                first += 1
            else:

                second += 1
        if first > second:
            return 0
        else:
            return 1
        
        
class Perceptron(classifier):
    
    
    def __init__(self,iterations, alpha):
        self.alpha = alpha
        self.iterations = iterations
        self.name = "Perceptron"
        
    
    def train(self,x1,x2):
        self.x1 = x1
        self.x2 = x2
        x1 = x1.T
        x2 = x2.T
        
    #     reverting the x2 values
        _x2 = np.negative(x2)
        X = np.append(x1,_x2,axis = 0)
        np.random.seed(0)
        np.random.shuffle(X)
        alpha = self.alpha
        self.w = np.ones(len(x1[0]))

        for k in range (1,self.iterations):

            alpha = alpha/k

            for x in X:

                if np.dot(x,self.w) < 0:
                    self.w = self.w + alpha * x

    def predict(self,x):
    
        if np.dot(x,self.w) > 0:
            return 0
        else:
            return 1
#         This is not accurate. Didn't find a better way to do this!
        
    def display(self,x2,x4,x5,x6):
        classifier = []
        for x3 in np.arange(100, 200, 0.5):
            V = list(solveset(np.dot(self.w,np.array([y,x2,x3,x4,x5,x6])),y))
            classifier.append(V)
        plt.subplots()
        plt.plot(np.arange(100,200,0.5),classifier,'r--')
        plt.plot(self.x1[2],self.x1[0],'o')
        plt.plot(self.x2[2],self.x2[0],'o')
        
    def display_x3(self,x2,x4,x5,x6):
        classifier = []
        for x1 in np.arange(30, 90, 1):
            V = list(solveset(np.dot(self.w,np.array([x1,x2,y,x4,x5,x6])),y))
            classifier.append(V)
        plt.subplots()
        plt.plot(np.arange(30,90,1),classifier,'r--')
        plt.plot(self.x1[0],self.x1[2],'o')
        plt.plot(self.x2[0],self.x2[2],'o')    
        

class Ho_Kashyap(classifier):
    
    def __init__(self,iterations, alpha):
        self.alpha = alpha
        self.iterations = iterations
        self.name = "Ho_Kashyap"
        
    
    def train(self,x1,x2):
        self.x1 = x1
        self.x2 = x2
        x1 = x1.T
        x2 = x2.T
        _x2 = np.negative(x2)
        X = np.append(x1,_x2,axis = 0)
        np.random.shuffle(X)
        self.w = np.ones((len(X[0]),1))
        b = np.ones((len(X),1))

        for k in range (1,self.iterations):

            e = np.dot(X,self.w) - b
            if all(i>0 for i in e):
                print(e)
                print("Finished by e > 0")
                return w
            b = b + self.alpha * (e + np.absolute(e))
            self.w = self.w + self.alpha*np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),b)
        print("Finished by maximum iterations")

      

    def predict(self,x):
    
        if np.dot(x,self.w) > 0:
            return 0
        else:
            return 1
        
        
    def display(self,x2,x4,x5,x6):
        classifier = []
        for x3 in np.arange(100, 200, 0.5):
            V = list(solveset(np.dot(self.w[:,0],np.array([y,x2,x3,x4,x5,x6])),y))
            classifier.append(V)
        plt.subplots()
        plt.plot(np.arange(100,200,0.5),classifier,'r--')
        plt.plot(self.x1[2],self.x1[0],'o')
        plt.plot(self.x2[2],self.x2[0],'o')
    def display_x3(self,x2,x4,x5,x6):
        classifier = []
        for x1 in np.arange(30, 90, 1):
            V = list(solveset(np.dot(self.w[:,0],np.array([x1,x2,y,x4,x5,x6])),y))
            classifier.append(V)
        plt.subplots()
        plt.plot(np.arange(30,90,1),classifier,'r--')
        plt.plot(self.x1[0],self.x1[2],'o')
        plt.plot(self.x2[0],self.x2[2],'o')
        
class Fisher_descriminate(classifier):
    
    
    def __init__(self):
        self.name = "Fisher_descriminate"
        
        
    
    def train(self,x1,x2):
        self.x1 = x1
        self.x2 = x2
        x1 = x1.T
        x2 = x2.T
        d = len(x1[0])
        mu1 = np.mean(x1,axis = 0)
        mu1 = np.reshape(mu1,(d,1))
        mu2 = np.mean(x2,axis = 0)
        mu2 = np.reshape(mu2,(d,1))
        s1 = np.zeros((d,d))
        s2 = np.zeros((d,d))
        for x in x1:
            x = np.reshape(x,(d,1))
            s1 += np.dot((x-mu1),(x-mu1).T)
        for x in x2:
            s2 += np.dot((x-mu2),(x-mu2).T)

        sw = s1 + s2
        self.V = np.dot(np.linalg.inv(sw),(mu1-mu2))
        
      

    def predict(self,x):
    
        if np.dot(self.V.T,x) > 0:
            return 0
        else :
            return 1
        
        
    def display(self,x2,x4,x5,x6):
        classifier = []
        print(self.V.T[0])
        for x3 in np.arange(0, 5, 0.1):
            V = list(solveset(np.dot(self.V.T[0],np.array([y,x2,x3,x4,x5,x6])),y))
            classifier.append(V)
        plt.subplots()
        plt.plot(np.arange(0,5,0.1),classifier,'r--')
        plt.plot(self.x1[2],self.x1[0],'o')
        plt.plot(self.x2[2],self.x2[0],'o')
    def display_x3(self,x2,x4,x5,x6):
        classifier = []
        for x1 in np.arange(-10, 0, 0.1):
            V = list(solveset(np.dot(self.V.T[0],np.array([x1,x2,y,x4,x5,x6])),y))
            classifier.append(V)
        plt.subplots()
        plt.plot(np.arange(-10,0,0.1),classifier,'r--')
        plt.plot(self.x1[0],self.x1[2],'o')
        plt.plot(self.x2[0],self.x2[2],'o')




def generate_Gaussian_random_vector(mu,sigma,d,n):
    
    generated_points = np.zeros((d,n))
    
    for j in range(d):
        r = 0
        for z in range(12):
            r = r + np.random.uniform(0,1,n)
        r = r - 6
        generated_points[j,:] = r
    w, v = LA.eig(sigma)
    q = np.dot(np.power(np.diag(w),1/2),generated_points) 
    x = np.dot(v,q)
    for i in range (n):
        x[:,i] = x[:,i] + mu   
    return x


def get_data():
    data = pd.read_csv("dataset.csv")
    data = np.delete(np.array(data),[2,5,6,8,10,11,12],axis = 1)
    data[np.where(data[:,-1]>0),-1] = 1
    return data


def diagonalize(x1,w1,v1,x2,w2,v2,sigma1,sigma2,mu1,mu2):
    

    sigma_y = np.dot(np.transpose(v1),sigma1)
    sigma_y2 = np.dot(np.transpose(v1),sigma2)
    sigma_y = np.dot(sigma_y,v1)
    sigma_y2 = np.dot(sigma_y2,v1)
    print("The sigma for x1 after digonalizing")
    print(sigma_y)
    sigma_z = np.dot(np.diag(np.power(w1,-1/2)),sigma_y)
    sigma_z2 = np.dot(np.diag(np.power(w1,-1/2)),sigma_y2)
    sigma_z = np.dot(sigma_z,np.diag(np.power(w1,-1/2)))
    sigma_z2_ = np.dot(sigma_z2,np.diag(np.power(w1,-1/2)))
    print("The sigma for x1 after converting to identitiy")
    print(sigma_z)
    w_z, v_z = LA.eig(sigma_z2_)
    sigma_v2 = np.matmul(np.transpose(v_z),sigma_z2_)
    sigma_v2_ = np.matmul(sigma_v2,v_z)
    sigma_v1 = np.matmul(np.transpose(v_z),sigma_z)
    sigma_v1 = np.matmul(sigma_v1,v_z)
    print("the sigma for x2 after diagonalizing")
    print(sigma_v2_)
    print(w_z)
    print("The sigma for x1 after all the changes")
    print(sigma_v1)
    y1 = np.matmul(np.transpose(v1),x1) 
    mu1 = np.matmul(np.transpose(v1),mu1)
    y2 = np.matmul(np.transpose(v1),x2)
    mu2 = np.matmul(np.transpose(v1),mu2)
    z1 = np.matmul(np.diag(np.power(w1,-1/2)),y1)
    mu1 = np.matmul(np.diag(np.power(w1,-1/2)),mu1)
    z2 = np.matmul(np.diag(np.power(w1,-1/2)),y2)
    mu2 = np.matmul(np.diag(np.power(w1,-1/2)),mu2)
    q1 = np.matmul(np.transpose(v_z),z1)
    mu1 = np.matmul(np.transpose(v_z),mu1)
    q2 = np.matmul(np.transpose(v_z),z2)
    mu2 = np.matmul(np.transpose(v_z),mu2)
    print("checking")
    sigma_z2_ei = np.dot(np.transpose(v1),v2)
    sigma_z2_ei = np.dot(np.diag(np.power(w1,-1/2)),sigma_z2_ei)
    sigma_v2 = np.dot(np.transpose(sigma_z2_ei),sigma_z2_)
    print(sigma_v2_)
    
    return q1,q2,sigma_v1,sigma_v2_,mu1,mu2,v_z


    
    
def generate_classifier(sigma1,mu1,sigma2,mu2,p2_p1 = 1):
    A = (np.linalg.inv(sigma2) - np.linalg.inv(sigma1))
    B = (np.dot(np.transpose(mu1),np.linalg.inv(sigma1)) - np.dot(np.transpose(mu2),np.linalg.inv(sigma2)))
    C = 1/2 * np.log(np.linalg.det(sigma2)/np.linalg.det(sigma1)) + np.log(p2_p1)+(np.dot(np.dot(np.transpose(mu2),np.linalg.inv(sigma2)),mu2) - np.dot(np.dot(np.transpose(mu1), np.linalg.inv(sigma1)),mu1))* 1/2
    return A,B,C

from sympy import *
x, y, z, t = symbols('x y z t')



def get_points(a,b,d,x3):
    first_classifier = []
    second_classifier = []
    x1_points = []
    
    for x1 in np.arange(-100, 100, 0.5):
        v =  list(solveset(Eq(a[0,0]*(x1**2) + a[1,1]*(y**2) + a[2,2]*(x3**2) + (a[0,1]+a[1,0])*x1*y+ (a[0,2]+a[2,0])*x1*x3 + (a[1,2]+a[2,1])*y*x3 + (b[0]*x1) + (b[1]*y) + (b[2] * x3) + d,0),y))
        if type(v[0])is not Add:
            if len(v) > 1 : 
                first_classifier.append(v[0])
                second_classifier.append(v[1])
                x1_points.append(x1)
#                 print(a)
            else:
                x1_points.append(x1)
                first_classifier.append(v[0])
                second_classifier.append(v[0])
#                 print(a)
        
#         print("next point")
    return first_classifier,second_classifier,x1_points


def get_points_6dimensions(a,b,d,x2,x4,x5,x6):
    first_classifier = []
    second_classifier = []
    x1_points = []
    
    for x1 in np.arange(30, 90, 1):
        v =  list(solveset(np.dot(np.dot(np.transpose(np.array([x1,x2,y,x4,x5,x6])),np.transpose(a)),np.array([x1,x2,y,x4,x5,x6])) + np.dot(b,np.array([x1,x2,y,x4,x5,x6])) + d,y))
        print(v)
        if type(v[0])is not Add:
            if len(v) > 1 : 
                first_classifier.append(v[0])
                second_classifier.append(v[1])
                x1_points.append(x1)
#                 print(a)
            else:
                x1_points.append(x1)
                first_classifier.append(v[0])
                second_classifier.append(v[0])
#                 print(a)
        
#         print("next point")
    return first_classifier,second_classifier,x1_points




def display_bayes_descriminate_func(x1,x2, z1,z2,z3,lim,i,j):
    plt.subplots()
    plt.plot(x1[i], x1[j], 'o',label = "x1")
    plt.plot(x2[i], x2[j], 'o', label = "x2")
    plt.plot(z3,z1,'r--',label = "classifier")
    plt.plot(z3,z2,'r--',label = "classifier")
    plt.xlim(0,lim)
    plt.ylim(0,lim)
    plt.legend()
    

def display_x3(x1,x2, z1,z2,z3,lim):
    plt.subplots()
    plt.title("First and second dimension")
    plt.plot(x1[0], x1[1], 'o',label = "x1")
    plt.plot(x2[0], x2[1], 'o', label = "x2")
    plt.plot(z3,z1,'r--',label = "classifier")
    plt.plot(z3,z2,'r--',label = "classifier")
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.legend()

    

def display_x1(x_1_,x_2_, z_1_,z_2_,z_3_,lim):
    plt.subplots()
    plt.title("second and third dimension")
    plt.plot(x_1_[1], x_1_[2], 'o',label = "x1")
    plt.plot(x_2_[1], x_2_[2], 'o', label = "x2")
#     plt.plot(z_3_,z_1_,'r--',label = "classifier")
    plt.plot(z_3_,z_2_,'r--',label = "classifier")
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.legend()
    


def baysian_n_points(x1,x2,sigma1,sigma2,true_mu1,true_m2,sigma0,mu0):
    x1 = x1.T
    x2 = x2.T
    n1 = len(x1)
    d1 = len(x1[0])
    n2 = len(x2)
    d2 = len(x2[0])
    mu1_points = []
    for n in range(1,n1+1):
        mu1 = 1/n * np.dot(np.dot(sigma1, np.linalg.inv(1/n*sigma1 + sigma0)),mu0) + np.dot(np.dot(sigma0,np.linalg.inv(1/n*sigma1 + sigma0)),1/n*np.sum(x1[:n],axis = 0))
        mu1_points.append(distance.euclidean(mu1, true_mu1))
#     mu2 = 1/n2 * np.dot(np.dot(sigma2, np.linalg.inv(1/n2*sigma2 + sigma0)),mu0) + np.dot(np.dot(sigma0,np.linalg.inv(1/n2*sigma2 + sigma0)),1/n2*np.sum(x2,axis = 0))
    return mu1_points

def euclidean(x1,x2):
    distance = 0
    for x,y in zip(x1,x2):
# 
        distance += (x-y)**2

        
    return np.sqrt(np.sum(distance))
        



def ML_points(x1,x2,true_sigma1,true_mu1):
    x1 = x1.T
    x2 = x2.T
    n1 = len(x1)
    d1 = len(x1[0])
    n2 = len(x2)
    d2 = len(x2[0])
    mu1_points = []
    sigma1_points = []
    for n in range(1,n1+1):
        sigma_1,mu_1,_,_ = maximumLiklihood(x1[:n].T,x2.T)
        mu1_points.append(euclidean(mu_1,true_mu1))
        sigma1_points.append(euclidean(sigma_1,true_sigma1))

    return sigma1_points,mu1_points

def plot_points(points, n):
    x_axis = np.arange(n)
    plt.plot(x_axis,points,'r--')
    plt.show()
    


def parzen_window_prob_density(x1,x2,h):
    
#     for x1
    x1_dist = np.zeros((len(x1),len(x1[0])))
    x2_dist = np.zeros((len(x2),len(x2[0])))
    x_axis = []
    x_index = 0
    print(len(x1[0]))
    for d in range(len(x1)):
        for x in x1[d]:        
            p_x = 0          
            for i in range(len(x1[d])):
                part1 = 1 / ( ((2* np.pi)**(1/2)) * (h))
                part2 = (-1/2) * (((x1[d,i]-x)/(h))**2)
                p_x += float(part1 * np.exp(part2))
                
            x1_dist[d,x_index] = p_x/((len(x1[0]))) 
            x_index += 1
        x_index = 0
        x1_dist[d] = x1_dist[d]/np.sum(x1_dist[d])
   
    
    x_index = 0
    for d in range(len(x2)):
        for x in x2[d]:
            p_x = 0
            for i in range(len(x2[0])):
                part1 = 1 / ( ((2* np.pi)**(1/2)) * (h))
                part2 =(-1/2) * (((x2[d,i]-x)/(h))**2)
                p_x += float(part1 * np.exp(part2))
            
            x2_dist[d,x_index] = p_x/(len(x2[0]))
            x_index += 1
        x_index = 0
        x2_dist[d] = x2_dist[d]/np.sum(x2_dist[d])
    return x1_dist,x2_dist


def plot_univariate_prob(x1_dist,x2_dist,interval_length):
    
    
    x_axis = np.arange(0,interval_length,0.5)
    for d in range(len(x1_dist)):
        plt.subplots()
        plt.title(f" dimension {d}")
        plt.plot(x_axis, x1_dist[d,:], 'r--',label = "x1")
        plt.plot(x_axis, x2_dist[d,:], 'b--',label = "x2")
        plt.legend()
        mean1 = np.multiply(x_axis,x1_dist[d,:])
        mean1 = np.sum(mean1,axis=0)
        mean1 = mean1/len(x1_dist[0])
        mean2 = x_axis * x2_dist[d,:]
        mean2 = np.sum(mean2)
        mean2 = mean2/len(x2_dist[0])
        varience1 = np.zeros((x1_dist.shape))
        varience2 = np.zeros((x2_dist.shape))
        for i in range (len(x1_dist[0])):
            varience1[0,i] = (x1_dist[d,i]-mean1)**2
            varience2[0,i] = (x2_dist[d,i]-mean2)**2
        print(f"the summ of x1_dist is {np.sum(x1_dist[2])}")
        print(f'The sample mean for x1, first dimension is {mean1}')
        print(f'The sample varience for x1, first dimension is {np.sum(varience1[0,:])}')
        print(f'The sample mean for x2, first dimension is {mean2}')
        print(f'The sample varience for x2, first dimension is {np.sum(varience2[0,:])}')
    


def K_cross_validation(K,clf ,X):

    n = len(X)
    d = len(X[0])
    partition = int(n/K)
    average = 0
    i = 0
    for k in range(K):
        max_len = min(i+partition,n)
        test = X[i:max_len]
        test1 = test[np.where(test[:,-1] == 0)]
        test2 = test[np.where(test[:,-1] == 1)]
        test1 = test1[:,:-1]
        test2 = test2[:,:-1]
        indices = np.arange(i,max_len)
        train = np.delete(X,indices,axis=0)
        train1 = train[np.where(train[:,-1] == 0)]
        train1 = train1[:,:-1]
        train2 = train[np.where(train[:,-1] == 1)]
        train2 = train2[:,:-1]
        clf.train(train1.T,train2.T)
        test_acc = clf.test_classifier(test1.T,test2.T)
        i = i + partition
        print(f'The accuracy for the {clf.get_name()} in the {k} set is : {test_acc}')
        average += test_acc
    print(f"The average accuracy for the {clf.get_name()} is {average/K}")
    
    

        

    

        
    
        
    

    
