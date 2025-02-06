import numpy as np
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import itertools
import pandas as pd
import time
from jax import numpy as jnp

##### BASIC FUNCTIONALITY FOR SIMPLE BANDIT BASE CLASS #####

def relu(x):
    return np.maximum(x, 0)

def H(t):
    return [1/2**(t+4) + k * 1/2**(t+3) for k in range(2**(t+3))]

# length here is half the length of the symmetric confidence interval
def gaussian_ci(bandit, x, length, alpha):
    n = np.ceil(2*np.log(2/alpha)/length**2).astype(np.int32)
    avg = np.mean(np.array([bandit(x) for _ in range(n)]))

    return avg, n

def E(t, x, LCB, L=1):
    exclude_bound = 9/2**(t+4)
    width = relu((LCB-exclude_bound)/L)
    return [x - width, x + width, t]

def element_in_set(x, E):
    for interval in E:
        if x >= interval[0] and x <= interval[1]:
            return True
    return False

def G(H, E):
    pts = []
    for h in H:
        if element_in_set(h, E):
            continue
        pts.append(h)

    return pts
    

def Hnd(r, d, t, leftcorner=None, c=2, t0=3):
    if leftcorner is None:
        leftcorner = 0.5*np.ones((d))
    Ht = []
    ones = np.ones((d))
    for k in np.ndindex(*[math.ceil(r*c**(t+t0)) for _ in range(d)]):
        Ht.append((np.array(k)/c**(t+t0) - leftcorner + ones*c**(-(t+t0+1)))/r)

    return Ht

def next_arms(arms, t_new, d, L):
    G_tnew = []
    for arm in arms:
        directions = itertools.product([-1,1], repeat=d)
        for dir in directions:
            G_tnew.append(arm + np.array(dir)/(L*math.sqrt(d)*2**(t_new + 3)))

    return G_tnew

##### UNOPTIMIZED BASE CLASS FOR BANDIT OPTIMIZER ######

class L2_ND_LBAO:
    def __init__(self, bandit, delta=0.01, 
                 L=1, d=1, interval_factor=1, id=0, threshold=None,
                 threshold_step: int=1):
        self.bandit = bandit
        self.delta = delta
        self.E = []
        self.G = []
        self.pulls = 0
        self.L = L
        self.current_min = None
        self.t = 1
        self.d = d
        self.arms = self.Hf(self.t)
        self.arms_pulled = []
        self.interval_factor = interval_factor
        self.pulls_per_arm = []
        self.save_location = None
        self.id = id
        self.mins = []
        self.threshold = 0 if threshold is None else threshold
        self.threshold_step = threshold_step

    def step(self, state, t):
        t_start = time.time()
        arms = self.arms if t==1 else next_arms(self.arms, t, self.d, self.L)
        self.G.append(arms)
        means = np.zeros((len(arms)))
        length = self.ci_length(t)
        alpha = self.alpha(t)
        arms_keep = []
        for i, h in enumerate(tqdm(arms)):
            means[i], n_i = gaussian_ci(self.bandit, h, length, alpha)
            self.pulls += n_i
            self.arms_pulled.append([h, t])
            if i == 0:
                self.pulls_per_arm.append(n_i)

        self.current_argmin = arms[np.argmin(means)]
        self.current_min = np.min(means)

        for i, h in enumerate(arms):
            if means[i] - length/2 - self.current_min <= 11/2**(t+4):
                arms_keep.append(h)

        self.arms = arms_keep
        t_end = time.time()
        if self.save_location is not None:
            self.save_round(arms, means, self.pulls_per_arm[-1], t_end-t_start)

        return self.current_min, means

    def run(self, state, n_steps=10, save_location=None, last_min=1):
        self.mins = []
        self.save_location = save_location
            
        for k in range(1, n_steps+1):
            self.t = k
            self.current_min, means = self.step(state, k)
            means = jnp.asarray(means)
            if jnp.max(means) - jnp.min(means) < self.threshold and k>=self.threshold_step:
                break
 

        return means, self.pulls
    
    def plot(self):
        if self.d == 1:
            rectangles = []
            eps = 0
            scale = 4000*2**9
            fig, ax = plt.subplots()
            ni_sum = 0
            for tm1, Gt in enumerate(self.G):
                for g in Gt:
                    t = tm1 + 1
                    l = g[0] - 0.5**(t+3)/self.L
                    r = l + 0.5**(t+2)/self.L
                    rectangles.append([Rectangle((l, (ni_sum)/scale + eps), 
                                                r-l, 
                                                self.pulls_per_arm[tm1]/scale + eps)])
                    ax.add_patch(Rectangle((l, (ni_sum)/scale + eps), 
                                                r-l, 
                                                self.pulls_per_arm[tm1]/scale + eps, alpha=0.3, facecolor="purple"))
                ni_sum += self.pulls_per_arm[tm1]
            
            x = np.linspace(0, 1, 1000)
            ax.plot(x, self.bandit.f(x))

            arms = np.array([[arm[0][0], np.sum(np.array(self.pulls_per_arm[:arm[1]]))] for arm in self.arms_pulled])
            ax.scatter(arms[:, 0], arms[:, 1]/scale + eps, c="red")

            if self.current_min is not None:
                ax.axvline(self.current_argmin, linestyle='--')
            plt.show()

        elif self.d == 2:
            raise NotImplementedError
        else:
            raise NotImplementedError
        
    def save_round(self, arms, means, pulls_per_arm, time):
        if self.d == 1:
            arms = [arm[0] for arm in arms]
        data = {"arms": arms, 
                "means": means, 
                "pulls": pulls_per_arm,
                "time": [time for _ in arms]}
        df = pd.DataFrame(data)
        df.to_csv(self.save_location+"/PE_experiment_"+self.id+"_"+str(self.t)+".csv", sep=";")

    def Hf(self, t):
        return Hnd(self.L*math.sqrt(self.d),self.d, t, np.zeros((self.d)))
    
    def alpha(self, t):
        return min(self.delta/(2**(t)*max(len(self.arms), 1)), 1)

    def ci_length(self, t):
        return 2**(-(t+4))
    
    