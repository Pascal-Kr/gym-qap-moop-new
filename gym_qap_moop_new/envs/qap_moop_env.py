import numpy as np
import gym
import gym_flp
from gym import spaces
from numpy.random import default_rng
import pickle
import os
import math
import matplotlib.pyplot as plt
from PIL import Image
from gym_flp import rewards
from IPython.display import display, clear_output
import anytree
from anytree import Node, RenderTree, PreOrderIter, LevelOrderIter, LevelOrderGroupIter
from gym_qap_moop_new.Helpfunctions.Helpfunctions import Functions
    

class QAPMOOPenv(gym.Env):

    def __init__(self):
        self.transport_intensity = None
        
        Distances = Functions()
        Flow = Functions()
        self.F = Flow.Flowmatrix()
        self.n = len(self.F[0])
        self.x = math.ceil((math.sqrt(self.n)))
        self.Noisedummy = Functions()
        self.Noise1 = self.Noisedummy.Noisematrix()
        self.Factory_Length = 60
        self.Factory_Width = 60
        self.Measuring_points = 17
        Machine_centers = Functions()
        self.centers_x, self.centers_y = Machine_centers.Machine_centers(self.Factory_Length, self.Factory_Width, self.x)
        self.D = Distances.Distancematrixnew(self.centers_x, self.centers_y)
        
        # Determine problem size relevant for much stuff in here:
        self.n = len(self.D[0])
        self.x = math.ceil((math.sqrt(self.n)))
        self.y = math.ceil((math.sqrt(self.n)))
        self.size = int(self.x*self.y)
        self.observation_space_values=(self.x,self.y,3)
        self.max_steps = self.n - 1

        self.action_space = spaces.Discrete(int((self.n**2-self.n)*0.5)+1)
        self.observation_space = spaces.Box(low = 0, high = 255, shape=(1, self.n, 3), dtype = np.uint8) # Image representation
        self.actions = self.pairwiseExchange(self.n)
        
        # Initialize Environment with empty state and action
        self.action = None
        self.state = None
        self.internal_state = None
        
        #Initialize moving target to incredibly high value. To be updated if reward obtained is smaller. 
        self.movingTargetRewardMHC = np.inf 
        self.movingTargetRewardReturnflow = np.inf 
        self.movingTargetRewardNoise_score = np.inf
        self.Actual_MHCmin = np.inf
        self.Actual_Returnflowmin = np.inf
        self.Actual_Noisemin = np.inf
 
    
        self.MHC = Functions()
        self.Returnflow = Functions()
        self.Reward = Functions() 
    
    def reset(self):
        self.step_counter = 0  #Zählt die Anzahl an durchgeführten Aktionen
        self.state_1D = default_rng().choice(range(1,self.n+1), size=self.n, replace=False) 
        
        self.internal_state = self.state_1D.copy()
        self.fromState = self.internal_state.copy()
        newState = self.fromState.copy()
        MHC, self.TM = self.MHC.computeMHC(self.D, self.F, newState)
        Returnflow, self.Flowmatrix_new = self.Returnflow.computeReturnflow(self.F, newState)
        self.Noisepositions = self.Noisedummy.computeNoise(self.Noise1, newState)
        Noise_measuring_points, Noise_average = self.Noisedummy.Noisecalculation(self.Factory_Length, self.Factory_Width, self.centers_x, self.centers_y, self.Noisepositions, self.n, self.Measuring_points)
        self.Noise_Areas_Values2, self.Noisemin2, self.Noisemax2 = self.Noisedummy.Noise_Areas2(Noise_measuring_points, self.n, self.Measuring_points)
        self.Under55, self.Under60, self.Under65, self.Under70, self.Under75, self.Under80, self.Under85, self.Over85  = self.Noisedummy.Noiseintervals(self.Noise_Areas_Values2)
        Noise_score = self.Noisedummy.Noise_interval_score(self.Under55, self.Under60, self.Under65, self.Under70, self.Under75, self.Under80, self.Under85, self.Over85)
        
        state_2D = np.array(self.get_image())
        
        self.initial_MHC = MHC
        self.last_MHC = self.initial_MHC
        self.transformedMHC = ((MHC-self.MHCmin)/(self.MHCmax-self.MHCmin))
        self.last_transformedMHC = self.transformedMHC 
        
        self.initial_Returnflow = Returnflow
        self.last_Returnflow = self.initial_Returnflow
        self.transformedReturnflow = ((Returnflow-self.Returnflowmin)/(self.Returnflowmax-self.Returnflowmin))
        self.last_transformedReturnflow = self.transformedReturnflow
        
        self.initial_Noise_measuring_points = Noise_measuring_points
        self.last_Noise_measuring_points = self.initial_Noise_measuring_points 
        self.initial_Noise_score = Noise_score
        self.last_Noise_score = self.initial_Noise_score
        self.transformedNoise_score = ((Noise_score-self.Noise_scoremin)/(self.Noise_scoremax-self.Noise_scoremin))
        self.last_transformedNoise_score = self.transformedNoise_score  
        return state_2D
    
    def step(self, action):
        # Create new State based on action 
        self.step_counter += 1 
        
        self.fromState = self.internal_state.copy()
        
        swap  = self.actions[action]
        self.fromState[swap[0]-1], self.fromState[swap[1]-1] = self.fromState[swap[1]-1], self.fromState[swap[0]-1]
        
        newState = self.fromState.copy()
        MHC, self.TM = self.MHC.computeMHC(self.D, self.F, newState)
        Returnflow, self.Flowmatrix_new = self.Returnflow.computeReturnflow(self.F, newState)
        self.Noisepositions = self.Noisedummy.computeNoise(self.Noise1, newState)
        Noise_measuring_points, Noise_average = self.Noisedummy.Noisecalculation(self.Factory_Length, self.Factory_Width, self.centers_x, self.centers_y, self.Noisepositions, self.n, self.Measuring_points)        
        self.Noise_Areas_Values2, self.Noisemin2, self.Noisemax2 = self.Noisedummy.Noise_Areas2(Noise_measuring_points, self.n, self.Measuring_points)
        self.Under55, self.Under60, self.Under65, self.Under70, self.Under75, self.Under80, self.Under85, self.Over85  = self.Noisedummy.Noiseintervals(self.Noise_Areas_Values2)
        Noise_score = self.Noisedummy.Noise_interval_score(self.Under55, self.Under60, self.Under65, self.Under70, self.Under75, self.Under80, self.Under85, self.Over85)
        
        if self.movingTargetRewardMHC == np.inf:
            self.movingTargetRewardMHC = MHC                    
        self.transformedMHC = ((MHC-self.MHCmin)/(self.MHCmax-self.MHCmin))
        self.MHCreward= (self.last_transformedMHC - self.transformedMHC)*100
        self.last_transformedMHC = self.transformedMHC   
        if MHC <= self.movingTargetRewardMHC:
            self.MHCreward +=10
            self.movingTargetRewardMHC = MHC       


        if self.movingTargetRewardReturnflow == np.inf:
            self.movingTargetRewardReturnflow = Returnflow          
        #self.last_Returnflow = Returnflow
        self.transformedReturnflow = ((Returnflow-self.Returnflowmin)/(self.Returnflowmax-self.Returnflowmin))
        self.Returnflowreward= (self.last_transformedReturnflow - self.transformedReturnflow)*100
        #self.Returnflowreward= (self.last_Returnflow-Returnflow)
        self.last_transformedReturnflow = self.transformedReturnflow          
        self.last_Returnflow = Returnflow
        if Returnflow <= self.movingTargetRewardReturnflow:
            self.Returnflowreward +=10
            self.movingTargetRewardReturnflow = Returnflow
     
        
        if self.movingTargetRewardNoise_score == np.inf:
            self.movingTargetRewardNoise_score = Noise_score            
        #self.last_Noise_score = Noise_score        
        self.transformedNoise_score = ((Noise_score-self.Noise_scoremin)/(self.Noise_scoremax-self.Noise_scoremin))        
        self.Noiserewardintervals= (self.last_transformedNoise_score - self.transformedNoise_score)*100
        #self.Noiserewardintervals= self.last_Noise_score-Noise_score
        self.last_transformedNoise_score = self.transformedNoise_score
        
        if Noise_score<=self.movingTargetRewardNoise_score:
            self.Noiserewardintervals +=10
            self.movingTargetRewardNoise_score = Noise_score
        #elif (Noise_score-self.last_Noise_score)>0:
         #   self.Noiserewardintervals-=5
        #self.last_Noise_score = Noise_score
        
        reward = self.Reward.computeTotalreward(self.MHCreward, self.Returnflowreward, self.Noiserewardintervals)
        
        newState = np.array(self.get_image())
        self.state = newState.copy()
        self.internal_state = self.fromState.copy()
        
        if self.step_counter==self.max_steps:
            done = True
        else:
            done = False
        
        return newState, reward, done, {'MHC': MHC, 'Return Flow': Returnflow,'Noise': Noise_score}
    
    def render(self, mode=None):
        if self.mode == 'rgb_array':
            #img = Image.fromarray(self.state, 'RGB')     
            img = self.get_image()

        
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def close(self):
        pass
        
    def pairwiseExchange(self, x):
        actions = [(i,j) for i in range(1,x) for j in range(i+1,x+1) if not i==j]
        actions.append((1,1))
        return actions      
    
        # FOR CNN #
    def get_image(self):
        rgb = np.zeros((self.x,self.y,3), dtype=np.uint8)
            
        sources = np.sum(self.TM, axis = 1)
        sinks = np.sum(self.TM, axis = 0)
            
        R = np.array((self.fromState-np.min(self.fromState))/(np.max(self.fromState)-np.min(self.fromState))*255).astype(int)
        G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
        B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
                        
        k=0
        a=0
        row_counter =0
        for s in range(len(self.fromState)):
            rgb[k][a] = [R[s], G[s], B[s]]
            a+=1
            if a>(self.x-1):
                row_counter+=1
                k = row_counter
                a=0
        
        newState = np.array(rgb)
        self.state = newState.copy()
        img = Image.fromarray(self.state, 'RGB')                     

        return img
    
    
    def Test_run(self, test_runs):
        self.MHCmin = np.inf
        self.MHCmax = 0
        self.Returnflowmin = np.inf
        self.Returnflowmax = 0
        self.Noiseutopia = np.inf
        self.Noisedystopia = 0
        self.Noise_scoremin = np.inf
        self.Noise_scoremax = 0
        for i in range(test_runs):
            self.state_1D = default_rng().choice(range(1,self.n+1), size=self.n, replace=False)
            random_state = self.state_1D
            
            MHC, self.TM = self.MHC.computeMHC(self.D, self.F, random_state)
            if MHC <= self.MHCmin:
                self.MHCmin = MHC
            if MHC >= self.MHCmax:
                self.MHCmax = MHC
            
            Returnflow, self.Flowmatrix_new = self.Returnflow.computeReturnflow(self.F, random_state)
            if Returnflow <= self.Returnflowmin:
                self.Returnflowmin = Returnflow
            if Returnflow >= self.Returnflowmax:
                self.Returnflowmax = Returnflow
            
            
            self.Noisepositions = self.Noisedummy.computeNoise(self.Noise1, random_state)
            Noise_measuring_points, Noise_average = self.Noisedummy.Noisecalculation(self.Factory_Length, self.Factory_Width, self.centers_x, self.centers_y, self.Noisepositions, self.n, self.Measuring_points)
            self.Noise_Areas_Values, self.Noisemin, self.Noisemax = self.Noisedummy.Noise_Areas(Noise_measuring_points, self.n, self.Measuring_points)
            self.Noise_Areas_Values2, self.Noisemin2, self.Noisemax2 = self.Noisedummy.Noise_Areas2(Noise_measuring_points, self.n, self.Measuring_points)
            if Noise_average <= self.Noiseutopia:
                self.Noiseutopia = Noise_average
            if Noise_average >= self.Noisedystopia:
                self.Noisedystopia = Noise_average
            
            self.Under55, self.Under60, self.Under65, self.Under70, self.Under75, self.Under80, self.Under85, self.Over85  = self.Noisedummy.Noiseintervals(self.Noise_Areas_Values2)
            Noise_score = self.Noisedummy.Noise_interval_score(self.Under55, self.Under60, self.Under65, self.Under70, self.Under75, self.Under80, self.Under85, self.Over85)    
            if Noise_score <= self.Noise_scoremin:
                self.Noise_scoremin = Noise_score
            if Noise_score >= self.Noise_scoremax:
                self.Noise_scoremax = Noise_score
                
        return self.MHCmin, self.MHCmax, self.Returnflowmin, self.Returnflowmax, self.Noiseutopia, self.Noisedystopia, self.Noise_scoremin, self.Noise_scoremax