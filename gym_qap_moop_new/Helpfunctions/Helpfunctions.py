import numpy as np
import math
import statistics

class Functions():

    def __init__(self, shape=None, dtype=np.float32):
        self.shape = shape
        
    def computeMHC(self, D, F, s):
        P = self.permutationMatrix(s)     
        Distance = np.dot(D,P)
        Flow = np.dot(F,P.T)
        transport_intensity = np.dot(Distance, Flow)
        MHC = np.trace(transport_intensity)
                
        return MHC, transport_intensity
    
    def computeReturnflow(self, F, s):
        P = self.permutationMatrix(s)
        Flow=np.dot(P,F)
        Flow= Flow.T    
        Positionsflow = np.zeros((len(Flow[0]),len(Flow[0])))
        Returnflow = 0
        q = 0                  #Row-Counter
        for i in range(len(Flow[0])):
            Maschine = s[i]
            for j in range(len(Flow[0])):
                Positionsflow[i][j] = Flow[Maschine-1][j]
        
        for x in range(len(Flow[0])):  
            for y in range(q):      #Counting row-wise the return flowsuntil the diagonal element
                Returnflow+= Positionsflow[x,y]  #Counting all return flows
            q=q+1                   #Jump in next row
                
        return Returnflow, Flow
    
    
    def permutationMatrix(self, a):
        P = np.zeros((len(a), len(a)))
        for idx,val in enumerate(a):
            P[idx][val-1]=1
        return P
    

    
    def computeNoise(self, Noisematrix, s):
        P = self.permutationMatrix(s)
        Noiseposition = np.zeros(len(Noisematrix))
        for i in range(len(Noisematrix)):
            #Noise for Machine x placed on Pos y
            for j in range(len(Noisematrix)):
                if P[i,j]==1:
                    Noiseposition[i]= Noisematrix[j]
        
                
        return Noiseposition
    
    
    

    def computeTotalreward(self, Value1, Value2, Value3):  #Weightings arbitrary adjustable
        Weighting1 = 0      #MHC 0.4
        Weighting2 = 0      #Return flow 0.2
        Weighting3 = 1      #Noise 0.4
        Reward = Weighting1 * Value1 + Weighting2 * Value2 + Weighting3 * Value3
        return Reward




    def Flowmatrix(self):
        F = np.zeros((9, 9))

        F[0][0]=0
        F[0][1]=10
        F[0][2]=12
        F[0][3]=15
        F[0][4]=17
        F[0][5]=11
        F[0][6]=20
        F[0][7]=22
        F[0][8]=19   #1

        F[1][0]=1
        F[1][1]=0
        F[1][2]=13
        F[1][3]=18
        F[1][4]=7
        F[1][5]=2
        F[1][6]=1       
        F[1][7]=1
        F[1][8]=104   #10
        
        
        F[2][0]=2
        F[2][1]=3
        F[2][2]=0
        F[2][3]=100
        F[2][4]=109
        F[2][5]=17
        F[2][6]=100
        F[2][7]=1
        F[2][8]=31   #31

        
        F[3][0]=5
        F[3][1]=1
        F[3][2]=11
        F[3][3]=0
        F[3][4]=0
        F[3][5]=78
        F[3][6]=247
        F[3][7]=178
        F[3][8]=1

        
        F[4][0]=2
        F[4][1]=17
        F[4][2]=12
        F[4][3]=9
        F[4][4]=0
        F[4][5]=1
        F[4][6]=10
        F[4][7]=1
        F[4][8]=79  #7
        
        F[5][0]=9
        F[5][1]=14
        F[5][2]=8
        F[5][3]=21
        F[5][4]=30
        F[5][5]=0
        F[5][6]=0
        F[5][7]=1
        F[5][8]=0
        
        F[6][0]=11
        F[6][1]=19
        F[6][2]=25
        F[6][3]=31
        F[6][4]=7
        F[6][5]=2
        F[6][6]=0
        F[6][7]=0
        F[6][8]=0

        F[7][0]=5
        F[7][1]=4
        F[7][2]=12
        F[7][3]=19
        F[7][4]=23
        F[7][5]=31
        F[7][6]=40
        F[7][7]=0
        F[7][8]=12
        
        F[8][0]=8    
        F[8][1]=11
        F[8][2]=25 #2
        F[8][3]=29   #3
        F[8][4]=9
        F[8][5]=7
        F[8][6]=2
        F[8][7]=5
        F[8][8]=0        

        
        return F
    
    
    
    
    def Noisematrix(self):
        L = np.zeros(9)

        L[0]=80  #Drehen    #80
        L[1]=75  #Bohren    #75
        L[2]=70  #Fräsen    #85
        L[3]=95 #Sägen     #105   #95  #60
        L[4]=70  #Lackieren   #70
        L[5]=55  #Prüfen      #55
        L[6]=63  #Warenausgang  #63
        L[7]=72
        L[8]=63  #
        
        return L
    
    def Machine_centers(self,L,W, Machine_numbers):
       Machine_centers_x=[]
       Machine_centers_y=[]
       x_step_width = L / (Machine_numbers*2)
       y_step_width = W / (Machine_numbers*2)
       for M in range(Machine_numbers):
           for N in range(Machine_numbers):
               Machine_centers_x.append(x_step_width + 2*M*x_step_width)      
               Machine_centers_y.append(y_step_width + 2*N*y_step_width)



       return Machine_centers_x, Machine_centers_y
   
    def Noisecalculation(self,L,W, MPx, MPy, Noise, Machine_numbers, Measuring_points):
        #Measuring points per row/column
        x_step_width=L/(Measuring_points-2)  #15 Measuring points, sollten aber 17 sein
        y_step_width=L/(Measuring_points-2)
        Noise_total=[]
        Noise_Areas_numbers=Measuring_points-1
        for k in range(Noise_Areas_numbers):   #x-Direction MP
            for l in range(Noise_Areas_numbers): #y-Direction MP
                i=0
                MP=[x_step_width*k,y_step_width*l] #Positions of Measuring points
                temp=0
                r = np.zeros(Machine_numbers)
                Noise_MP = np.zeros(Machine_numbers)
                for i in range(Machine_numbers): 
                    r[i] = math.sqrt(((MPx[i]-MP[0])**2+(MPy[i]-MP[1])**2))
                    if r[i]==0:
                        r[i]=1
                    Noise_MP[i] = Noise[i] - 20 * math.log10(r[i])
                    
                    temp += 10**(0.1*Noise_MP[i])
                    
                Noise_neu = 10 * math.log10(temp)
                Noise_total.append(Noise_neu)
        Noise_average = statistics.mean(Noise_total)


        return Noise_total, Noise_average
        
    
    
    def Noise_Areas(self, Noise_total, Machine_numbers, Measuring_points):
        Measuring_points=15
        Dummycounter = 0
        Noise_Areas_Values = []
        Jump_Area_Value = 0
        for k in range(1, Machine_numbers+1):   #9 Bereiche zu untersuchen
            Jump_Area_Value += 5  #Verschiebung in x-Richtung zum nächsten Bereich  #Sprungbereich = math.ceil((math.sqrt(Maschinenanzahl)))
            Noise_Values=[]
            for m in range(math.ceil((math.sqrt(Machine_numbers)))+3):     #6 "Zeilen" pro Bereich
                for l in range(Dummycounter,Dummycounter+6):                   #6 "Spalten" pro Bereich
                    #print(l)
                    Current_Noise_Value = Noise_total[l]
                    Noise_Values.append(Current_Noise_Value)
                Dummycounter+=Measuring_points
            Noise_Value = statistics.mean(Noise_Values)
            Noise_Areas_Values.append(Noise_Value)
            Dummycounter = Jump_Area_Value
            if k % 3 == 0:      #Nach oberen Bereich Dummy = 6
                Jump_Area_Value += 65   #Messpunkte+1
                Dummycounter = Jump_Area_Value
        Noise_Min = min(Noise_Areas_Values)
        Noise_Max = max(Noise_Areas_Values)
        return Noise_Areas_Values, Noise_Min, Noise_Max
    


    
    def Distancematrixnew(self, MPx, MPy):
        D = np.zeros((9, 9))

        for i in range(9):
            for j in range(9):
                D[i][j] = math.sqrt((MPx[j]-MPx[i])**2+(MPy[j]-MPy[i])**2)
  
        return D
    
    
    def Noiseintervals(self,Noise_Meausring_points):
        Under55=0
        Under60=0
        Under65=0
        Under70=0
        Under75=0
        Under80=0
        Under85=0
        Over85=0
        for z in range(len(Noise_Meausring_points)):
            if Noise_Meausring_points[z]<55:
                Under55+=1
            if Noise_Meausring_points[z]>=55 and Noise_Meausring_points[z]<60:
                Under60+=1
            if Noise_Meausring_points[z]>=60 and Noise_Meausring_points[z]<65:
                Under65+=1
            if Noise_Meausring_points[z]>=65 and Noise_Meausring_points[z]<70:
                Under70+=1
            if Noise_Meausring_points[z]>=70 and Noise_Meausring_points[z]<75:
                Under75+=1
            if Noise_Meausring_points[z]>=75 and Noise_Meausring_points[z]<80:
                Under80+=1
            if Noise_Meausring_points[z]>=80 and Noise_Meausring_points[z]<85:
                Under85+=1
            if Noise_Meausring_points[z]>=85:
                Over85+=1
        return Under55, Under60, Under65, Under70, Under75, Under80, Under85, Over85
    
    def Noise_interval_score(self, Under55, Under60, Under65, Under70, Under75, Under80, Under85, Over85):
        a=1
        b=2
        c=3
        d=4
        e=5
        f=6
        g=9
        h=12
        Noise_score = a * Under55 + b * Under60 + c * Under65 + d * Under70 + e * Under75 + f * Under80 + g * Under85 + h * Over85
        return Noise_score
    
    
    def Noise_Areas2(self, Noise_total, Machine_numbers, Measuring_points):
        Noise_Areas_numbers = Measuring_points-1
        Dummycounter = 0
        Noise_Areas_Values = []
        Jump_Area_Value = 0
        for k in range(1, 240):   
            Jump_Area_Value += 1  #Verschiebung in x-Richtung zum nächsten Bereich  #Sprungbereich = math.ceil((math.sqrt(Maschinenanzahl)))
            Noise_Values=[]
            for m in range(2):     
                for l in range(Dummycounter,Dummycounter+2):                   #6 "Spalten" pro Bereich
                    #print(l)
                    Current_Noise_Value = Noise_total[l]
                    Noise_Values.append(Current_Noise_Value)
                Dummycounter+=Noise_Areas_numbers
            Noise_Value = statistics.mean(Noise_Values)
            Noise_Areas_Values.append(Noise_Value)
            Dummycounter = Jump_Area_Value
        Noise_Min = min(Noise_Areas_Values)
        Noise_Max = max(Noise_Areas_Values)
        return Noise_Areas_Values, Noise_Min, Noise_Max
    
    
    
 