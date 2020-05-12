# from IPython import get_ipython
# get_ipython().magic('reset -sf')
import numpy as np
import matplotlib.pyplot as plt

class Linear:
    flux_name = 'Linear' 
    def __init__(self):
        def flux(self,u):
            return u
        self.__class__.flux = flux 
        def fluxd(self,u):
            return 1
        self.__class__.fluxd = fluxd
        
class Burger:
    flux_name = 'Burger' 
    def __init__(self):
        def flux(self,u):
            return u**2/2
        self.__class__.flux = flux 
        def fluxd(self,u):
            return u
        self.__class__.fluxd = fluxd
        
class Test_Problem:
    global N
    def __init__(self,test):
        if test==1:
            self.a,self.b=-1,1
            self.x=np.linspace(self.a,self.b,N+1)
            self.dx=(self.b-self.a)/N
            self.u0=-np.sin(np.pi*self.x)
            self.T=2
        elif test==2:
            self.a,self.b=-1,1
            self.x=np.linspace(self.a,self.b,N+1)
            self.dx=(self.b-self.a)/N
            self.u0=(np.abs(self.x)<1/3.0)*1+(np.abs(self.x)>=1/3.0)*(0)
            self.T=0.5
        elif test==3:
            self.a,self.b=-1,1
            self.x=np.linspace(self.a,self.b,N+1)
            self.dx=(self.b-self.a)/N
            self.u0=(1+0.5*np.sin(np.pi*self.x))
            self.T=1.5


def Add_ghost_cells(ub,bc):
    global gcn
    if bc=='Periodic':
        ub=np.insert(ub,N+1,ub[1:gcn+1])
        ub=np.insert(ub,0,ub[N-gcn:N])
    elif bc=='Neumann':
        ub=np.insert(ub,N+1,gcn*[ub[-1]])
        ub=np.insert(ub,0,gcn*[ub[0]])
        
    return ub

   
def ENO3_Reconstruction_from_averages(ub):
    # Solution of Problem 2.1 in
    # https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980007543.pdf
    global N, bc, gcn
    ub=Add_ghost_cells(ub,bc)
    dd=np.zeros([2,N+1+2*gcn])
    urb=np.insert(ub[1:N+1+2*gcn],N+2*gcn,0)
    urrb=np.insert(ub[2:N+1+2*gcn],N+2*gcn-1,2*[0])
    dd[0,:]=urb-ub
    dd[1,:]=urrb-2*urb+ub
    r=np.zeros(N+1+2*gcn)
    ulp=np.zeros(N+1+2*gcn)
    urm=np.zeros(N+1+2*gcn)
    
    for i in range(2,N+2*gcn):
        r[i]=0
        if abs(dd[0,i-1])<=abs(dd[0,i]):
            if abs(dd[1,i-2])<=abs(dd[1,i-1]):
                r[i]=2
            else:
                r[i]=1
        else:
            if abs(dd[1,i-1])<=abs(dd[1,i]):
                r[i]=1  
            else:
                r[i]=0
                
    for i in range(2,N+2*gcn-1):
        if r[i]==0:
            ulp[i]=(11/6)*ub[i]+(-7/6)*ub[i+1]+(1/3)*ub[i+2]
            urm[i]=(1/3)*ub[i]+(5/6)*ub[i+1]-(1/6)*ub[i+2]
        elif r[i]==1:
            ulp[i]=(1/3)*ub[i-1]+(5/6)*ub[i]-(1/6)*ub[i+1]
            urm[i]=-(1/6)*ub[i-1]+(5/6)*ub[i]+(1/3)*ub[i+1]
        else:
            ulp[i]=-(1/6)*ub[i-2]+(5/6)*ub[i-1]+(1/3)*ub[i]
            urm[i]=(1/3)*ub[i-2]-(7/6)*ub[i-1]+(11/6)*ub[i]
            
    return urm[gcn-1:N+gcn+1],ulp[gcn:N+gcn+2]

def monotone_flux(ulm,ulp,u):
    global cl
    af=np.max(np.abs(cl.fluxd(u))) 
    fl=0.5*(cl.flux(ulm)+cl.flux(ulp))-0.5*af*(ulp-ulm)
    return fl
    
def numerical_flux(u0):
    ulm,ulp=ENO3_Reconstruction_from_averages(u0)
    # Note here, l is used to denote u_(i-1/2). ulm(0)=u_(-1/2)^-
    # To varify the sign prperty of ENO reconstruction:
    # print((ulm[1:N+1]-ulp[1:N+1])/(ub[0:N]-ub[1:N+1])>=0)
    fl=monotone_flux(ulm,ulp,u0)
    return fl
           
def Lu(u0):
    global dx, N
#    fr=cl.flux(u0)
#    fl=cl.flux(np.insert(u0[:-1],0,u0[N]))
    fl=numerical_flux(u0)
    return (fl[1:N+2]-fl[0:N+1])/dx

def time_evolution(u0,dt):
    methodd='3rdSSP'
    if methodd=='EulerForward':
        u1=u0-dt*Lu(u0)
    elif methodd=='3rdSSP':
        u_1=u0-dt*Lu(u0)
        u_2=(3/4)*u0+(1/4)*u_1-(1/4)*dt*Lu(u_1)
        u1=(1/3)*u0+(2/3)*u_2-(2/3)*dt*Lu(u_2)
    return u1

N=102
bc='Periodic' ; gcn=3  
tp=Test_Problem(2)
cl=Burger()
t,CFL=0,0.2
u0=tp.u0
dx=tp.dx

while t<tp.T:
    af=np.max(np.abs(cl.fluxd(tp.u0))) 
    dt=CFL*dx/af
    u1=time_evolution(u0,dt)
    t=t+dt 
    u0=u1 
fig=plt.figure(1)
plt.plot(tp.x,u1,'-*')
plt.show()
  
