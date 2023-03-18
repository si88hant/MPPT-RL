import gym
from gym import spaces
import numpy as np
import random

class Shaded(object):
    def __init__(self):
        self.G = [100, 1000]  # read irradiance # Solar radiation in mW / sq.cm / Esto es la irradiancia para panel cuando [sombredo,sin_sombra]
        #self.T = 25  # read temperature # ojo con kelvin 273
        #self.SH = [4, 10, 7, 10, 10, 10]  # Shaded modules
        self.Mp = 10  # Modules in parallel
        self.Ng = [40, 38, 22]  # Parallel-connected series assemblies
        self.Iscr_sh = 0.375

    def data(self, Irradiancia_maxima, T, V, SH):

        # G_ar = np.array(G) # Solar radiation in mW/sq.cm

        pv = Panel()
        IUN = []
        ISH = []
        Ipv = []

        for j in range(len(self.Ng)):
            # IUN_i = pv.calc_pv(Irradiancia_maxima, T, V, SH[j*2])[0] # asi estaria bien hecho, pero como dejamos fija la irradiancia maxima lo tomamos del self.G[1]
            IUN_i = pv.calc_pv(self.G[1], T, V, SH[j*2])[0]
            if IUN_i < 0:
                IUN_i = 0
            IUN.append(IUN_i)
            ISH_i = pv.calc_pv(self.G[0], T, V, SH[j*2+1])[0]
            if ISH_i< 0:
                ISH_i = 0
            ISH.append(ISH_i)

        for jj in (range(len(self.Ng))):
            if IUN[jj] > self.Iscr_sh:
                Ipv.append(IUN[jj])
            else:
                Ipv.append(ISH[jj])

        
        IT = Ipv[0] * self.Ng[0] + Ipv[1] * self.Ng[1] + Ipv[2] * self.Ng[2]
        VT = V
        PT = IT*VT
        return IT, VT, PT 

class Panel(object):

    def __init__(self):
        self.TK = 273 # Kelvin temperature
        self.Tr1 = 40  # Reference temperature in degree fahrenheit
        # self.S = 100  # Solar radiation in mW / sq.cm
        self.ki = 0.00023  # in A / K
        self.Iscr = 3.75  # SC Current at ref.temp. in A
        self.Irr = 0.000021  # in A
        self.k = 1.38065e-23  # Boltzmann constant
        self.q = 1.6022e-19  # charge of an electron
        self.A = 2.15 # ideality factor
        self.Eg0 = 1.166 # band gap energy
        self.alpha = 0.473
        self.beta = 636
        # panel composed of Np parallel modules each one including Ns photovoltaic cells connected
        self.Np = 1
        self.Ns = 36

    def calc_pv(self, G, T, vx, SH):
        # cell temperature
        Tcell = T + self.TK
        # cell reference temperature in kelvin
        Tr = ((self.Tr1 - 32) * (5 / 9)) + 273
        # band gap energy of semiconductor
        Eg = self.Eg0 - (self.alpha * Tcell * Tcell) / (Tcell + self.beta) * self.q
        # generated photocurrent
        Iph = (self.Iscr + self.ki * (Tcell - Tr)) * (G / 1000)
        # cell reverse saturation current
        Irs = self.Irr * ((Tcell / Tr) ** 3) * np.exp(self.q * Eg / (self.k * self.A) * ((1 / Tr) - (1 / Tcell)))
        # panel output current
        I = self.Np * Iph - self.Np * Irs * (np.exp(self.q / (self.k * Tcell * self.A) * vx * (1/SH) / self.Ns) - 1)
        # panel output voltage
        V = vx # este es el Vg?
        # panel power
        P = vx * I

        return I,V,P

class MpptEnvShaded_1(gym.Env):
    metadata = {}

    def __init__(self):
        self._Pmaxnpy = np.load('Pmax.npy')
        self._Max_steps = 100

        # self.observation_space = spaces.Dict(
        #     {
        #         "Voltage": spaces.Box(low = 0.0, high = 212.0, shape=(1,), dtype=np.float32),
        #         "Power": spaces.Space(shape=(1,), dtype=np.float32),
        #         "deltaPower": spaces.Space(shape=(1,), dtype=np.float32)
        #     }
        # )

        # self.action_space = spaces.Box(low = -15.0, high = 15.0, shape=(1,), dtype=np.float32)

        self.min_actionValue = -15.0
        self.max_actionValue = 15.0

        self.max_stateValue = 55000.
        self.min_stateValue = -100.

        self.state_dim = 3
        self.action_dim = 1

        self.action_space = spaces.Box(low=self.min_actionValue, high=self.max_actionValue,
                                       shape=(self.action_dim,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=self.min_stateValue, high=self.max_stateValue,
                                       shape=(self.state_dim,), dtype=np.float32)

    def _get_obs(self):
        # return {"Voltage" : self._Voltage, "Power" : self._Power, "deltaPower" : self._deltaPower}
        return np.array([self._Voltage, self._Power, self._deltaPower]).flatten()
    
    def _get_info(self):
        return {"Steps" : self._Steps, "Current" : self._Current, "Temperature" : self._Temp, "Irradiance" : self._Irr, "Shading" : self._SH}
        # return np.array([self._Steps, self._Current, self._Temp, self._Irr, self._SH]).flatten()
        
    def reset(self):
        self._Voltage = 0.
        self._Power = 0.
        self._deltaPower = 0.
        observation = self._get_obs()

        self._Steps = 0
        self._Current = 0.
        self._Temp = 25.
        self._Irr = 1000.
        a = random.sample([1,2,3,4,5,6,7,8,9,10],1)[0]
        b = random.sample([1,2,3,4,5,6,7,8,9,10],1)[0]
        c = random.sample([1,2,3,4,5,6,7,8,9,10],1)[0]
        self._SH = [a, 10, b, 10, c, 10]
        info = self._get_info()

        # return observation, info
        return observation
    
    def step(self, action):
        self._Steps += 1
        self._Voltage += action
        oldP = self._Power

        # PVobj = PV(self._Irr, self._Temp, self._SH, self._Voltage)
        # self._Current, self._Voltage, self._Power = PVobj.array()

        pv = Shaded()
        self._Current, self._Voltage, self._Power = pv.data(self._Irr, self._Temp, self._Voltage, self._SH)

        self._deltaPower = self._Power - oldP
        done = bool(self._Steps>=self._Max_steps)

        # reward = self.reward_function(self._Power, done)
        reward = self.reward_function()
        observation = self._get_obs()
        info = self._get_info()

        return observation, float(reward), done, info

    def reward_function(self):
        return self._Power/self._Pmaxnpy[self._SH[0]][self._SH[2]][self._SH[4]]
    
    def render(self):
        pass

    def close(self):
        pass
    
    ### Temporary
    def setTempIrr(self,last_state,T,G,SH):
        self.Temp = T
        self.Irr = G
        self.SH = SH
        return last_state

