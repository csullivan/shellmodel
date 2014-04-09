import numpy as np
import itertools
import pylab
import fileinput


class ShellModel:
    nparticles = 0
    nstates = 0
    nspstates = 0    
    nSD = 0
    states = []
    index = []
    l = []
    j = []
    jz = []
    spenergy = []
    SD = []
    SD_lookup = []
    Htb = []
    H = []
    def __init__(self,data):
        print "Start"
        self.nparticles = self.input_parser(data,'Particles')    
        self.states = self.input_parser(data,'States')    
        self.Htb = self.input_parser(data,'Matrix Elements')    
        self.init_qm_numbers()
    def init_qm_numbers(self):
        self.index = self.states[:,0]
        self.l = self.states[:,1]
        self.j = self.states[:,2]/2.0
        self.jz = self.states[:,3]/2.0
        self.spenergy = self.states[:,4]
        self.nspstates = int(np.max(self.states[:,0]))
        self.nstates = int(np.max(self.states[:,0]))
    def input_parser(self,file,searchExp):
        input_file = open(file)
        for line in input_file:
            if searchExp in line:
                if searchExp == 'Particles':
                    return int(line.split()[2])
                if searchExp == 'States':
                    states = []
                    while True:
                        try:
                            line = next(input_file)
                            if line == '\n':
                                """ Break if empty line is 
                                found after reading last state """
                                break
                        except:
                            break
                        line = line.split()
                        line = map(float,line)
                        states.append(line)
                    states = np.asarray(states)
                    return states
                if searchExp == 'Matrix Elements':
                    H_tb = []
                    while True:
                        try:
                            line = next(input_file)
                            if line == '\n':
                                """ Break if empty line is 
                                found after reading last state """
                                break
                        except:
                            break
                        line = line.split()
                        line = map(float,line)
                        H_tb.append(line)
                    H_tb = np.asarray(H_tb)
                    return H_tb

                    
    def init_SD(self,restrictions=None):
        """ Builds all combinations of SD for nparticles in 
            nstates using the bit representation """
        state_list = range(0,self.nstates)
        for combo in itertools.combinations(state_list,self.nparticles):
            self.SD.append(sum(2**i for i in combo))
        if restrictions:
            self.restrict()
        self.SD = list(reversed(sorted(self.SD))) #ordering for purposes of H
        self.nSD = len(self.SD)
        self.SD_lookup = set(self.SD)
    def restrict(self):
        """ Only allow SD with pairs """
        subset_SD = []
        ################
        def pair(state):
            pairs = []            
            """ Selects every other sp state from 0 to nstates """
            for n in range(0,self.nstates)[::2]:
                """ Moves through the state and picks out pairs """
                if state & (0b11<<n) == (0b11<<n):
                    pairs.append(1)
            if len(pairs)==self.nparticles/2:
                return True
            else:
                return False                
        ################
        for state in self.SD:
            if pair(state):
                subset_SD.append(state)
        self.SD = subset_SD

    def twobody_ME(self,i,j,k,l,state):
        """ First step is to check that states k and l
        are withing the SD """
        ###################
        def phase(i,state):
            """ i should be {1..8} representing one of the spstates """
            occupied = []
            for n in range(1,self.nspstates+1):
                if n == i:
                    break
                if state & (0b1<<self.nspstates - n) == (0b1<<self.nspstates - n):
                    occupied.append(1)                

            if len(occupied) % 2 != 0:
                """ the odd condition is due to the fact that the 
                state 11110000 is really a4*a3*a2*a1*|0>, so the counting
                must be done from left to right. But the counting I do is
                right to left and I then compensate with the above condition """
                return -1
            else:
                return 1
        ###################                
        """ Because of how P+ and P- are defined the matrix elements should
            always be a1*a2*a2a1  for the spme 1 2 1 2 """
        if state & (0b1<<self.nspstates-l) != (0b1<<self.nspstates-l) or \
                state & (0b1<<self.nspstates-k) != (0b1<<self.nspstates-k):
            return 0
        else:
            phase_factor = phase(k,state)
            state = state ^ (0b1<<self.nspstates-k)
            phase_factor = phase_factor*phase(l,state)
            state = state ^ (0b1<<self.nspstates-l)
            if state & (0b1<<self.nspstates-j) == (0b1<<self.nspstates-j) or \
                    state & (0b1<<self.nspstates-i) == (0b1<<self.nspstates-i):
                """ If the state is already populated, creation should fail """
                return 0
            else:
                phase_factor = phase_factor*phase(j,state)
                state = state + 2**(self.nspstates - j)
                phase_factor = phase_factor*phase(i,state)
                state = state + 2**(self.nspstates - i)
        return state*phase_factor          
            
    def init_H(self):
        self.H = np.zeros(shape=(self.nSD,self.nSD))
        self.H += self.setup_twobody()
        self.H += self.setup_onebody()

    def setup_twobody(self):
        M = np.zeros(shape=(self.nSD,self.nSD))
        """ Two body interaction from sp-me's in file """
        for alpha in self.SD:            
            """ Loop over SD, and for each find the contribution of
            the sp_2body matrix elements """
            for sp_tbme in self.Htb:
                i = int(sp_tbme[0])
                j = int(sp_tbme[1])
                k = int(sp_tbme[2])
                l = int(sp_tbme[3])
                V = sp_tbme[4]
                alpha_prime = self.twobody_ME(i,j,k,l,alpha)
                if abs(alpha_prime) in self.SD_lookup:
                    beta_ind = self.SD.index(abs(alpha_prime))
                    alpha_ind = self.SD.index(alpha)
                    M[alpha_ind,beta_ind] += np.sign(alpha_prime)*V #pg458
                    if(alpha_ind != beta_ind): #Symmetric matrix
                        M[beta_ind,alpha_ind] += np.sign(alpha_prime)*V
        return M

    def setup_onebody(self):
        M = np.zeros(shape=(self.nSD,self.nSD))
        """ One body interaction """
        for state in self.SD:
            occupied = []
            ob_ind = self.SD.index(state)
            for n in reversed(range(0,self.nspstates)):
                if state & (0b1<<n) == (0b1<<n):
                    occupied.append(self.nspstates-n)
            for sp_state in occupied:                
                M[ob_ind,ob_ind] += self.spenergy[sp_state-1]
        return M        

    def vary_perturbation_strength(self,g):
        for ME in self.Htb:
            ME[4] = ME[4]*g
                  
                
if __name__=='__main__':
    data = 'example.dat'
    smObj = ShellModel(data)
    smObj.init_SD(restrictions=True)
    smObj.init_H()
    print smObj.H
    eigenvalues = np.linalg.eigvals(smObj.H)
    print eigenvalues
    smObj.vary_perturbation_strength(5)
