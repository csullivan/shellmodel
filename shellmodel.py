import numpy as np
import itertools
import pylab
from pylab import grid
import fileinput
import scipy.linalg


class ShellModel:
    nparticles = 0
    nspstates = 0    
    nSD = 0
    states = []
    index = []
    n = []
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
        self.n = self.states[:,1]
        self.l = self.states[:,2]
        self.j = self.states[:,3]/2.0
        self.jz = self.states[:,4]/2.0
        self.spenergy = self.states[:,5]
        self.nspstates = int(np.max(self.states[:,0]))
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
            nspstates using the bit representation """
        state_list = range(0,self.nspstates)
        for combo in itertools.combinations(state_list,self.nparticles):
            self.SD.append(sum(2**i for i in combo))
        if restrictions:
            self.restrict()
        self.SD = list(reversed(sorted(self.SD))) #ordering for purposes of H
        self.nSD = len(self.SD)
        self.SD_lookup = set(self.SD)

    def pair(self,state):
        pairs = []            
        """ Selects every other sp state from 0 to nspstates """
        for n in range(0,self.nspstates)[::2]:
            """ Moves through the state and picks out pairs """
            if state & (0b11<<n) == (0b11<<n):
                pairs.append(1)
        if len(pairs)==self.nparticles/2:
            return True
        else:
            return False                

    def restrict(self):
        """ Only allow SD with pairs """
        # subset_SD = []
        # for state in self.SD:
        #     if self.pair(state):
        #         subset_SD.append(state)
        # self.SD = subset_SD
        subset_SD = []
        for state in self.SD:
            if self.total_M(state)==0:
                subset_SD.append(state)
        self.SD = subset_SD
        subset_SD = []
        for state in self.SD:
            if self.number_of_broken_pairs(state)==0:
                subset_SD.append(state)
        self.SD = subset_SD


    def total_M(self,state):
        """ Returns the total angular momentum
        projection for a given slater-determinant """
        total_angular_momentum_projection = 0.0
        for n in reversed(range(0,self.nspstates)):
            if state & (0b1<<n) == (0b1<<n):
                total_angular_momentum_projection += \
                    self.states[self.nspstates-n-1][4]/2.0
        return total_angular_momentum_projection

    def number_of_broken_pairs(self,state):
        occupied = []
        npairs = 0
        for n in reversed(range(0,self.nspstates)):
            if state & (0b1<<n) == (0b1<<n):
                occupied.append(self.states[self.nspstates-n-1])
        for spstate in occupied:
            for spstate2 in occupied:                
                if spstate[0] != spstate2[0] and \
                        spstate2[1]==spstate[1] and spstate2[2]==spstate[2] and \
                        spstate2[3]==spstate[3]:
                    npairs += 1                    
        # double counting, thus / npairs by 2
        return self.nparticles/2 - npairs/2
            

    def twobody_ME(self,i,j,k,l,state):
        """ First step is to check that states k and l
        are withing the SD """
        ###################
        def phase(i,state):
            """ i should be an element in 
            {1 ... number_of_single_particle_states} 
            representing one of the spstates """
            occupied = []
            for n in range(1,self.nspstates+1):
                if n == i:
                    break
                if state & (0b1<<self.nspstates - n) == (0b1<<self.nspstates - n):
                    occupied.append(1)                

            if len(occupied) % 2 != 0:
                """ If the number of permuatations is odd """
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
            ME[4] = -g

    def init_H(self):
        self.H = np.zeros(shape=(self.nSD,self.nSD))
        self.H += self.setup_twobody()
        self.H += self.setup_onebody()

    def occupied_sp_states(self,sd):
        occupied = []
        for n in reversed(range(0,self.nspstates)):
            if sd & (0b1<<n) == (0b1<<n):
                #occupied.append(self.nspstates-n)
                occupied.append(self.nspstates-n)
        return occupied

    def J_raise(self,sd):
        """ Total J+ operator """
        final_states = []
        for n in reversed(range(0,self.nspstates)):
            if sd & (0b1<<n) == (0b1<<n):
                #look up sp states j and mj
                _n = self.states[self.nspstates-n-1][1]
                _l = self.states[self.nspstates-n-1][2]
                _2j = self.states[self.nspstates-n-1][3]
                _2mj = self.states[self.nspstates-n-1][4]
                """ check to see if sp state with same n,l,j and mj+1 exists.
                If so, sd = 0, otherwise destroy sp state with (j,mj), and create (j,mj+1) """
                for spstate in self.states:
                    """ run through list of sp-states and find a state with n,l,j,mj+1 """
                    if spstate[1]==_n and spstate[2]==_l and spstate[3]==_2j and spstate[4]==_2mj+2:
                        """ if such a state exists, check if it's occupied. If so, a^\dagger = 0,
                        else apply the operator destroying n,l,j,mj and create n,l,j,mj+1 """
                        if sd & (0b1<<(self.nspstates-int(spstate[0]))) == (0b1<<(self.nspstates-int(spstate[0]))):
                            continue
                        else:                                                      
                            """ store the final slater determinant along with the eigenvalue in a list """
                            final_sd = sd ^ (0b1<<n)
                            final_sd = final_sd ^ (0b1 << (self.nspstates-int(spstate[0])))
                            final_states.append([np.sqrt((_2j/2.0+_2mj/2.0+1)*(_2j/2.0-_2mj/2.0)),final_sd])
        return final_states

    def J_lower(self,sd):
        """ Total J- operator """
        final_states = []
        for n in reversed(range(0,self.nspstates)):
            if sd & (0b1<<n) == (0b1<<n):
                #look up sp states j and mj
                _n = self.states[self.nspstates-n-1][1]
                _l = self.states[self.nspstates-n-1][2]
                _2j = self.states[self.nspstates-n-1][3]
                _2mj = self.states[self.nspstates-n-1][4]
                """ check to see if sp state with same n,l,j and mj-1 exists.
                If so, sd = 0, otherwise destroy sp state with (j,mj), and create (j,mj-1) """
                for spstate in self.states:
                    """ run through list of sp-states and find a state with n,l,j,mj-1 """
                    if spstate[1]==_n and spstate[2]==_l and spstate[3]==_2j and spstate[4]==_2mj-2:
                        """ if such a state exists, check if it's occupied. If so, a^\dagger = 0,
                        else apply the operator destroying n,l,j,mj and create n,l,j,mj-1 """
                        if sd & (0b1<<(self.nspstates-int(spstate[0]))) == (0b1<<(self.nspstates-int(spstate[0]))):
                            continue
                        else:                                                      
                            """ store the final slater determinant along with the eigenvalue in a list """
                            final_sd = sd ^ (0b1<<n) # annhilation operation
                            final_sd = final_sd ^ (0b1 << (self.nspstates-int(spstate[0]))) # creation operation
                            final_states.append([np.sqrt((_2j/2.0-_2mj/2.0+1)*(_2j/2.0+_2mj/2.0)),final_sd])
        return final_states

    def J_z(self,sd):
        """ Total Jz operator """
        total_mj = 0
        for n in reversed(range(0,self.nspstates)):
            if sd & (0b1<<n) == (0b1<<n):
                #look up sp states j and mj
                _2mj = self.states[self.nspstates-n-1][4]
                total_mj += _2mj/2.0
        return [total_mj,sd]

        
           
    def J_squared(self,sd):
        """ Calculating J^2 using quasispin operators """
        J_lower_J_raise_sd_final = [0,0]
        #print bin(sd),sd
        J_raise_sd = self.J_raise(sd)
        if len(J_raise_sd)!=0:
            for state in J_raise_sd:
                #print bin(state[1]),state[1]
                J_lower_J_raise_sd = self.J_lower(state[1])
                for i in range(0,len(J_lower_J_raise_sd)):
                    #print J_lower_J_raise_sd                        
                    if J_lower_J_raise_sd[i][1] == sd:
                        J_lower_J_raise_sd[i][0] *= state[0]
                    else:
                        print bin(J_lower_J_raise_sd[i][1]),bin(sd),"off-diaganol J^2 element"
                        continue # need to investigate off-diagonol elements
                        # Morten indicates that one should only look at <psi|J^2|psi>
                        # so I am correct in just ignoring the off diaganol piece here
                        #print "STOP: error in J^2"
                        #exit()
                    J_lower_J_raise_sd_final[0] += J_lower_J_raise_sd[i][0]
                    J_lower_J_raise_sd_final[1] = J_lower_J_raise_sd[i][1]
        else:
            J_lower_J_raise_sd_final[1] = sd
        J_z_sd = self.J_z(sd)
        J_z_J_z_sd = [J_z_sd[0]*J_z_sd[0],sd]
        if J_lower_J_raise_sd_final[1]==J_z_sd[1]==sd:
            return J_lower_J_raise_sd_final[0]+J_z_J_z_sd[0]+J_z_sd[0]
        else:
            print "STOP: something wrong with quasispin operators"
            exit()

    def total_J(self,sd,Jsq=None):
        if Jsq != None:
            return 0.5*(-1+np.sqrt(1+4*Jsq))            
        else:
            J_sq = self.J_squared(sd)
            return 0.5*(-1+np.sqrt(1+4*J_sq))
        
    def find_J_of_eigenstates(self,coefs):
        """ Using eigenstates from diaganolizing H, calculate <psi|J^2|psi> """
        H_eigenstates = []
        for i in range(0,self.nSD):
            H_eigenstates.append([coefs[i],self.SD])
        print H_eigenstates
        for estate in H_eigenstates:
            for sd in estate[1]:
                print self.J_squared(sd)
                
            
            
        
                
            
        
        
        
                  
                
if __name__=='__main__':
    do_plot = False
    data = 'shell_np4_ne3_ns8.dat'
    smObj = ShellModel(data)
    smObj.init_SD(restrictions=True)
    if do_plot:
        energies = []
        perturbation = []
        for g in range(-10,11):
            g = g/10.0
            smObj.vary_perturbation_strength(g)
            smObj.init_H()
            eigenvalues = scipy.linalg.eigh(smObj.H,eigvals_only=True)
            energies.append(eigenvalues)
            perturbation.append(g)
            print smObj.H
            print eigenvalues
        energies = np.asarray(energies)
        perturbation = np.asarray(perturbation)
        for n in range(0,smObj.nSD):
            pylab.plot(perturbation,energies[:,n])
        pylab.xlabel('g')
        pylab.xlim(-1.0,1.0)
        grid()
        pylab.savefig("/user/sullivan/public_html/shell.pdf")
    else:
        smObj.vary_perturbation_strength(1.0)
        print [bin(i) for i in smObj.SD]
        #print [smObj.total_J(i) for i in smObj.SD]
        smObj.init_H()
        print smObj.H
        #eigenvalues = scipy.linalg.eigh(smObj.H,eigvals_only=True)
        eigenvectors = np.linalg.eigh(smObj.H)
        smObj.find_J_of_eigenstates(eigenvectors[1])
            
            
        
    

