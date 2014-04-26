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
        """ Only allow SD with M = 0 """
        if True:
            subset_SD = []
            for state in self.SD:
                if self.total_M(state)==0:
                    subset_SD.append(state)
            self.SD = subset_SD
        """ Only allow SD with <=n broken pairs """
        if True:
            n = 0
            subset_SD = []
            for state in self.SD:
            #if self.number_of_broken_pairs(state)==0:
                if self.number_of_broken_pairs(state)==n:
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
                        spstate2[3]==spstate[3] and spstate[4] == -spstate2[4]:
                    npairs += 1                    
        # double counting, thus / npairs by 2        
        return self.nparticles/2 - npairs/2
            

    def twobody_ME(self,i,j,k,l,state):
        """ First step is to check that states k and l
        are within the SD """
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
        """ Becausshell_np4_ne4_ns8.date of how P+ and P- are defined the matrix elements should
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
        for i in range(0,len(self.Htb)):
            self.Htb[i][4] *= g

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

    def J_lower_J_raise(self,sd):
        J_lower_J_raise_sd_final = []
        #print bin(sd),sd
        J_raise_sd = self.J_raise(sd)
        if len(J_raise_sd)!=0:
            for state in J_raise_sd:
                #print bin(state[1]),state[1]
                J_lower_J_raise_sd = self.J_lower(state[1])
                for i in range(0,len(J_lower_J_raise_sd)):
                    J_lower_J_raise_sd[i][0] *= state[0]
                    J_lower_J_raise_sd_final.append(J_lower_J_raise_sd[i])
        else:
            J_lower_J_raise_sd_final.append([0,sd])
        return J_lower_J_raise_sd_final

    def J_squared(self,sd):
        """ Calculating J^2 using quasispin operators """
        J_squared_sd = []
        J_lower_J_raise_sd = self.J_lower_J_raise(sd)
        list_of_unique_sd = []
        for _sd in J_lower_J_raise_sd:
            list_of_unique_sd.append(_sd[1])
        list_of_unique_sd = list(set(list_of_unique_sd))
        if len(list_of_unique_sd)==len(J_lower_J_raise_sd):
            J_squared_sd = J_lower_J_raise_sd
        else:
            for unique_sd in list_of_unique_sd:
                temp = [0, unique_sd]
                for _sd in J_lower_J_raise_sd:
                    if _sd[1] == unique_sd:
                        temp[0]+=_sd[0]
                J_squared_sd.append(temp)    
        J_z_sd = self.J_z(sd)
        J_z_J_z_sd = [J_z_sd[0]*J_z_sd[0],sd]
        for _sd in J_squared_sd:
            if _sd[1]==sd:
                _sd[0] += J_z_sd[0] + J_z_J_z_sd[0]                
        return J_squared_sd

    def operator_expectation_value(self,operator,final,initial):
        if type(final)==type(initial)==list:
            True
        elif type(initial)==list: 
            final = [[1,final]]
        elif type(final)==list: 
            initial = [[1,initial]]
        else:
            initial = [[1,initial]]
            final = [[1,final]]

        final_operator_sd = []
        for sd in initial:
            operator_sd = operator(sd[1])
            for _sd in operator_sd:
                _sd[0] *= sd[0]
                final_operator_sd.append(_sd)
        list_of_unique_sd = []
        operator_sd = []
        for _sd in final_operator_sd:
            list_of_unique_sd.append(_sd[1])
        list_of_unique_sd = list(set(list_of_unique_sd))
        if len(list_of_unique_sd)==len(final_operator_sd):
            operator_sd = final_operator_sd
        else:
            for unique_sd in list_of_unique_sd:
                temp = [0, unique_sd]
                for _sd in final_operator_sd:
                    if _sd[1] == unique_sd:
                        temp[0]+=_sd[0]
                operator_sd.append(temp)    

        J_squared_eigenvalue = 0
        for op_sd in operator_sd:
            for query_sd in final:
                if query_sd[1]==op_sd[1]:
                    J_squared_eigenvalue += query_sd[0]*op_sd[0]
        return J_squared_eigenvalue
                                             
           
    def total_J(self,sd=None,Jsq=None):
        if Jsq != None:
            return 0.5*(-1+np.sqrt(1+4*Jsq))            
        elif sd == None:
            print "STOP: no slater determinant or Jsquared eigenvalue given"
            exit()
        else:
            J_sq = self.J_squared(sd)
            return 0.5*(-1+np.sqrt(1+4*J_sq))
        
                          
                
if __name__=='__main__':
    do_plot = False
    #data = 'shell_np2_ne2_ns8.dat'
    data = "shell_np2_ne1_ns4.dat"
    #data = 'shell_usdb.dat'
    smObj = ShellModel(data)
    smObj.init_SD(restrictions=True)
    if do_plot:
        energies = []
        perturbation = []
        scale = [1.0/10.0,1.0/9.0,1.0/8.0,1.0/7.0,1.0/6.0,1.0/5.0,1.0/4.0,1.0/3.0,1.0/2.0,1,2,3,4,5,6,7,8,9,10]
        for g in scale:
            smObj.vary_perturbation_strength(g)
            smObj.init_H()
            eigenvalues = scipy.linalg.eigh(smObj.H,eigvals_only=True)
            energies.append(eigenvalues)
            perturbation.append(np.log10(g))
            print smObj.H
            print eigenvalues
        energies = np.asarray(energies)
        perturbation = np.asarray(perturbation)
        for n in range(0,smObj.nSD):
            pylab.plot(perturbation,energies[:,n])
        pylab.xlabel('g')
        #pylab.xlim(0.1,10.0)
        grid()
        pylab.savefig("/user/sullivan/public_html/shell.pdf")
    else:
        #smObj.vary_perturbation_strength(1.0)
        print [bin(i) for i in smObj.SD]
        print smObj.nSD
        smObj.init_H()
        for row in smObj.H:
            print "[",
            for item in row:
                print format(item,'+.00f'),
            print "]"
            
        eigenvectors = np.linalg.eigh(smObj.H)

        # find j-values of H eigenstates 
        for eigenstate in eigenvectors[1]:
            initial_eigenstate = []
            for i in range(0,smObj.nSD):            
                initial_eigenstate.append([eigenstate[i],smObj.SD[i]])             
            J_sqrd = smObj.operator_expectation_value(smObj.J_squared,initial_eigenstate,initial_eigenstate)
            J = smObj.total_J(Jsq=J_sqrd)
            print J_sqrd,J
