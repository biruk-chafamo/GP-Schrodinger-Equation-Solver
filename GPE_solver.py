from dataclasses import dataclass, field
import math
import numpy as np
from numpy.fft import *
from collections.abc import Callable
from numpy.fft import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import MaxNLocator

@dataclass
class Constants:
    """
    Preset constants. Set up in order to act as dimensionless units.
    """
    hbar:float      = 1                             # reduced plank's constant
    amu :float      = 1.6605e-27                    # atomic mass unit
    mass:float      = 1                             # atomic mass of Rb87
    wx  :float      = 1                             # trap frequencies
    a_s :float      = 6e-3                          # scattering length
    N   :int        = 2000                          # number of atoms   
    dx  :float      = 0.05                          # space step
    dt  :float      = 0.05**2 / 10                  # time step
    M   :int        = 2**4                          # half the number of fourier modes
    dim :int        = 1                             # dimensions of the wavefunctions
    lx  :float      = field(init=False)             # HO length
    g  :float       = field(init=False)             # interaction constant

    def __post_init__(self):
        self.lx = np.sqrt(self.hbar / (self.mass * self.wx))     
        self.g  = self.N * 2 * (self.hbar**2) * self.a_s / (self.mass * self.lx**2)  


class Solution:

    def __init__(self, constants: Constants, V: Callable) -> None:
        """
        Initiates the Solution class with the following space grid, 
        frequency grid, external potential grid; makes an initial guess for the 
        wave function; and sets the proper Fourier function for the given dimension.


        Args:
            constants (Constants): list of constants to be used throughout the solution
            V (Callable): external potential trap function used for trapping the BEC
        """
        self.constants = constants
        self.vis = Visualization(self)
        assert(self.constants.dim in (1, 2, 3))
        self.indecies = np.arange(start=-self.constants.M, stop=self.constants.M, step=1)
        self.positions = self.generate_spatial_grid()
        self.frequencies = self.generate_frequency_grid()
        self.ext_V = self.generate_external_potential(V)
        self.guessed_psi = self.guess_psi()
        if self.constants.dim == 1:
            self.fft = np.fft.fft
            self.ifft = np.fft.ifft
        else:
            self.fft = np.fft.fft2
            self.ifft = np.fft.ifft2
        self.ground_psi = self.guessed_psi * 0  # initializing ground psi as [0,0,0,...]


    def generate_spatial_grid(self) -> dict:
        """
        creates the spatial grid based on the specified dimension and spatial range 
        of the solution

        Returns:
            positions (dict<int,np.array>): meshgrid of position in each dimension
        """
        XYZ = [self.indecies * self.constants.dx for _ in range(self.constants.dim)]     
        positions = dict()
        for idx, axis in enumerate(np.meshgrid(*XYZ)):
            positions[idx] = axis    
        return positions

    def generate_frequency_grid(self) -> dict:
        """
        creates the frequency grid based on the specified dimension and spatial range 
        of the solution

        Returns:
            frequencies (dict<int,np.array>): meshgrid of the wavenumbers in each dimension
        """
        K_XYZ = [self.indecies * 2 * np.pi / (self.constants.dx * self.constants.M)  for _ in range(self.constants.dim)]     
        frequencies = dict()
        for idx, freq in enumerate(np.meshgrid(*K_XYZ)):
            frequencies[idx] = freq    
        return frequencies

    def generate_external_potential(self, V:Callable) -> np.array:
        """
        creating an array of the external potential for all positions based on a given function V

        Args: 
            V
        Returns:
            ext_V (np.array): external potential with the same shape as the computational grid 
        """
        positions = self.positions.values()  # Iterable<np.array>
        ext_V = V(self.constants, *positions)
        return ext_V

    def guess_psi(self) -> np.array:
        """
        generates an initial avefunction, psi, based on the given constants. This is not the true 
        ground potential of the system. 

        Returns:
            psi (np.array): guessed wavefunction of the Bose-Einstien Condensates with |psi|^2 = 1
        """
        alpha = np.sqrt(self.constants.N / self.constants.lx)*(1 / np.pi)**(1 / 4)
        # alpha = 1
        psi = alpha * np.exp(
            sum([-pos**2 / (2*self.constants.lx**2) for pos in self.positions.values()])
        )
        psi = psi/np.sqrt((np.linalg.norm(np.absolute(psi)**2)))  # setting |psi|^2 = 1
        return psi

    def SSFM(self, psi: np.array, dt: float, exp_D: np.array, i: complex) -> np.array:
        """
        Implements the split-step Fourier method to evolve psi across either time or 
        imaginary time by dt. If i == 1 then step occurs in imaginary time. If i is complex i,
        then step occurs in time.

        Args:
            psi (np.array): wavefunction 
            dt (float):     time step
            i (complex):    either 1 or imaginary i

        Returns:
            psi: non-normalized psi evolved by dt
        """
        psi = self.exp_nonlinear_step(psi, dt,i)
        psi = self.exp_linear_step(psi, exp_D)
        psi = self.exp_nonlinear_step(psi, dt,i)
        return psi

    def RK4IP(self, psi: np.array, dt: float, exp_D: np.array, i: complex) -> np.array:
        """
        Implements the Runge-Kutta 4 method in the interaction pitcure as shown in 
        Caradoc-Davies, Benjamin Michael. Vortex dynamics in Bose-Einstein condensates. 
        Diss. PhD thesis, University of Otago (NZ), 2000

        Args:
            psi (np.array): initial wavefunction of the BEC
            dt (float):     time step
            exp_D (np.array): e^(const * dt * k^2) where k is the wavenumber stored in self.frequencies
            i (complex):    can either be 1 for imaginary time evolutions or 0+1i for regular time evolution

        Returns:
            psi_t (np.array): wavefunction of the BEC evolved by time dt
        """
        psi_I = self.exp_linear_step(psi, exp_D)
        s1 = self.exp_linear_step(self.nonlinear_step(psi, dt, i), exp_D)
        s2 = self.nonlinear_step(psi_I + s1/2, dt, i)
        s3 = self.nonlinear_step(psi_I + s2/2, dt, i)
        s4 = self.nonlinear_step(self.exp_linear_step(psi_I + s3, exp_D), dt, i)
        psi_t = self.exp_linear_step((psi_I + s1/6 + s2/3 + s3/3), exp_D) + s4/6
        return psi_t

    def exp_linear_step(self, psi: np.array, exp_D: np.array) -> np.array:
        """
        performs the equivalent operation as e^(const * dt * ∇²)psi. Implemented by applying the operator
        exp_D on the Fourier transform of psi and then reversing the transform again

        Args:
            psi (np.array):     wavefunction of the BEC
            exp_D (np.array):   e^(const * dt * k^2) where k is the wavenumber stored in self.frequencies

        Returns:
            psi_new (np.array): result of the operator e^(const * ∇²) acting on psi
        """
        psi_m = fftshift(self.fft(psi)) / (2 * self.constants.M)        # converting to momentum space
        psi_m = exp_D * psi_m                                           # applying linear operator
        psi_new = self.ifft(ifftshift(psi_m)) * (2 * self.constants.M)  # converting back to spatial space 
        return psi_new
    

    def nonlinear_step(self, psi: np.array, dt: float, i) -> np.array:
        """
        performs the nonlinear operator const * dt * N = const * dt * (V + g|psi|^2) on psi 

        Args:
            psi (np.array): wavefunction of the BEC
            dt (float):     time step
            i :             can either be 1 for imaginary time evolutions or 0+1i for regular time 
                            evolution

        Returns:
            psi_new (np.array): result of the operator const * dt * N acting on psi
        """
        psi_new = -psi * dt * i * (self.ext_V + (self.constants.g * np.absolute(psi)**2)) / self.constants.hbar   # applying nonlinear step on psi
        return psi_new

    def exp_nonlinear_step(self, psi: np.array, dt: float, i) -> np.array:
        """
        performs the nonlinear operator e^(const * dt * N) = e^(const * dt * (V + g|psi|^2)) on psi 

        Args:
            psi (np.array): wavefunction of the BEC
            dt (float):     time step
            i :             can either be 1 for imaginary time evolutions or 0+1i for regular time 
                            evolution

        Returns:
            psi_new (np.array): result of the operator e^(const * dt * N) acting on psi
        """
        if i == complex(1):
            i = 1
        psi_new = psi * np.exp((-0.5 * dt * i) * (self.ext_V + (self.constants.g * np.absolute(psi)**2)) / self.constants.hbar)   # applying exp nonlinear step on psi
        return psi_new

    def set_exp_D(self, dt: float, i: complex) -> np.array:
        """
        calculates e^(const * dt * k^2) where k is the wavenumber stored in self.frequencies

        Args:
            dt (float): time step
            i (complex): can either be 1 for imaginary time evolutions or 0+1i for regular time evolution

        Returns:
            exp_D (np.array)
        """
        if i == complex(1):
            i = 1
        self.exp_D = np.exp((-0.5 * dt * i) * (self.constants.hbar/self.constants.mass)* sum([freq**2 for freq in self.frequencies.values()]))
        return self.exp_D
    
    def set_exp_D_half(self, dt: float, i: complex) -> np.array:
        """
        calculates e^(const * (dt/2) * k^2) where k is the wavenumber stored in self.frequencies

        Args:
            dt (float): time step
            i (complex): can either be 1 for imaginary time evolutions or 0+1i for regular time evolution

        Returns:
            exp_D_half (np.array)
        """
        if i == complex(1):
            i = 1
        self.exp_D_half = np.exp((-0.5 * (dt / 2) * i) * (self.constants.hbar/self.constants.mass)* sum([freq**2 for freq in self.frequencies.values()]))
        return self.exp_D_half

    def time_evolve(self, ground_psi:np.array, stepper_method: str, dt: float, Nt: int, snapshots:int, adaptive_step: bool, dt_min: float, err_tol: tuple = (0,1)) -> list:      
        """
        evolves the ground state psi across time. Note that the external potential (self.ext_V) must be changed or set
        to zero in order to observe changes in the wavefunction across time. 

        Args:
            ground_psi (np.array):  the ground state wavefunction of the BEC
            stepper_method (str):   method used to perform the integration (RK4IP or SSFM)
            dt (float):             initial time step (remains constant if adaptive_step is False)
            Nt (int):               number of times to evolve psi by the initial dt
            snapshots (int):        number of periodical snapshots of psi to save in memory
            adaptive_step (bool):   if True, will adapt dt based on value of err_tol
            dt_min (float):         adaptive step will not reduce dt below dt_min
            err_tol (tuple):        acceptable range of error between evolving psi by dt once and
                                    evolving psi by dt/2 twice. Defaults to (0,1).

        Raises:
            ValueError: if either 'RK4IP' or 'SSFM' are not chosen for stepper_method
            ValueError: if ground state is not solved prior to calling this method

        Returns:
            saved_psi (list): list of snapshots of psi during the time evolution 
        """
        #-------------------------------------- checks -----------------------------------#
        assert(err_tol[1] > err_tol[0])
        if stepper_method not in ('RK4IP','SSFM'):
            raise ValueError("choose either 'RK4IP' or 'SSFM'.")
        if ground_psi.all() == 0:
            raise ValueError('must find ground state first by calling solve_ground_state()')
        #---------------------------------------------------------------------------------#

        stepper = self.RK4IP if stepper_method == 'RK4IP' else self.SSFM
        print(f'using {stepper_method} {stepper}')
        psi = ground_psi
        norm_squared_psi = np.linalg.norm(psi)**2
        exp_D = self.set_exp_D(dt, 1j) if stepper_method == 'SSFM' else self.set_exp_D_half(dt, 1j)
        
        t, idx, self.saved_psi, self.saved_t = 0, 0, [psi], [0]
        t_max = Nt * dt
        while t < t_max:
            psi_1 = stepper(psi, dt, exp_D, 1j)
            if adaptive_step:
                psi_2 = stepper(psi, dt/2, exp_D, 1j) 
                psi_2 = stepper(psi_2, dt/2, exp_D, 1j)
                dt_old = dt
                dt, update_psi = self.adapt_step(psi_1, psi_2, dt, dt_min, err_tol)
                if dt_old != dt: # only changing exp_D if dt changes
                    exp_D = self.set_exp_D(dt, 1j) if stepper_method == 'SSFM' else self.set_exp_D_half(dt, 1j)  
                if update_psi:  # will be true if error is acceptable
                    psi = psi_2
                    idx += 1
                    t += dt
            else:
                psi = psi_1
                idx += 1
                t += dt
            
            if idx % (Nt / snapshots) == 0: # TODO: should not be based on idx but time
                self.saved_psi.append(psi)
                self.saved_t.append(t)
        print(f'idx: {idx}, Nt: {Nt}')
        return self.saved_psi, idx

    def time_evolve_interference(self, psi:np.array, stepper_method: str, dt: float, Nt: int, snapshots:int, adaptive_step: bool, dt_min: float, err_tol: tuple = (0,1)) -> list:   
        """
        evolves the interference between the ground state wavefunction and a duplicate wavefunction 
        shifted by some amount. Note that the external potential (self.ext_V) must be changed or set
        to zero in order to observe changes in the wavefunction across time. 

        Args:
            ground_psi (np.array):  the ground state wavefunction of the BEC
            stepper_method (str):   method used to perform the integration (RK4IP or SSFM)
            dt (float):             initial time step (remains constant if adaptive_step is False)
            Nt (int):               number of times to evolve psi by the initial dt
            snapshots (int):        number of periodical snapshots of psi to save in memory
            adaptive_step (bool):   if True, will adapt dt based on value of err_tol
            dt_min (float):         adaptive step will not reduce dt below dt_min
            err_tol (tuple):        acceptable range of error between evolving psi by dt once and
                                    evolving psi by dt/2 twice. Defaults to (0,1).

        Raises:
            ValueError: if either 'RK4IP' or 'SSFM' are not chosen for stepper_method
            ValueError: if ground state is not solved prior to calling this method

        Returns:
            saved_psi (list): list of snapshots of psi during the time evolution 
        """
        shift = int(self.constants.M/2)
        shifted_psi = np.roll(psi, shift)  # TODO: don't use roll because it wraps array around grid
        interference_psi = psi + shifted_psi
        return self.time_evolve(interference_psi, stepper_method, dt, Nt, snapshots, adaptive_step, dt_min, err_tol)
    
    def time_evolve_interference2(self, psi_1:np.array, psi_2:np.array, stepper_method: str, dt: float, Nt: int, snapshots:int, adaptive_step: bool, dt_min: float, err_tol: tuple = (0,1)) -> list:   
        """
        evolves the interference between the ground state wavefunction and a duplicate wavefunction 
        shifted by some amount. Note that the external potential (self.ext_V) must be changed or set
        to zero in order to observe changes in the wavefunction across time. 

        Args:
            ground_psi (np.array):  the ground state wavefunction of the BEC
            stepper_method (str):   method used to perform the integration (RK4IP or SSFM)
            dt (float):             initial time step (remains constant if adaptive_step is False)
            Nt (int):               number of times to evolve psi by the initial dt
            snapshots (int):        number of periodical snapshots of psi to save in memory
            adaptive_step (bool):   if True, will adapt dt based on value of err_tol
            dt_min (float):         adaptive step will not reduce dt below dt_min
            err_tol (tuple):        acceptable range of error between evolving psi by dt once and
                                    evolving psi by dt/2 twice. Defaults to (0,1).

        Raises:
            ValueError: if either 'RK4IP' or 'SSFM' are not chosen for stepper_method
            ValueError: if ground state is not solved prior to calling this method

        Returns:
            saved_psi (list): list of snapshots of psi during the time evolution 
        """
        interference_psi = psi_1 + psi_2
        return self.time_evolve(interference_psi, stepper_method, dt, Nt, snapshots, adaptive_step, dt_min, err_tol)
 

    def solve_ground_state(self, dt: float, lower_bound: float) -> np.array:
        """
        evolves the guessed wavefunction in imaginary time to solve for the ground state

        Args:
            dt (float): time step
            lower_bound (float): perecentage difference in chemical potential between successive wavefunctions
                                 below which the ground state is assumed to be found

        Returns:
            psi (np.array): ground state wavefunction
        """
        psi = self.guessed_psi
        exp_D = self.set_exp_D(dt, 1)
        norm_squared_psi = np.linalg.norm(psi)**2
        psi_mid_old = self.find_midpoint(psi)
        mu_old, mu_err = 1, 1
        idx = 0
        while mu_err > lower_bound:
            psi = self.SSFM(psi, dt, exp_D, 1)
            
            psi_mid_new = self.find_midpoint(psi)           
            mu_new = np.log(psi_mid_old / psi_mid_new) / dt  # TODO: check dimensionality
            mu_err = np.absolute((mu_new - mu_old) / mu_new)
            # print(psi_mid_new,psi_mid_old, 'idx ====', idx)
            psi_mid_old = psi_mid_new
            mu_old = mu_new
            idx += 1

            psi = self.normalize(norm_squared_psi, psi)  # normalizing magnitude

            if idx > 1e6:
                print('no solution found')
                break
        self.ground_psi = psi
        print('\n done \n')
        return psi
    
    def normalize(self, norm_squared_psi: float, psi: np.array) -> np.array:
        """
        makes |psi|^2 equal to norm_squared_psi

        Args:
            norm_squared_psi (float): magnitude squared of the initial wavefunction
            psi (np.array): new wavefunction 

        Returns:
            np.array: wavefuction with equal magnitude squared as norm_squared_psi
        """
        return psi * np.sqrt(norm_squared_psi) / np.sqrt(np.linalg.norm(psi)**2)

    def find_midpoint(self, psi: np.array) -> float:
        """
        gets the midpoint value of psi based on the dimension of the computational grid

        Args:
            psi (np.array): wavefunction of the BEC

        Returns:
            float: midpoint of psi
        """
        middle_idx = int(self.constants.M)
        if self.constants.dim == 1:
            return psi[middle_idx]
        elif self.constants.dim == 2:
            return psi[middle_idx][middle_idx]
        else:
            return psi[middle_idx][middle_idx][middle_idx]
         

    def adapt_step(self, psi_1: np.array, psi_2: np.array, dt:float,  dt_min: float, err_tol: tuple, factor: float = 1.3):
        """
        increases or decreases dt based on the error produced during integration. dt is reduced if error
        is above error tolerance limit and vice versa. dt is unchanged if error is within error tolerance
        range. 

        Args:
            psi_1 (np.array):  result from evolving psi once using dt
            psi_2 (np.array): result from evolving psi twice using dt/2
            dt (float):  time step
            dt_min (float): adaptive step mechanism will not reduce dt below dt_min
            err_tol (tuple): acceptable range of error between evolving psi by dt once and
                            evolving psi by dt/2 twice. Defaults to (0,1).
            factor (float, optional): factor to increase dt. Defaults to 1.3.

        Returns:
            dt (float): new dt
            update_psi (bool): whether to use the previously evolved psi or to try again with smaller dt
        """
        denom = np.linalg.norm(psi_2)
        if denom != 0:
            error = np.abs(np.linalg.norm(psi_2 - psi_1) / denom)
        else:
            error = np.linalg.norm(psi_2 - psi_1)
        
        # error tolerance exceeded
        if error > err_tol[1]:                              
            if dt == dt_min:
                update_psi = True
            elif dt / 2 < dt_min:
                update_psi = False
                dt == dt_min
            else:
                update_psi = False
                dt = dt / 2
        # error within tolerance range
        elif error >= err_tol[0] and error <= err_tol[1]:   
            update_psi = True
        # error below tolerance range
        else:                                               
            print(f'changing dt {dt}')
            update_psi = True
            dt = dt * factor
        return dt, update_psi




class Visualization:
    def __init__(self, solution: Solution) -> None:
        self.sol = solution
    
    def guessed_vs_ground_psi_plot(self) -> None:
        """
        compares the initial guess of the psi with the ground state solution
        """
        if self.sol.constants.dim == 1:
            self.psi_density_plot_2d(self.sol.guessed_psi, label='guessed psi')
            self.psi_density_plot_2d(self.sol.ground_psi, label='solved psi')
            plt.xlabel('x')
            plt.ylabel('|ψ|²') 
            plt.show()
        elif self.sol.constants.dim == 2:
            self.psi_density_plot_3d(self.sol.guessed_psi)
            plt.show()
            self.psi_density_plot_3d(self.sol.ground_psi)
            plt.show()
        
    def guessed_psi_plot(self):
        if self.sol.constants.dim == 1:
            self.psi_density_plot_2d(self.sol.guessed_psi, label='guessed psi')
            plt.xlabel('x')
            plt.ylabel('|ψ|²') 
            plt.show()
        elif self.sol.constants.dim == 2:
            self.psi_density_plot_3d(self.sol.guessed_psi)
            plt.show()
    
    def psi_density_plot_2d(self, psi: np.array, label:str='') -> None:
        assert(self.sol.constants.dim == 1)
        # TODO: titles, axis labels and such
        plt.plot(self.sol.positions[0], np.absolute(psi)**2,label=label)
        plt.legend()

    
    def psi_density_plot_3d(self, psi: np.array,title:str='') -> None:
        # TODO: titles, axis labels and such
        assert(self.sol.constants.dim > 1)
        if self.sol.constants.dim == 2:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(
                self.sol.positions[0], self.sol.positions[1], np.absolute(psi)**2,
                cmap=cm.coolwarm, linewidth=1, antialiased=False
            )
            fig.suptitle(title)

    def psi_snapshots_plot_2d(self, plot_count:int) -> None:
        """
        plots multiple snapshots of psi accross time

        Args:
            plot_count (int): number of snapshots to include
        """
        assert(self.sol.constants.dim == 1)
        # plot_count = 5  # number of snapshots of |psi(x,t)|^2 to show in the plot
        interval = max(int(len(self.sol.saved_psi) / plot_count), 1)
        for t, psi_t in zip(self.sol.saved_t[::interval], self.sol.saved_psi[::interval]):
            psi_t_abs = np.absolute(psi_t)**2  # probability density across space at time t
            plt.plot(self.sol.positions[0], psi_t_abs, label = f't={t:.3f}') 
        plt.xlabel('x')
        plt.ylabel('|ψ|²') 
        plt.legend()
        plt.show()

    def heat_graph(self, plot_count: int) -> None:
        """
        plots heat maps of multiple snapshots of psi across time

        Args:
            plot_count (int): number of snapshots to include
        """
        assert(self.sol.constants.dim in (2,3))
        saved_psi = self.sol.saved_psi
        saved_t = self.sol.saved_t
        interval = math.floor(len(saved_psi)/(plot_count)) # idx interval
        plot_x_len, plot_y_len = 3, math.floor(plot_count/3)  # plot grid x and y length
        fig, axs = plt.subplots(plot_x_len, plot_y_len, sharex=True)
        for idx in range(plot_count): 
            snapshot = saved_psi[idx * interval]
            t = saved_t[(idx * interval) % 99]
            x_cord, y_cord = idx // plot_y_len, idx % plot_y_len
            if self.sol.constants.dim == 2:
                axs[x_cord, y_cord].contourf(
                    self.sol.positions[0], self.sol.positions[1], np.absolute(snapshot)**2, 
                    20, cmap='coolwarm'
                )
            else:
                axs[x_cord, y_cord].scatter(
                    self.sol.positions[0], self.sol.positions[1], self.sol.positions[2], 
                    c=np.absolute(snapshot)**2, cmap=plt.hot())
            axs[x_cord, y_cord].set_title(f'Fig ({idx}): time {t:.1f}s')
            axs[x_cord, y_cord].ticklabel_format(axis="both", style="sci", scilimits=(0,0))
            axs[x_cord, y_cord].ticklabel_format()
            axs[x_cord, y_cord].xaxis.set_major_locator(MaxNLocator(integer=True))
        for ax in axs.flat:
            ax.set(xlabel='X', ylabel='Y')
        # fig.delaxes(axs[2][2])
        # fig.delaxes(axs[2][1])
        fig.set_figheight(11)
        fig.set_figwidth(14)
        fig.suptitle('heat graph of 2d solution')
        plt.show()


def V(constans: Constants,*positions) -> np.array:
    """
    Example potential function to pass in to Solution. Symmetrical harmonic oscillator
    centered at x = 0

    Args:
        constans (Constants): use the same instance of Constants class used for Solution

    Returns:
        np.array: potential grid with shape of spatial grid 
    """
    e =  np.ones(3)  # equal contribution from all three dimesnions
    factor = 0.5*constans.mass*constans.wx**2 
    return sum([factor * e[idx] * ((pos)**2) for idx, pos in enumerate(positions)])

def V_symmetric(constans: Constants,*positions) -> np.array:
    """
    Example potential function to pass in to Solution. Symmetrical harmonic oscillator
    centered at x = 0

    Args:
        constans (Constants): use the same instance of Constants class used for Solution

    Returns:
        np.array: potential grid with shape of spatial grid 
    """
    e =  np.ones(3)  # equal contribution from all three dimesnions
    factor = 0.5*constans.mass*constans.wx**2 
    return sum([factor * e[idx] * ((pos)**2) for idx, pos in enumerate(positions)])

def V_shifted(constans: Constants,*positions) -> np.array:
    """
    Example potential function to pass in to Solution. Symmetrical harmonic oscillator
    centered at x = 3
    Args:
        constans (Constants): use the same instance of Constants class used for Solution

    Returns:
        np.array: potential grid with shape of spatial grid 
    """
    e =  np.ones(3)  # equal contribution from all three dimesnions
    factor = 0.5*constans.mass*constans.wx**2
    return sum([factor * e[idx] * ((pos + 3)**2) for idx, pos in enumerate(positions)])

def V_cigar_shaped(constans: Constants,*positions) -> np.array:
    """
    Example potential function to pass in to Solution. Cigar shaped harmonic oscillator
    centered at x = 0

    Args:
        constans (Constants): use the same instance of Constants class used for Solution

    Returns:
        np.array: potential grid with shape of spatial grid 
    """
    e =  np.array((1, 3, 3))  # equal contribution from just two dimesnions
    factor = 0.5*constans.mass*constans.wx**2 
    return sum([factor * e[idx] * ((pos)**2) for idx, pos in enumerate(positions)])