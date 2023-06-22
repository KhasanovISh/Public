# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:56:28 2021.

@author: Tigrus

Library for calculus in multilayers scheme with gradient layer v 2.0 release
ver 11.08.2021
"""


from .SuPPPort import *


class ExperimentSPR:
    """Experiment class for numeric calculus."""

    grad_resolution = 100  # resolution for gradient layer calculation
    wavelength = 0

    def __init__(self, polarisation='p'):
        """Init empty."""
        self.layers = dict()
        self.k0 = 2.0 * np.pi / 1e-6
        self.polarisation = polarisation

        # for system conserving
        self.wl_asylum = dict()
        self.k0_asylum = 1e-6

    def __setattr__(self, name, val):
        """Sync wavelength and k0."""
        if name == "k0":
            self.__dict__["k0"] = val
            self.__dict__["wavelength"] = 2 * np.pi / val
        elif name == "wavelength":
            self.__dict__["wavelength"] = val
            self.__dict__["k0"] = 2 * np.pi / val
        else: self.__dict__[name] = val

    def __getattr__(self, attrname):
        """Getter for n and d."""
        if attrname == "n":
            val = []
            for L in range(0, len(self.layers)):
                if isinstance(self.layers[L].n, MaterialDispersion):
                    # Parametric metal
                    val.append(self.layers[L].n.CRI(self.wavelength))
                elif isinstance(self.layers[L].n, FunctionType):
                    # Gradient layer fubction
                    val.append(self.layers[L].n(0))
                elif isinstance(self.layers[L].n, Anisotropic):
                    # Anisotropic layer fubction
                    val.append(self.layers[L].n.n0)
                else:
                    # Homogenious
                    val.append(self.layers[L].n)
            return val
        if attrname == "d":
            val = [0]
            for L in range(1, len(self.layers)-1):
                val.append(self.layers[L].thickness)
            val.append(0)
            return val

    def Save_par(self):
        """Conserving scheme parametrs."""
        self.wl_asylum = self.layers.copy()
        self.k0_asylum = self.k0

    def Load_par(self):
        """Rescuing scheme parametrs."""
        self.layers = self.wl_asylum.copy()
        self.k0 = self.k0_asylum

    # -----------------------------------------------------------------------
    # --------------- Work with layers --------------------------------------
    # -----------------------------------------------------------------------

    def add(self, new_layer):
        """Add one layer.

        Parameters
        ----------
        permittivity : complex, Metall_CRI or lambda
            permittivity for layer.
        thickness : float
            layer thickness.
        """
        self.layers[len(self.layers)] = new_layer

    def delete(self, num):
        """Delete one layer.

        Parameters
        ----------
        num : int
            layer number.
        """
        if num < 0 or num > len(self.layers)-1:
            print("Deleting layer out of bounds!")
            return
        if num == len(self.layers) - 1:
            self.layers.pop(num)
        else:
            for i in range(num, len(self.layers)-1):
                self.layers[i] = self.layers.pop(i+1)

    def insert(self, num, new_layer):
        """Insert layer layer

        Parameters
        ----------
        num : int
            layer number to insert.
        new_layer : [array]
            [permittivity, thickness]
        """
        if num < 0 or num > len(self.layers)-1:
            print("Inserting layer out of bounds! Layer add in the end of the list")
            self.add(new_layer)
        else:
            for i in range(0, len(self.layers) - num + 1):
                self.layers[len(self.layers) - i] = self.layers[len(self.layers) - i - 1]
            self.layers[num] = new_layer

    # -----------------------------------------------------------------------
    # --------------- Profiles calculations ---------------------------------
    # -----------------------------------------------------------------------

    def R(self, angles=None, wavelenghts=None, angle=None, is_complex=false):
        """Representation for every R.

        Parameters
        ----------
        angles : arary, optional
            angles range. The default is None.
        wavelenghts : arary, optional
            wavelenghts range. The default is None.
        angle : float, optional
            angle for r(lambda). The default is None.
        is_complex : boolean, optional
            return real or complex. The default is false.

        Returns
        -------
        arary
            array of R.

        """
        if angles is not None:
            if is_complex:
                return self.R_theta_cmpl(angles)
            else:
                return self.R_theta_re(angles)
        if wavelenghts is not None:
            if angle is None:
                print("Angle not defined!")
                return None
            if is_complex:
                return self.R_lambda_cmpl(angle, wavelenghts)
            else:
                return self.R_lambda_re(angle, wavelenghts)   
        print("Parametrs do not defined!")
        return

    def R_theta_re(self, degree_range):
        """Parameters.

        degree_range : range(start, end, seps)
            range of function definition in degree .

        Returns
        -------
        Rr : array[float]
            reflection array in range.
        """
        Rr = [np.abs(self.R_deg(theta))**2 for theta in degree_range]
        return Rr

    def R_theta_cmpl(self, degree_range):
        """Parameters.

        degree_range : range(start, end, seps)
            range of function definition in degree .

        Returns
        -------
        Rr : array[complex]
            reflection array in range.
        """
        Rr = [self.R_deg(theta) for theta in degree_range]
        return Rr

    def R_lambda_re(self, angle_grad, lambda_range):
        """Parameters not work yet.

        angle_grad : float
            angle of calculus in [0 - 90]
        lambda_range : range(start, end, seps)
            range of function definition.

        Returns
        -------
        Rr : array[float]
            reflection array in range.
        """
        Rr = []
        for i in lambda_range:
            self.wavelength = i
            Rr.append(np.abs(self.R_deg(angle_grad))**2)
        return Rr

    def R_lambda_cmpl(self, angle_grad, lambda_range):
        """Parameters not work yet.

        angle_grad : float
            angle of calculus in [0 - 90]
        lambda_range : range(start, end, seps)
            range of function definition.

        Returns
        -------
        Rr : array[complex]
            reflection array in range.
        """
        Rr = []
        for i in lambda_range:
            self.wavelength = i
            Rr.append(self.R_deg(angle_grad))
        return Rr

    def R_deg(self, theta):
        """Parameters.

        theta : int
            angle to calculate on.

        Returns.
        -------
        R : float
            complex reflection in selected angle.
        """
        theta = np.pi * theta / 180
        n = self.n
        d = self.d
        kx0_sqrt = self.k0 * n[0] * np.sin(theta)
        kx0 = np.power(kx0_sqrt, 2)

        k_z = [SM.sqrt(np.power(self.k0*n[i], 2) - kx0) for i in range(1, len(n))]

        # k_z[grad] valid only for input!
        k_z.insert(0, np.sqrt(np.power(self.k0*n[0], 2) - kx0))
 
        # find r and t for normal layers
        if self.polarisation == 'p':
            rp = [(k_z[i]*n[i+1]**2 - k_z[i+1]*n[i]**2) /
                 (k_z[i]*n[i+1]**2 + k_z[i+1]*n[i]**2)
                 for i in range(0, len(n)-1)]

            tp = [2*(k_z[i]*n[i]**2) /
                 (k_z[i]*n[i+1]**2 + k_z[i+1]*n[i]**2)
                 for i in range(0, len(n)-1)]

        if self.polarisation == 's':
            rs = [(k_z[i] - k_z[i+1]) /
                 (k_z[i] + k_z[i+1])
                 for i in range(0, len(n)-1)]

            ts = [2*(k_z[i]) /
                 (k_z[i] + k_z[i+1])
                 for i in range(0, len(n)-1)]

        # All layers for p
        if self.polarisation == 'p':
            # reflectivities for anisotropic layers for this angles for 'p'
            # 's' dont feels extraordinary n
            for i in range(1, len(n)-1):
                if isinstance(self.layers[i].n, Anisotropic):
                    # before layer
                    if not isinstance(self.layers[i - 1].n, Anisotropic):
                        # if previous is anisotropic - r  is modified in prev step
                        rp[i-1] = self.layers[i].n.r_in(n[i-1], kx0_sqrt, self.k0)
                    # after layer
                    if not isinstance(self.layers[i + 1].n, Anisotropic):
                        # with next isotropic layer
                        rp[i] = self.layers[i].n.r_out(n[i+1], kx0_sqrt, self.k0)
                    else:
                        # with next anisotropic layer
                        x1 = self.layers[i].n.p_div_q(kx0_sqrt, self.k0)
                        x2 = self.layers[i+1].n.p_div_q(kx0_sqrt, self.k0)
                        rp[i] = (x2 - x1) / (x2 + x1)

            M0 = np.array([[1/tp[0], rp[0]/tp[0]], [rp[0]/tp[0], 1/tp[0]]])
            for i in range(1, len(n)-1):
                if isinstance(self.layers[i].n, Anisotropic):
                    # go through and out anisotropic layer
                    kz_pl = self.layers[i].n.kz_plus(kx0_sqrt, self.k0)
                    kz_mn = self.layers[i].n.kz_minus(kx0_sqrt, self.k0)
                    Mi = np.array([[np.exp(-1j*kz_pl*d[i])/tp[i],
                                    np.exp(-1j*kz_pl*d[i])*rp[i]/tp[i]],
                                   [np.exp(1j*kz_mn*d[i])*rp[i]/tp[i],
                                    np.exp(1j*kz_mn*d[i])/tp[i]]])
                elif isinstance(self.layers[i].n, FunctionType):
                    # Gradient layer
                    Mi = self.GradLayerMatrix(theta, n, d, i)
                else:
                    # Normal layer
                    Mi = np.array([[np.exp(-1j*k_z[i]*d[i])/tp[i],
                                    np.exp(-1j*k_z[i]*d[i])*rp[i]/tp[i]],
                                   [np.exp(1j*k_z[i]*d[i])*rp[i]/tp[i],
                                    np.exp(1j*k_z[i]*d[i])/tp[i]]])
                M0 = M0@Mi
            if M0[0, 0] == 0:
                R = 1
            else:
                R = M0[1, 0]/M0[0, 0]

        # All layers for 's'
        else:
            if self.polarisation_ratio == 0:
                Rs = 0
            else:
                M0 = np.array([[1/ts[0], rs[0]/ts[0]], [rs[0]/ts[0], 1/ts[0]]])
                for i in range(1, len(n)-1):
                    if isinstance(self.layers[i].n, FunctionType):
                        # Gradient layer
                        Mi = self.GradLayerMatrix(theta, n, d, i)
                    else:
                        # Normal layer
                        Mi = np.array([[np.exp(-1j*k_z[i]*d[i])/ts[i],
                                        np.exp(-1j*k_z[i]*d[i])*rs[i]/ts[i]],
                                       [np.exp(1j*k_z[i]*d[i])*rs[i]/ts[i],
                                        np.exp(1j*k_z[i]*d[i])/ts[i]]])
                    M0 = M0@Mi
                if M0[0, 0] == 0:
                    R = 1
                else:
                    R = M0[1, 0]/M0[0, 0]
        return R

    def GradLayerMatrix(self, theta, n, d, grad_num):
        """Include output but not input layer.

        Parameters.
        theta : int
            angle to calculate on.
        n : array[float]
            refractive index array from 0 to N.
        d : array[float]
            layers thickness with d[0]=d[n]=0 on semiinfinite lborder layers.
        number : int
            gradient layer number.

        Returns.
        -------
        Mtot : matrix [2,2]
            gradient layer matrix to reflection calculus.
        """
        Mtot = np.array([[1, 0], [0, 1]])
        dx = d[grad_num]/self.grad_resolution
        kx0 = (self.k0 * n[0] * np.sin(theta))**2
        n_range = np.linspace(0, 1, self.grad_resolution)

        ngrad = self.layers[grad_num].n(n_range)

        for i in range(1, self.grad_resolution):
            ni = ngrad[i-1]
            ni1 = ngrad[i]

            ki = SM.sqrt((self.k0*ni)**2 - kx0)
            ki1 = SM.sqrt((self.k0*ni1)**2 - kx0)
            if self.polarisation == 'p':
                a = ki * ni1**2
                b = ki1 * ni**2
            else:
                a = ki
                b = ki1
            r = (a - b) / (a + b)
            t = 2*a / (a + b)
            kidx = ki*dx
            M = np.array([[np.exp(-1j*kidx) / t,
                           np.exp(-1j*kidx)*r / t],
                          [np.exp(1j*kidx)*r / t,
                           np.exp(1j*kidx) / t]])
            Mtot = Mtot@M

        # Output layer
        ni = ni1
        ni1 = n[grad_num + 1]
        ki = SM.sqrt((self.k0*ni)**2 - kx0)
        ki1 = SM.sqrt((self.k0*ni1)**2 - kx0)
        if self.polarisation == 'p':        
            r = (ki * ni1**2 - ki1 * ni**2) / (ki * ni1**2 + ki1 * ni**2)
            t = (2 * ki * ni1**2) / (ki * ni1**2 + ki1 * ni**2)
        else:
            r = (ki - ki1) / (ki + ki1)
            t = (2 * ki) / (ki + ki1)
        M = np.array([[np.exp(-1j*ki*dx)/t, np.exp(-1j*ki*dx)*r/t],
                     [np.exp(1j*ki*dx)*r/t, np.exp(1j*ki*dx)/t]])
        Mtot = Mtot@M
        return Mtot

    # -----------------------------------------------------------------------
    # --------------- Gradient layer restoration ----------------------------
    # -----------------------------------------------------------------------

    startsearchpoint = 1.1
    boundssearch = (1, 1.2)
    count_min = 2

    def Difference_homogeniety(self, N_array):
        """Difference in homogeneity in R(ϴ).

        Parameters
        ----------
        N_array : array n[i]
            Set of homogeniuos layers.

        Returns
        -------
        float
            Difference.
        """
        # change permittivities
        for i in range(0, len(N_array)):
            self.layers[self.grad_num + i][0] = N_array[i]

        # find discrepancy with new parametrs
        Rr = self.R_theta_re(self.my_reflections_data[0])
        Diff = [(Rr[i] - self.my_reflections_data[1][i])**2
                for i in range(len(self.my_reflections_data[0]))]
        return sum(Diff)

    def Difference_homogeniety_3d(self, N_array):
        """Difference in homogeneity in R(ϴ, λ).

        Parameters
        ----------
        N_array : array n[i]
            Set of homogeniuos layers.

        Returns
        -------
        float
            Difference.
        """
        # chang
        # change permittivities
        for i in range(0, len(N_array)):
            self.layers[self.grad_num + i][0] = N_array[i]

        # find new parametrs
        Rmin_curve = np.array(self.Get_SPR_curve(self.my_reflections_data[:,0]))

        # Discrepancy
        Diff = [((Rmin_curve[i][1]/self.my_reflections_data[i][1] - 1)**2 +
                 (Rmin_curve[i][2]/self.my_reflections_data[i][2] - 1)**2)
                for i in range(0, len(self.my_reflections_data[:,0]))]
        return sum(Diff)

    def Difference_homogeniety_4d(self, N_array):
        """Difference in homogeneity in set R(ϴ) in range of λ.

        Parameters
        ----------
        N_array : array n[i]
            Set of homogeniuos layers.

        Returns
        -------
        float
            Difference.
        """
        # change permittivities
        for i in range(0, len(N_array)):
            self.layers[self.grad_num + i][0] = N_array[i]

        Diff = 0
        # find discrepancy with new parametrs
            
        for key, value in self.my_reflections_data.items():
            if isinstance(key, float):
                self.wavelength = key
                Rr = self.R_theta_re(self.my_reflections_data['theta'])
                Diff += sum([(Rr[i] - value[i])**2 for i in range(len(value))])

        return Diff

    def Restore_Grad(self, method, data, resolution,
                     layer_num, plot_graphics=False):
        """Grad layer restoring from R(theta).

        Parameters
        ----------
        Method: string
            2d 3d of 4d
        Data : array
            [theta, R], [theta, λ, R] or , [theta, R(λ1), R(λ1) ...]
        Resolution : int
            max layers.
        Layer_num: int
            layer to restore profile
        Plot_graphics: bool
            Show or not graphics in every step

        Returns
        -------
        list
            DESCRIPTION.
        """
        # Check conditions
        self.grad_num = layer_num
        if self.grad_num < 1 or self.grad_num >= len(self.layers)-1:
            print('Gradient layer not detected!')
            return [0]

        if method == "2d": Dif_func = self.Difference_homogeniety
        elif method == "3d": Dif_func = self.Difference_homogeniety_3d
        elif method == "4d": Dif_func = self.Difference_homogeniety_4d
        else:
            print("Error! Not valid method is specified!")
            return[0]

        if resolution < 2:
            resolution = 2
        if resolution > 50:
            resolution = 50

        initial_guess = []

        # Begin if it's all right!
        self.my_reflections_data = data

        # from 2 increasing layers count
        for number_of_layers in range(self.count_min, resolution):
            self.Save_par()

            # Increase dimension of initial guess with every step
            if len(initial_guess) == 0:
                initial_guess = [self.startsearchpoint]*self.count_min
            else:
                x = np.linspace(0, 1, num=len(initial_guess))
                f = sp.interpolate.interp1d(x, initial_guess)
                xnew = np.linspace(0, 1, num=number_of_layers)
                initial_guess = f(xnew)

            # Delete gradient and add set of new monolayers
            new_layer_height = self.layers[self.grad_num].thickness/number_of_layers
            self.Del_layer(self.grad_num)
            for i in range(0, number_of_layers):
                self.Ins_layer(self.grad_num, [0, new_layer_height])

            # Minimize descreapancy with these scheme
            bounds = tuple([self.boundssearch]*number_of_layers)
            res = minimize(Dif_func, initial_guess, method='BFGS') # BFGS
            initial_guess = res.x
            print(f'Step = {number_of_layers}, gradient: {initial_guess}')

            # Optional: show visual results every step
            if plot_graphics:
                for i in range(0, len(initial_guess)):
                    self.layers[self.grad_num + i][0] = initial_guess[i]
                x = np.linspace(0, 1, num=len(initial_guess))
                f = sp.interpolate.interp1d(x, initial_guess)
                n_profile(f, name=f"Gradient on step {number_of_layers}", dpi=200)
                # if method == "2d":          
                #     plot_graph(self.my_reflections_data[0], self.R_theta_re(self.my_reflections_data[0]))
                # if method == "3d":
                #     self.Plot_SPR_curve(self.Get_SPR_curve(self.my_reflections_data[:,0]), True)
                # if method == "4d":
                #     fig = plt.figure(dpi=200)
                #     ax = fig.gca()
                #     for key, value in data.items():
                #         if isinstance(key, float):
                #             ax.plot(data["theta"], value, label="λ={:.2}nm".format(key*1e6))
                #     ax.set_xlabel('θ, °')
                #     ax.set_ylabel('R')
                #     plt.title(f"Profiles on step {number_of_layers}")
                #     plt.legend(loc='best', prop={'size': 7})
                #     ax.grid()
                #     plt.show()

            self.Load_par()

        x = np.linspace(0, 1, num=len(initial_guess))
        f = sp.interpolate.interp1d(x, initial_guess)
        return f

    # -----------------------------------------------------------------------
    # --------------- secondary functions -----------------------------------
    # -----------------------------------------------------------------------

    bounds2dsearch = (15, 28)

    def Get_SPR_curve(self, lambda_range):
        """Get minimum R(ϴ, λ) for actual set in range.

        Parameters
        ----------
        lambda_range : range
            Range to search.

        Returns
        -------
        array
            [λ, ϴ(SPP), R(SPP)]
        """
        Rmin_curve = []
        # bnds = self.bounds2dsearch

        self.k0_asylum = self.k0

        for wl in lambda_range:
            self.wavelength = wl

            # lambda func R(theta) and search of minimum
            R_dif = lambda th: np.abs(self.R_deg(th))**2
            theta_min = minimize_scalar(R_dif, bounds=self.bounds2dsearch, method='Bounded')

            # minimum value
            Rw_min = np.abs(self.R_deg(theta_min.x))**2

            # bnds = (theta_min.x - 0.5, theta_min.x + 0.5)
            Rmin_curve.append([wl, theta_min.x, Rw_min])

        self.k0 = self.k0_asylum
        return np.array(Rmin_curve)

    def TIR(self):
        """Return Gives angle of total internal reflecion."""
        # initial conditions
        warning = None
        TIR_ang = 0
        if (sum(self.d) > 2.0 * self.wavelength):
            warning = 'Warning! System too thick to determine\
                total internal reflection angle explicitly!'

        # Otto scheme is when last layer is metal
        if self.n[len(self.n)-1].real < self.n[0].real:
            TIR_ang = np.arcsin(self.n[len(self.n)-1].real/self.n[0].real)
        else:
            # Kretchman scheme is when second layer is metal
            if self.n[1].real > self.n[0].real:
                warning = 'Warning! System too complicated to\
                    determine total internal reflection angle explicitly!'
            for a in range(1, len(self.n)-1):
                if self.n[a].real < self.n[0].real:
                    TIR_ang = np.arcsin(self.n[a].real /
                                        self.n[0].real)
                    break

        # If not found
        if TIR_ang == 0:
            warning = 'Warning! Total internal\
                reflection not occures in that system'
            TIR_ang = np.pi/2

        # if warnings occure
        if warning is not None:
            print(warning)

        return 180*TIR_ang/(np.pi)

    def show_info(self):
        """Show set parametrs."""
        print(" --- Unit parametrs ---")
        print("w: ", self.k0)
        print("λ: ", self.wavelength)
        print("n: ", self.n)
        print("d: ", self.d)
        self.Plot_Grad(dpi=200)

    def Plot_Grad(self, dpi=None):
        """Plot all gradient layers in set."""
        found = False
        for L in range(0, len(self.layers)):
            if isinstance(self.layers[L].n, FunctionType):
                if self.layers[L].name == None:
                    title = f"Gradient profile in layer #{L}"
                else:
                    title = self.layers[L].name
                n_profile(self.layers[L].n,
                          name=title, dpi=dpi)
                found = True
        if not found: print("No gradient layers found.")
        else: print("Gradient profiles shown in plots.")

    def Plot_SPR_curve(self, Rmin_curve, plot_2d=False, view_angle=None):
        """Plot R(ϴ, λ).

        Parameters
        ----------
        Rmin_curve : array
            [λ, ϴ(SPP), R(SPP)].
        plot_2d : bool, optional
            If plots R(ϴ) and R(λ)are shown. The default is False.
        view_angle : float, optional
            angle to rotate 3d. The default is None.
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xs=Rmin_curve[:, 0], ys=Rmin_curve[:, 1], zs=Rmin_curve[:, 2])
        ax.set_zlim3d(0, max(Rmin_curve[:, 2]))

        ax.set_xlabel('λ , nm')
        ax.set_ylabel('θ, °')
        ax.set_zlabel('R')
        if view_angle is not None:
            ax.view_init(view_angle[0], view_angle[1])

        plt.show()

        if plot_2d:
            fig, ax = plt.subplots()
            ax.grid()
            ax.plot(Rmin_curve[:, 0], Rmin_curve[:, 1])
            ax.set_xlabel('λ , nm')
            ax.set_ylabel('θ, °')
            plt.show()

            fig, ax = plt.subplots()
            ax.grid()
            ax.plot(Rmin_curve[:, 0], Rmin_curve[:, 2])
            ax.set_xlabel('λ , nm')
            ax.set_ylabel('R')
            plt.show()


def main(): print('This is library, you can\'t run it :)')


if __name__ == "__main__": main()
