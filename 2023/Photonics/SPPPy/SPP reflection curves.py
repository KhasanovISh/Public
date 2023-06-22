# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:39:02 2021.

@author: THzLab
"""

# Импортируем все из библиотеки TKinter
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from SPPPy import *
from SPPPy import Layer as LYR

window = Tk()


# --------------------------------------------------------------------------
# -------------------- Classes ---------------------------------------------
# --------------------------------------------------------------------------


class VerticalScrolledFrame(Frame):
    """ Honestly stealed from internet."""
    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)            
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior, anchor=NW)
        def _configure_interior(event):
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)
        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)


class Parametr:
    """Parametr type.

    Contains name, Scale
    """

    def __init__(self, my_frame, name, min_val=1, max_val=10, step=1,
                 sync_pars=(None, None, None), dim_box=None):
        """Create all forms.

        Parameters
        ----------
        my_frame : Frame
            External frame to contain all elements.
        name : string
            Name label text.
        min_val : float, optional
            minimum scale value. The default is 1.
        max_val : float, optional
            miximum scale value. The default is 10.
        step : float, optional
            Scale step. The default is 1.
        sync_pars : tuple[3]
            (sync_scale, sync_text, sync_dim)
        dim_box : dict
            text and value of dim
        """
        self.my_frame = my_frame
        self.minimum = min_val
        self.maximum = max_val

        self.frm_x = Frame(my_frame, width=20, height=500)
        self.frm_x.pack(fill=X)

        self.lbl_x = Label(self.frm_x, text=f"{name} = ", font=("Calibri", 9))
        self.lbl_x.pack(side=LEFT)

        self.scl_x = Scale(self.frm_x, orient=HORIZONTAL, showvalue=False,
            sliderlength=10, from_=self.minimum, to=self.maximum,
            resolution=step, command=self.scl_x_change, variable = sync_pars[0])
        self.scl_x.pack(side=LEFT)

        # Look if text labels is connected
        self.txt_x = StringVar()
        if sync_pars[1] is None:
            self.txt_x = StringVar()
        else:
            self.txt_x = sync_pars[1]

        self.txt_x.trace_add('write', self.ent_x_change)
        self.ent_x = Entry(self.frm_x, textvariable=self.txt_x, width=6)
        self.ent_x.insert(0, 1)
        self.ent_x.pack(side=LEFT)

        self.dim_box = dim_box
        if dim_box is not None:
            self.dim_box_x = ttk.Combobox(self.frm_x, values=list(dim_box),
                    width=6, state="readonly", textvariable=sync_pars[2])
            self.dim_box_x.current(0)
            self.dim_box_x.pack(side=LEFT, padx="4")
        
    def ent_x_change(self, *args):
        """Move info from Scale to entry."""
        if self.ent_x.get() != '' and (self.scl_x.get() != float(self.ent_x.get())):
            my_txt = self.ent_x.get()
            if float(my_txt) > self.maximum:
                my_txt = self.maximum
                self.ent_x.delete(0, END)
                self.ent_x.insert(0, my_txt)
            if float(my_txt) < self.minimum:
                my_txt = self.minimum
                self.ent_x.delete(0, END)
                self.ent_x.insert(0, my_txt)

            self.scl_x.set(my_txt)

            #Autodraw
            for i in range(0,len(R_tabs)):
                if R_tabs[i].chk_state.get():
                    R_tabs[i].build_curve()

    def scl_x_change(self, *args):
        """Move info from Entry to Scale."""
        if (float(self.scl_x.get()) != float(self.ent_x.get())):
            self.ent_x.delete(0, END)
            self.ent_x.insert(0, self.scl_x.get())

            #Autodraw
            for i in range(0,len(R_tabs)):
                if R_tabs[i].chk_state.get():
                    R_tabs[i].build_curve()

    def get(self):
        """Get parametr value.

        Returns
        -------
        a : float
            value from scale, if it defineв.
        """
        a = self.scl_x.get()
        if self.dim_box is not None:
            a = a * self.dim_box[self.dim_box_x.get()]
        return a

    def set_par(self, par):
        """Set parametr value.

        Parameters
        ----------
        par : int
            Parametr value.
        """
        if par < self.scl_x.cget("from") or par > self.scl_x.cget("to"):
            for key, value in self.dim_box.items():
                if par/value > self.scl_x.cget("from") and par/value < self.scl_x.cget("to"):

                    self.scl_x.set(par/value)
                    self.dim_box_x.set(key)
                    return
            print("ERRORR in set: ", par)
        else:self.scl_x.set(par)


# -------------------- Layers classes --------------------------------------


class Layer():
    """Layer type."""

    # Global parametrs
    is_last = True  # if layer is last needs to hide d
    is_set = False  # hide or show layer of this type, when type swithces

    def __init__(self, parent_frame):
        """Init for class.

        Parameters
        ----------
        parent_frame : Frame
            External frame to contain layer.
        my_master : Slass: layers container
            Link to god to prey for reincarnation or death.
        """

        self.parent_frame = parent_frame
        self.my_frame = Frame(parent_frame)  # frame to hide and show layer


        # set main parametrs
        self.set_type()
        self.d_par = Parametr(self.my_frame, ' d', 1, 1000, 0.1,
                          dim_box={"nm":1e-9, "µm":1e-6})
        if self.is_last:
            self.d_par.frm_x.pack_forget()



    def set_type(self):
        """ Null proc to redefine in heir class."""
        return

    def turn_on(self):
        """Show thus layer type when swithced."""
        if not self.is_set:
            self.is_set = True
            self.my_frame.pack(fill=BOTH, expand=True)

    def turn_off(self):
        """Hide thus layer type when swithced."""
        if  self.is_set:
            self.is_set = False
            self.my_frame.pack_forget()

    def set_last(self):
        """ Hide d if layer is last (semi infinite layer)."""
        if not self.is_last:
            self.is_last = True
            self.d_par.frm_x.pack_forget()

    def unset_last(self):
        """Show d if layer is not last (thin layer)."""
        if self.is_last:
            self.is_last = False
            self.d_par.frm_x.pack(fill=X)


class Layer_m(Layer):
    """Metall layer."""

    my_type = "Metal" 
    def set_type(self):
        """Overloading - set metall parametrs."""

        self.n_par = Parametr(self.my_frame, ' n', 0.01, 10, 0.01,
                             dim_box={"x1":1, "x10":10, "x100":100})
        self.k_par = Parametr(self.my_frame, ' k', 0.01, 10, 0.01,
                             dim_box={"x1":1, "x10":10, "x100":100})

    def get_parametrs(self):
        """Take actual layer parametrs.

        Returns
        -------
        complex
            SPPPy.Layer metall.
        """
        return LYR(self.n_par.get() + 1j*self.k_par.get(),
                     self.d_par.get(), name=self.my_type)

    def set_parametrs(self, eps_d):
        """Set layer parametrs.

        Parameters
        ----------
        eps_d : array
            [imag_eps (as its a metal), real_d].
        """
        self.n_par.set(re(eps_d[0]))
        self.k_par.set(im(eps_d[0]))
        self.d_par.set(re(eps_d[1]))


class Layer_d(Layer):
    """Dielectric layer"""

    my_type = "Dielectric"
    def set_type(self):
        """Overloading - set dielectric parametrs."""
        self.turn_on()
        self.n_par = Parametr(self.my_frame, ' n', 1, 10, 0.01,
                            dim_box={"x1":1, "x10":10, "x100":100})

    def get_parametrs(self):
        """Take actual layer parametrs.

        Returns
        -------
        complex
            SPPPy.Layer dielectric
        """
        return LYR(self.n_par.get(), self.d_par.get(), name=self.my_type)

    def set_parametrs(self, eps_d):
        """Set layer parametrs.

        Parameters
        ----------
        eps_d : array
            [real_eps (as its a dielectric), real_d].
        """
        self.n_par.set(re(eps_d[0]))
        self.d_par.set(re(eps_d[1]))


class Layer_grad(Layer):
    """ Gradient layer"""

    my_type = "Gradient" 
    def set_type(self):
        """Overloading - set dielectric parametrs."""

        # Grad parametrs:
        self.grad_frame = Frame(self.my_frame)
        self.grad_frame.pack(fill=X)

        self.grad_func_var = StringVar()
        self.grad_func_values = ("a*x+b", "a*(x+b)**2",
                                 "a*(x+b)**3")

        # lambda functions in layers container
        self.spin_grad_func = ttk.Combobox(self.grad_frame, values=self.grad_func_values,
                                           state="readonly")
        self.spin_grad_func.current(0)
        self.spin_grad_func.bind("<<ComboboxSelected>>")
        self.spin_grad_func.pack(side=LEFT)

        self.A = Parametr(self.my_frame, 'A', step=0.1)
        self.B = Parametr(self.my_frame, 'B', step=0.1)

    def get_parametrs(self):
        """Take actual layer parametrs.

        Returns
        -------
        complex
            SPPPy.Layer gradient
        """
        vals = (self.spin_grad_func.get(), self.A.get(), self.B.get())
        return LYR(vals, self.d_par.get() , name=self.my_type)

    def set_parametrs(self, arr):
        """Set layer parametrs.

        Parameters
        ----------
        arr : array
            [function, A, B, d].
        """
        self.spin_grad_func.current(self.grad_func_values.index(arr[0]))
        self.A.set_par(float(arr[1]))
        self.B.set_par(float(arr[2]))
        self.d_par.set_par(float(arr[3]))


class Layer_a(Layer):
    """Metall layer."""

    my_type = "Anisotropic" 
    def set_type(self):
        """Overloading - set metall parametrs."""

        self.n0_par = Parametr(self.my_frame, 'n0', 1, 10, 0.01,
                             dim_box={"x1":1, "x10":10, "x100":100})
        self.n1_par = Parametr(self.my_frame, 'n1', 1, 10, 0.01,
                             dim_box={"x1":1, "x10":10, "x100":100})
        self.theta_par = Parametr(self.my_frame, ' ϴ', 0, 90, 0.01)

    def get_parametrs(self):
        """Take actual layer parametrs.

        Returns
        -------
        complex
            SPPPy.Layer anisotropic
        """
        return LYR(Anisotropic(self.n0_par.get(), self.n1_par.get(),
                self.theta_par.get()), self.d_par.get() , name=self.my_type)

    def set_parametrs(self, eps_d):
        """Set layer parametrs.

        Parameters
        ----------
        eps_d : array
            [imag_eps (as its a metal), real_d].
        """
        self.n0_par.set(eps_d[0])
        self.n1_par.set(eps_d[1])
        self.theta_par.set(eps_d[2])


class Layer_disp(Layer):
    """Metall layer."""

    my_type = "Dispersion"
    def change_material(self, *args):
        """Switch material."""
        self.my_material = MaterialDispersion(self.actual_material.get())
        self.lbl_1['text'] = self.my_material.min_lam
        self.lbl_2['text'] = self.my_material.max_lam

    def set_type(self):
        """Overloading - set metall parametrs."""
        # Materials list
        Refraction_data = pd.read_csv("MetPermittivities.csv", sep=',')
        self.materials_list =  Refraction_data['Element'].drop_duplicates().tolist()

        # Combobox with materials
        self.frm_y = Frame(self.my_frame, width=20, height=70)
        self.frm_y.pack(fill=X)
        self.lbl_ = Label(self.frm_y, text=" Material: ", font=("Calibri", 9))
        self.lbl_.pack(side=LEFT)

        self.actual_material = StringVar()
        self.material_box = ttk.Combobox(self.frm_y, values=self.materials_list,
               state="readonly", textvariable=self.actual_material)
        self.material_box.current(0)
        self.material_box.bind("<<ComboboxSelected>>", self.change_material)
        self.material_box.pack(padx="4")

        # limits information
        self.frm_x = Frame(self.my_frame, width=20, height=500)
        self.frm_x.pack(fill=X)
        self.lbl_0 = Label(self.frm_x, text="Defined for λ in [", font=("Calibri", 9))
        self.lbl_1 = Label(self.frm_x, text="0", font=("Calibri", 9))
        self.lbl_00 = Label(self.frm_x, text=",", font=("Calibri", 9))
        self.lbl_2 = Label(self.frm_x, text="0", font=("Calibri", 9))
        self.lbl_000 = Label(self.frm_x, text="] µm", font=("Calibri", 9))
        self.lbl_0.pack(side=LEFT)
        self.lbl_1.pack(side=LEFT)
        self.lbl_00.pack(side=LEFT)
        self.lbl_2.pack(side=LEFT)
        self.lbl_000.pack(side=LEFT)

        # initialize
        self.material_box.set(self.materials_list[0])
        self.change_material()

    def get_parametrs(self):
        """Take actual layer parametrs.

        Returns
        -------
        complex
            SPPPy.Layer dispersion
        """
        return LYR(self.my_material, self.d_par.get() , name=self.my_type)

    def set_parametrs(self, eps_d):
        """Set layer parametrs.

        Parameters
        ----------
        eps_d : array
            [imag_eps (as its a metal), real_d].
        """
        self.n0_par.set(eps_d[0])
        self.n1_par.set(eps_d[1])
        self.theta_par.set(eps_d[2])


class Layers_container:
    """Container of all layers.

    Fit to canvas of scrollable frame
    """

    class One_layer:
        """Class for one layer, contains all types"""
        
        def __init__(self, parent_frame, close_action):
            """Create new layer

            Parameters
            ----------
            parent_frame : frame
                external frame fo fit.
            close_action : lambda
                close external frame and other.
            """
            self.parent_frame = parent_frame
            self.my_frame = Frame(self.parent_frame, width=20, height=500, relief=RAISED, bd=2)
            self.my_frame.pack(fill=X)

            self.control_frame = Frame(self.my_frame)
            self.control_frame.pack(fill=X)
            self.del_layer = Button(self.control_frame, text="x",
                                    font=("Times new roman", 9),
                                    command=close_action)
            self.del_layer.pack(side=RIGHT)

            # ALL TYPES OF MATERIALS Layer title = change type
            OptionList = ["Dielectric", "Metal", "Gradient", "Anisotropic", "Dispersion" ] 
            self.my_type_str = StringVar()
            self.my_type_str.set(OptionList[0])
            my_type_option = OptionMenu(self.control_frame, self.my_type_str,
                                    *OptionList, command=self.switch_layer)
            # opt.config(width=2, font=('Helvetica', 8))
            my_type_option.pack(fill=X)
    
            self.my_sublayers = (Layer_d(self.my_frame),
                                 Layer_m(self.my_frame),
                                 Layer_grad(self.my_frame),
                                 Layer_a(self.my_frame),
                                 Layer_disp(self.my_frame))

        def switch_layer(self, *args):
            """Change layer type"""
            for i in self.my_sublayers:
                if i.is_set : i.turn_off()
                if i.my_type == self.my_type_str.get():
                    i.turn_on()

        def get_parametrs(self):
            """Get parametrs for active layer"""
            for i in self.my_sublayers:
                if i.is_set:
                    return i.get_parametrs()
            return None

        def set_last(self):
            """Turn off d par"""
            for i in self.my_sublayers:
                i.set_last()

        def unset_last(self):
            """Turn on d par"""
            for i in self.my_sublayers:
                i.unset_last()            

    # Layer container body
    my_layers = []

    def __init__(self, parent_frame):
        """Parameters.

        ----------
        parent_frame : Frame
            External frame, canvas of scrollable frame.
        """
        self.Apar = 1
        self.Bpar = 1
        self.grad_profiles = {
            "a*x+b":        lambda x: (self.Apar/2 - 2.5) * x + 1.5 + self.Bpar/2,
            "a*(x+b)**2":   lambda x: self.Apar* 2 * (x-0.5)**2 + self.Bpar/2 + 0.5,
            "a*(x+b)**3":   lambda x: self.Apar* 2 * (x-0.5)**3 + self.Bpar/2}
        self.parent_frame = parent_frame

    def add_layer(self):
        """Add frame and all types of layer."""
        def delete_last(*args):
            if len(self.my_layers)>1:
                self.my_layers[-1].my_frame.destroy()
                self.my_layers.pop()
                self.my_layers[-1].set_last()
            else: messagebox.showerror("Error", "System must have at least two layers!")

        self.my_layers.append(self.One_layer(self.parent_frame, delete_last))
        if len(self.my_layers)>1 :self.my_layers[-2].unset_last()

    def get_system_parametrs(self):
        """Take dielectric permittivityes and thicknesses of layers.

        Returns
        -------
        eps_out : [float]
            dielectric permittivityes.
        d_out : [float]
            thicknesses of layers, 0 for first and last as semi infinite.
        """
        #First layer
        Layers_set = {0: LYR(prysm_n.get(), 0)}

        for i in range(0, len(self.my_layers)):
                a = self.my_layers[i].get_parametrs()
                if a.name != "Gradient":
                    Layers_set[i+1] = a
                else:
                    self.Apar = a.n[1]
                    self.Bpar = a.n[2]
                    nnn = self.grad_profiles[a.n[0]]
                    Layers_set[i+1] = LYR(nnn, a.thickness, name="Gradient")
        print(Layers_set)
        return Layers_set

    def set_system_parametrs(self, Layers_set):
        """Set parametrs of homogeneous layers (not for prysm).

        Parameters
        ----------
        Layers_set : [complex, float]
            array of eps and d for each layer.
        """
        # clean all layers
        while len(self.n_layer) > 0:
            self.n_layer[0][0].parent_frame.destroy()
            self.n_layer.pop(0)
            self.n_frames.pop(0)
        
        # add new layers and fill parametrs
        for i in range(0, Layers_set.shape[0]):
            self.add_layer()
            if im(Layers_set[i,0]) != 0: # metal
                self.n_layer[i][1].set_parametrs([Layers_set[i,0], re(Layers_set[i,1])])
                self.switch(self.n_layer[i][0], 'm')
            elif abs(Layers_set[i,0]) != 0:
                self.n_layer[i][0].set_parametrs([re(Layers_set[i,0]), re(Layers_set[i,1])])
                self.switch(self.n_layer[i][0], 'd')
            else: self.switch(self.n_layer[i][0], 'g')


# -------------------- Drawing tabs classes ---------------------------------


class DrawingType_tab:
    """Common class for all types of drawings.

    Draw in tabs
    """

    def __init__(self, parent_frame, sync_lambda=(None, None, None),
                 sync_theta_range=(None, None),
                 sync_theta=(None, None, None),
                 sync_lambda_range=(None, None)):
        """Init common logic frames.

        Parameters
        ----------
        parent_frame : Frame
            External frame.
        """
        # 4 r(thea)
        self.sync_lambda = sync_lambda
        self.sync_theta_range = sync_theta_range
        # 4 r(lambda) 
        self.sync_theta = sync_theta
        self.sync_lambda_range = sync_lambda_range
    
        self.parent_frame = parent_frame
        # self.parent_tab = parent_tab
        self.frame_draw_bt = Frame(parent_frame, bd=5)
        self.frame_draw_bt.pack(side=BOTTOM, fill=X)
        self.draw_my_parametr()
        self.draw_btn = Button(self.frame_draw_bt,
                               text="= > Draw dispersion curve < =",
                               font=("Times new roman", 10),
                               command=lambda: self.build_curve())
        self.draw_btn.pack(side=LEFT)
        self.chk_state = BooleanVar()
        self.chk_state.set(False)
        self.chk = Checkbutton(self.frame_draw_bt,
            text='Autodraw with parametrs change', var=self.chk_state)
        self.chk.pack(side=RIGHT)

        # Draw window
        self.canvas_frame = Frame(self.parent_frame, relief=SUNKEN, bd=3)
        self.canvas_frame.pack(fill=BOTH, expand=True)
        self.canvas_frame_int = Frame(self.canvas_frame, bd=2)
        self.canvas_frame_int.pack(fill=BOTH, expand=True)

        self.reflection_curve = plt.Figure()
        self.canvas = FigureCanvasTkAgg(self.reflection_curve,
                                        self.canvas_frame_int)
        self.canvas.get_tk_widget().place(relx=-0.02, rely=-0.1,
                                          relwidth=1.1, relheight=1.15)
        self.R_plot = self.reflection_curve.add_subplot(1, 1, 1)
        self.mark_axex()

    def plot_curve(self, x, y):
        """Procedure to draw on canvas.

        Parameters
        ----------
        x : [float]
            Abscissas array.
        y : [float]
            Ordinates array.
        """
        self.R_plot.clear()
        self.mark_axex()
        self.R_plot.plot(x, y)
        self.canvas.draw()

    def draw_my_parametr(self):
        """Proc to reload."""
        return

    def mark_axex(self):
        """Proc to reload."""
        return

    def build_curve(self):
        """Proc to reload."""
        return

    def get_parametr(self):
        """Get tab main parametr.

        Returns
        -------
        float
            wavelength un unit
        """
        return self.lam_par.get()


class Grad_profile_tab(DrawingType_tab):
    """Class for gradient tab. Shows gradient profile"""

    def mark_axex(self):
        """Mark plot axes."""
        self.R_plot.set_xlabel('d %')
        self.R_plot.set_ylabel('n')

    def build_curve(self):
        """Build gradient profile."""
        self.R_plot.clear()
        self.mark_axex()

        A = LayersContainer.get_system_parametrs()
        n_range = np.linspace(0, 1, 100)

        for key, value in A.items():
            if value.name == "Gradient":
                n_prof = [value.n(i) for i in n_range]
                self.R_plot.plot(n_range, n_prof, label = key)

        self.canvas.draw()


class Theta_tab(DrawingType_tab):
    """R from theta logic frames."""

    min_theta = 0
    max_theta = 90

    def draw_my_parametr(self):
        """Re R(theta) parametrs: lambda."""
        self.top_frame = Frame(self.parent_frame, width=20, height=500)
        self.top_frame.pack(fill=X)

        self.frm_lam = Frame(self.top_frame, width=20, height=500)
        self.frm_lam.pack(side=LEFT)
        self.lam_par = Parametr(self.frm_lam, 'λ', 1, 1000, 0.1,
                 sync_pars=self.sync_lambda, dim_box={"nm":1e-9, "µm":1e-6})

        self.txt1 = Label(self.top_frame, text="   ϴ ∈ [", font=("Calibri", 9))
        self.txt1.pack(side=LEFT)

        self.sync_theta_range[0].trace_add('write', self.min_theta_change)
        self.theta_min = Entry(self.top_frame, width=4,
                               textvariable=self.sync_theta_range[0])
        self.theta_min.insert(0, 0)
        self.theta_min.pack(side=LEFT)

        self.txt2 = Label(self.top_frame, text=",", font=("Calibri", 9))
        self.txt2.pack(side=LEFT)

        self.sync_theta_range[1].trace_add('write', self.max_theta_change)
        self.theta_max = Entry(self.top_frame, width=4,
                               textvariable=self.sync_theta_range[1])
        self.theta_max.insert(0, 90)
        self.theta_max.pack(side=LEFT)


        self.txt2 = Label(self.top_frame, text="]", font=("Calibri", 9))
        self.txt2.pack(side=LEFT)

    def min_theta_change(self, *args):
        """Tests for min theta."""
        try:
            self.min_theta = float(self.sync_theta_range[0].get())
            if self.min_theta < 0:
                self.min_theta = 0
                self.sync_theta_range[0].set(0)
            if self.min_theta > 90:
                self.min_theta = 90
                self.sync_theta_range[0].set(90)
        except ValueError:
            self.sync_theta_range[0].set(self.min_theta)
        return

    def max_theta_change(self, *args):
        """Tests for max theta."""
        try:
            self.max_theta = float(self.sync_theta_range[1].get())
            if self.max_theta < 0:
                self.max_theta = 0
                self.sync_theta_range[1].set(0)
            if self.max_theta > 90:
                self.max_theta = 90
                self.sync_theta_range[1].set(90)
        except ValueError:
            self.sync_theta_range[1].set(self.max_theta)
        return

    def mark_axex(self):
        """Mark plot axes."""
        self.R_plot.set_xlabel('ϴ')
        self.R_plot.set_ylabel('R')

    def build_curve(self):
        """Calculate Real R from theta curve."""
        return
    
    def get_parametr(self):
        """Get parametrs value."""
        return self.lam_par.get()

    def set_parametr(self, par):
        """Set parametrs value.

        Parameters
        ----------
        par : float
            lambda value
        """
        self.lam_par.set(par)


class Re_theta_tab(Theta_tab):
    """Real R from theta logic frames."""

    def build_curve(self):
        """Calculate Real R from theta curve."""
        Unit = ExperimentSPR()
        Unit.steps = 50
        Unit.wavelength = self.get_parametr()
        Unit.layers = LayersContainer.get_system_parametrs()
        Unit.show_info()
        theta_range = np.linspace(self.min_theta, self.max_theta, 500)
        self.plot_curve(theta_range, Unit.R_theta_re(theta_range))
    

class Im_theta_tab(Theta_tab):
    """Complex r from theta logic frames."""

    def build_curve(self):
        """Calculate Imaginary r from theta curve."""
        Unit = ExperimentSPR()
        Unit.steps = 30
        Unit.wavelength = self.get_parametr()
        
        Unit.layers = LayersContainer.get_system_parametrs()
        Unit.show_info()
        theta_range = np.linspace(0, 90, 1000)

        b =  Unit.R_theta_cmpl(theta_range)
        x = []
        y = []
        for i in range(0, len(b)):
            x.append(re(b[i]))
            y.append(im(b[i]))
        self.plot_curve(x, y)


class Lambda_tab(DrawingType_tab):
    """R from lambda logic frames."""

    min_lambda = 0.001
    max_lambda = 100

    def draw_my_parametr(self):
        """Re R(theta) parametrs: lambda."""
        self.top_frame = Frame(self.parent_frame, width=20, height=500)
        self.top_frame.pack(fill=X)

        self.frm_theta = Frame(self.top_frame, width=20, height=500)
        self.frm_theta.pack(side=LEFT)
        self.theta_par = Parametr(self.frm_theta, 'ϴ', 0, 90, 0.1,
                 sync_pars=self.sync_theta)

        self.txt1 = Label(self.top_frame, text="   λ ∈ [", font=("Calibri", 9))
        self.txt1.pack(side=LEFT)

        self.sync_lambda_range[0].trace_add('write', self.min_lambda_change)
        self.lambda_min = Entry(self.top_frame, width=5,
                               textvariable=self.sync_lambda_range[0])
        self.lambda_min.insert(0, 0.001)
        self.lambda_min.pack(side=LEFT)

        self.txt2 = Label(self.top_frame, text=",", font=("Calibri", 9))
        self.txt2.pack(side=LEFT)

        self.sync_lambda_range[1].trace_add('write', self.max_lambda_change)
        self.lambda_max = Entry(self.top_frame, width=5,
                               textvariable=self.sync_lambda_range[1])
        self.lambda_max.insert(0, 100)
        self.lambda_max.pack(side=LEFT)


        self.txt2 = Label(self.top_frame, text="] µm", font=("Calibri", 9))
        self.txt2.pack(side=LEFT)

    def min_lambda_change(self, *args):
        """Tests for min lambda."""
        try:
            self.min_lambda = float(self.sync_lambda_range[0].get())
            if self.min_lambda < 0.001:
                self.min_lambda = 0.001
                self.sync_lambda_range[0].set(0.001)
            if self.min_lambda > 100:
                self.min_lambda = 100
                self.sync_lambda_range[0].set(100)
        except ValueError:
            self.sync_lambda_range[0].set(self.min_lambda)
        return

    def max_lambda_change(self, *args):
        """Tests for max lambda."""
        try:
            self.max_lambda = float(self.sync_lambda_range[1].get())
            if self.max_lambda < 0.001:
                self.max_lambda = 0.001
                self.sync_lambda_range[1].set(0.001)
            if self.max_lambda > 100:
                self.max_lambda = 100
                self.sync_lambda_range[1].set(100)
        except ValueError:
            self.sync_lambda_range[1].set(self.max_lambda)
        return

    def mark_axex(self):
        """Mark plot axes."""
        self.R_plot.set_xlabel('λ, µm')
        self.R_plot.set_ylabel('R')

    def build_curve(self):
        """Calculate Real R from theta curve."""
        return

    def get_parametr(self):
        """Get parametrs value."""
        return self.theta_par.get()

    def set_parametr(self, par):
        """Set parametrs value.

        Parameters
        ----------
        par : float
            lambda value
        """
        self.theta_par.set(par)


class Re_lambda_tab(Lambda_tab):
    """Real R from lambda logic frames."""

    def build_curve(self):
        """Calculate Real R from theta curve."""
        Unit = ExperimentSPR()
        Unit.steps = 30
        Unit.wavelength = self.get_parametr()
        Unit.layers = LayersContainer.get_system_parametrs()
        lambda_range = np.linspace(self.min_lambda*1e-6, self.max_lambda*1e-6, 500)
        self.plot_curve(np.array(lambda_range)*1e6, Unit.R_lambda_re(self.get_parametr(), lambda_range))


class Im_lambda_tab(Lambda_tab):
    """Complex R from theta logic frames."""

    def build_curve(self):
        """Calculate Real R from theta curve."""
        Unit = ExperimentSPR()
        Unit.steps = 30
        Unit.wavelength = self.get_parametr()
        
        Unit.layers = LayersContainer.get_system_parametrs()
        
        lambda_range = np.linspace(self.min_lambda*1e-6, self.max_lambda*1e-6, 500)

        b = Unit.R_lambda_cmpl(self.get_parametr(), lambda_range)
        x = []
        y = []
        for i in range(0, len(b)):
            x.append(re(b[i]))
            y.append(im(b[i]))
        self.plot_curve(x, y)


# -------------------- Functions -------------------------------------------


def save_file():
    """
    Save file proc.

    Returns none.
    """
    print('save file')
    # save layers
    Layers.save_grad()
    a = LayersContainer.get_system_parametrs()

    MyFile = pd.DataFrame(a)
    MyFile = MyFile.T
    layer_count = len(a[0])
    MyFile['type'] = ['layer']*layer_count

    # Get tabs parametrs
    MyFile.loc[layer_count + 0] = [experiment_lambda_scale.get(), 0, 'lambda']
    MyFile.loc[layer_count + 1] = [experiment_theta_scale.get(), 0, 'theta']


    # Fill file
    MyFile = MyFile.set_index('type')
    file_name = filedialog.asksaveasfilename(filetypes=[("csv files", "*.csv")])
    if file_name != '':
        MyFile.to_csv(file_name + '.csv', sep=' ')
        print(MyFile)


def open_file():
    """
    Open file proc.

    Returns none.
    """
    print('open file')

    realy = messagebox.askquestion("Open file",
                "Actual set will be erased, are You Sure?", icon='warning')

    if realy == 'yes':
        # Recieving parametrs
        file_name = filedialog.askopenfilename(filetypes=[("csv files", "*.csv")])
        if file_name != '':
            data = pd.read_csv(file_name, sep=' ')

            my_layers = data.values[data['type'] == 'layer'][:, 1:3].astype(complex)

            my_lam = data.values[data['type'] == 'lambda'][:, 1].astype(complex)
            my_thet = data.values[data['type'] == 'theta'][:, 1].astype(complex)

            grad_par = data.values[data['type'] == 'grad'][:, 1:]



            # Set recieved parametrs
            prysm_n.set(re(my_layers[0, 0])) # Prism not in layers class


            # set lambda
            R_tabs[0].set(re(my_lam[0]))
            R_tabs[1].set(re(my_lam[0]))
            # set theta
            R_tabs[2].set(re(my_thet[0]))
            R_tabs[3].set(re(my_thet[0]))

            for i in range(0, len(R_tabs)):
                R_tabs[i].build_curve()


# --------------------------------------------------------------------------
# -------------------- Window forms ----------------------------------------
# --------------------------------------------------------------------------


window['bg'] = '#fafafa'
window.title('SPP reflection viewer')
window.geometry('810x510')
window.resizable(width=False, height=False)

# Frame for ploting
frame_left = Frame(window, relief=RAISED , bd=5)
frame_left.place(x=5, y=5, width=500, height=500)

# Frame fo parametrs
frame_right = Frame(window, relief=RAISED, bd=5)
frame_right.place(x=505, y=5, width=300, height=500)


# -------------------- Left frame - Drawing --------------------------------

# Bottom frame - draw and autodraw
# Create notebook tabs
DrawTabs = ttk.Notebook(frame_left)
DrawTabs.pack(fill=BOTH, expand=True)

R_theta_real = Text(frame_left)
R_theta_imag = Text(frame_left)
R_lambd_real = Text(frame_left)
R_lambd_imag = Text(frame_left)
Grad_profile = Text(frame_left)

DrawTabs.add(R_theta_real, text='    Real R(ϴ)   ') # 0
DrawTabs.add(R_theta_imag, text=' Imaginary R(ϴ) ') # 1
DrawTabs.add(R_lambd_real, text='    Real R(λ)   ') # 2
DrawTabs.add(R_lambd_imag, text=' Imaginary R(λ) ') # 3
DrawTabs.add(Grad_profile, text=' Gradient profile ') # 4

R_tabs = []

# -------------------- Right frame - Parametrs -----------------------------


prysm_n = Parametr(frame_right, ' Prism n', 1, 10, 0.01,
                    dim_box={"x1":1, "x10":10, "x100":100})

# Middle - Layers container
frame_labels = VerticalScrolledFrame(frame_right, relief=SUNKEN, bd=3)
LayersContainer = Layers_container(frame_labels.interior)

# Bottom - Control panel
frame_system_control = Frame(frame_right, bd = 5)
frame_system_control.pack(side=BOTTOM, fill=X)

add_btn = Button(frame_system_control, text="Add layer",
            font=("Times new roman", 9), command=LayersContainer.add_layer)
add_btn.pack(side=LEFT)

save_btn = Button(frame_system_control, text="Save set",
            font=("Times new roman", 9), command=save_file)
save_btn.pack(side=RIGHT)

open_btn = Button(frame_system_control, text="Open set",
            font=("Times new roman", 9), command=open_file)
open_btn.pack(side=RIGHT)

frame_labels.pack(fill=BOTH, expand=True)

# Init first layer and set it to metal
LayersContainer.add_layer()
LayersContainer.my_layers[-1].my_type_str.set('Metal')
LayersContainer.my_layers[-1].switch_layer()

# Create class object in corresponding tab in left frame
lambda_parametr_sync = (DoubleVar(), StringVar(), StringVar())
theta_range_sync = (StringVar(), StringVar())

theta_parametr_sync = (DoubleVar(), StringVar())
lambda_range_sync = (StringVar(), StringVar())

R_tabs.append(Re_theta_tab(R_theta_real, sync_lambda=lambda_parametr_sync, 
                           sync_theta_range=theta_range_sync))
R_tabs.append(Im_theta_tab(R_theta_imag, sync_lambda=lambda_parametr_sync, 
                           sync_theta_range=theta_range_sync))

R_tabs.append(Re_lambda_tab(R_lambd_real, sync_theta=theta_parametr_sync,
                            sync_lambda_range=lambda_range_sync))
R_tabs.append(Im_lambda_tab(R_lambd_imag, sync_theta=theta_parametr_sync,
                            sync_lambda_range=lambda_range_sync))

R_tabs.append(Grad_profile_tab(Grad_profile))

# End program
window.mainloop()