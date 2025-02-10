from picawe import SystemModel
import numpy as np
from picawe.defaults import PLOT_PARAMETERS, PLOT_LABELS
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
from typing import Collection

class TimeSeries:
    """

    Attributes:
        states (list): Time series of `SteadyState` objects.
        kite_model (SystemModel): System model used to generate the time series.

    """


    def __init__(self,
                 kite_model: SystemModel,
                 ):

        # System configuration.
        self.kite_model = kite_model

        # Time series states.
        self.states = []
    
    
    # @property
    # def converged(self):
    #     return np.all(np.array([s.converged for s in self.states])) if len(self.states) > 0 else False

    # @property
    # def duration(self):
    #     if self.states:
    #         return self.states[-1].time - self.states[0].time
    #     else:
    #         return np.nan

    # @property
    # def energy(self):
    #     # Return the energy due to ground power at each state.
    #     if self.states:
    #         arr = np.array([[state.time, state.power_ground] for state in self.states])
    #         time = arr[:, 0]
    #         power = arr[:, 1]
    #         return np.trapz(power, x=time)
    #     else:
    #         return np.nan

    # @property
    # def average_power(self):
    #     return self.energy / self.duration

    # @property
    # def average_factor_reeling(self):
    #     return np.trapz(np.array([(s.factor_reeling, s.time) for s in self.states])) / self.duration

    # @property
    # def average_tension_ground(self):
    #     tensions = np.array([s.tension_ground for s in self.states])
    #     times = np.array([s.time for s in self.states])
    #     integrated_tension = np.trapz(tensions, times)
    #     return integrated_tension / self.duration
    
    def return_variable(self, variable: str):
        var_func = self.kite_model.extract_function(variable)
        var = []
        for state in self.states:
            input_dict = {name: state[name] for name in var_func.name_in()}
            output = var_func(**input_dict)[variable]
            var.append(float(output))

        return np.array(var)

    def plot_trace_on_plane(
            self, plot_markers=True, plot_kwargs=None, ax=None,
            gradient_color: tuple = None, plane=('x', 'z')
    ):
        """Plot of the downwind versus vertical position of the kite.

        Parameters:
            :param plot_kwargs: Line plot keyword arguments.
            :param plot_markers: Use the steady state results to mark non-converged points and points where control or
                path limits are violated.
            :param gradient_color: tuple of (attribute, colormap) to shade the trajectory, e.g. ('speed', 'coolwarm')
            :param plane: tuple of state attributes that define the plane, e.g. ('x', 'z')

        """
        if plot_kwargs is None:
            plot_kwargs = {}

        norm = plot_kwargs.get('norm', None)
        linewidth = plot_kwargs.get('linewidth', 2)
        figsize = plot_kwargs.get('figsize', (None, None))
        cbar = plot_kwargs.get('cbar', True)
        legend = plot_kwargs.get('legend', False)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        x = self.return_variable(plane[0])
        y = self.return_variable(plane[1])
        speed = self.return_variable('speed_tangential')

        if gradient_color is None:
            # ax.scatter(x_traj, z_traj, **plot_kwargs)  # Scatter plots the dots at each time step.
            ax.plot(x, y)
        else:

            vals = np.array([s.__getattribute__(gradient_color[0]) for s in self.states])

            if norm is None:
                norm = Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
            cmap = plt.get_cmap(gradient_color[1])

            # Matplotlib has no feature to plot colormap over line..
            # So we plot each line segment individually and assign a color
            points = np.array([x, y]).T
            fc = cmap(norm((vals[:-1] + vals[1:]) / 2))
            for i, segment in enumerate(zip(points[:-1], points[1:])):
                ax.plot(*np.array(segment).T, c=fc[i], linewidth=linewidth)

            # Create colorbar
            m = cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array(np.linspace(norm.vmin, norm.vmax, 30))

            if cbar:
                cb = fig.colorbar(m, aspect=15, label=PLOT_LABELS.get(gradient_color[0], gradient_color[0]), ax=ax)
            # cb.set_ticks(np.linspace(norm.vmin, norm.vmax, 8))

        # if plot_markers:
        #     # Plot all points for which the steady state did not converge.
        #     for state in self.states:
        #         if not state.converged:
        #             ax.plot(
        #                 getattr(state, plane[0]),
        #                 getattr(state, plane[1]),
        #                 'kx', label='not converged'
        #             )

        #         # Plot all points for which control limits are violated
        #         for violation in state.assess_limit_violations(self.control_limits):
        #             ax.plot(
        #                 getattr(violation, plane[0]),
        #                 getattr(violation, plane[1]),
        #                 'ro',
        #                 label=violation.summary
        #             )

        #         # Plot all points for which path limits are violated
        #         for violation in state.assess_limit_violations(self.path_limits):
        #             ax.plot(
        #                 getattr(violation.state, plane[0]),
        #                 getattr(violation.state, plane[1]),
        #                 'go',
        #                 label=violation.summary
        #             )

        ax.set_xlabel(PLOT_LABELS.get(plane[0], plane[0]))
        ax.set_ylabel(PLOT_LABELS.get(plane[1], plane[1]))

        # if ax.get_xlim()[0] > 0.:
        #     plt.xlim([0., None])
        # plt.ylim([0., None])
        ax.grid(True)
        ax.set_aspect('equal')

        if legend:
            ax.legend(loc=legend if type(legend) is str else None)

        return fig, ax

    def trajectory_plot3d(self, fig=None, ax=None, animate=False, animate_kwargs=None, plot_markers=None, plot_kwargs=None, gradient_color=None):
        """Animation of the 3D trajectory of the kite.

        Args:
            fig_num (int, optional): Number of figure used for the plot, if None a new figure is created.
            animate (bool, optional): Make animation of the plot by changing the view angle.
            plot_kwargs (dict, optional): Line plot keyword arguments.
            plot_point_type (int, optional): If not None, only plot points for which the phase identifier corresponds to
                the given integer.
            gradient_color: tuple of attribute + colormap name

        """
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.colors import ListedColormap

        if ax is None:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
        else:
            fig = ax.figure

        if plot_markers is None:
            plot_markers = []
        if plot_kwargs is None:
            plot_kwargs = {}
        if animate_kwargs is None:
            animate_kwargs = {}

        label = plot_kwargs.get('label', None)
        marker_label = plot_kwargs.get('marker_label', None)
        marker_color = plot_kwargs.get('marker_color', None)
        norm = plot_kwargs.get('norm', None)
        plot_ground_station = plot_kwargs.get('ground_station', True)
        color = plot_kwargs.get('color', None)
        legend = plot_kwargs.get('legend', False)

        t = self.return_variable('t')
        x = self.return_variable('x')
        y = self.return_variable('y')
        z = self.return_variable('z')
        speed = self.return_variable('speed_tangential')

        t = t[~np.isnan(t)]
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        z = z[~np.isnan(z)]
        speed = speed[~np.isnan(speed)]

        t = t.round(6)  # required to get rid of numeric error

        if gradient_color is not None:
            vals = np.array([s.__getattribute__(gradient_color[0]) for s in self.states])

            if 'angle' in gradient_color[0]:
                vals = np.degrees(vals)

            norm = Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals)) if norm is None else norm
            cmap = plt.get_cmap(gradient_color[1])

            # Matplotlib has no feature to plot colormap over line..
            # So we plot each line segment individually and assign a color
            points = np.array([x, y, z]).T
            fc = cmap(norm((vals[:-1] + vals[1:]) / 2))
            for i, segment in enumerate(zip(points[:-1], points[1:])):
                start, end = segment
                if np.linalg.norm(np.array(end) - np.array(start)) < 10:  # Don't plot big discontinuities. TODO remove?
                    ax.plot(*np.array(segment).T, c=fc[i])

            # Create colorbar
            m = cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array(vals)

            if plot_kwargs.get('colorbar', True):
                fig.colorbar(m, shrink=0.5, aspect=10, label=PLOT_LABELS.get(gradient_color[0], gradient_color[0]), ax=ax)

        else:
            ax.plot(x, y, z, label=label, color=color)

        # Plot the markers if given
        if plot_markers:
            ax.plot(
                x[np.isin(t, plot_markers)],
                y[np.isin(t, plot_markers)],
                z[np.isin(t, plot_markers)],
                's', markerfacecolor='None', label=marker_label, color=marker_color
            )

        if legend:
            ax.legend()

        if plot_ground_station:
            ax.plot(0, 0, 0, marker='o', color='tab:brown')  # plot ground station

        if plot_kwargs.get('labels', True):
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')

        if not plot_kwargs.get('ticks', True):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        plt.grid(True)

        x_min, x_max = ax.get_xlim()
        ax.set_ylim(-0.5*(x_max-x_min), 0.5*(x_max-x_min))  # set y lim as the same of x, but centered
        ax.set_zlim(ax.get_zlim()[0], max(ax.get_zlim()[1], 100))  # z lim minimum height 100 m seems reasonable
        ax.set_aspect('equal')  # Set equal aspect ratio of ax, else the path looks distorted

        if animate:
            # Rotate the axes and update plot.
            def init():
                ax.view_init(animate_kwargs.get('elevation_angle', 30), 0)
                return [fig]
            def animate(i):
                ax.view_init(animate_kwargs.get('elevation_angle', 30), i)
                return [fig]

            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=720, interval=2, blit=True)
            writervideo = animation.FFMpegWriter(fps=30)
            anim.save('trajectory_plot.mp4', writer=writervideo)

        ax.view_init(30, 50)

        return fig, ax

    def plot_traces(self, y_params: Collection, x_param: str = 't', y_labels: dict=None, y_scaling=None, plot_markers=None, fig_num=None, axes=None, plot_kwargs: dict = None):
        """Plot the time trace of a parameter from multiple sources.

        Args:
            y_params: list of strings with y_labels to plot
            plot_markers: list of x-values at which to plot a marker
            y_labels (tuple, optional): Y-axis y_labels corresponding to `plot_parameters`.
            y_scaling (tuple, optional): Scaling factors corresponding to `plot_parameters`.
            fig_num (int, optional): Number of figure used for the plot, if None a new figure is created..

            plot_kwargs:
                legend: bool or str. If str, this will be the location at which legend is plotted.

        :return fig, axes. Axes is always a list of Axis objects, even if only one 1 y parameter is plotted.
        """

        if plot_kwargs is None:
            plot_kwargs = {}

        unwrap = plot_kwargs.get('unwrap', True)
        remove_x_labels = plot_kwargs.get('remove_x_labels', False)
        label = plot_kwargs.get('label', None)
        linestyle = plot_kwargs.get('linestyle', None)
        marker_label = plot_kwargs.get('marker_label', None)
        marker_color = plot_kwargs.get('marker_color', None)
        legend = plot_kwargs.get('legend', False)
        color = plot_kwargs.get('color', None)
        x_label = plot_kwargs.get('x_label', PLOT_LABELS.get(x_param, x_param))

        ncols = plot_kwargs.get('ncols', 1)
        nrows = int(np.ceil(len(y_params)/ncols))

        if y_labels is None:
            y_labels = {}
        if y_scaling is None:
            y_scaling = [None for _ in range(len(y_params))]
        if fig_num:
            axes = plt.figure(fig_num).get_axes()
        if axes is None:
            fig, axes = plt.subplots(nrows, ncols, sharex='all', num=fig_num)
            if len(y_params) == 1:
                axes = [axes]
        else:
            fig = axes[0].figure

        if plot_markers is None:
            plot_markers = []

        x = self.return_variable(x_param)

        for i, (p, f, ax) in enumerate(zip(y_params, y_scaling, axes)):
            try:
                y = self.return_variable(p)
            except AttributeError:
                print(f'Cannot plot the trace of attribute {p} as it does not exist. Valid attributes are: '
                      f'{self.states[0].list_traceable_attributes()}')
                continue

            # Plot angles in degrees and if required, unwrap to avoid large discontinuities
            if 'angle' in p or 'rate' in p:
                if unwrap:
                    y = np.unwrap(y)
                y = np.degrees(y)

            ax.plot(x, y, label=label, c=color, linestyle=linestyle)

            # Plot the markers if given
            if plot_markers:
                marker_vals = y[np.isin(x, plot_markers)]
                ax.plot(plot_markers, marker_vals, 's', markerfacecolor='None', label=marker_label, color=marker_color)

            # Label axes and set ticks
            try:
                y_lbl = y_labels[p] if p in y_labels.keys() else PLOT_LABELS[p]
            except KeyError:
                print(f'Label not specified for {p} and not in defaults.')
                y_lbl = p

            ax.set_ylabel(y_lbl)
            ax.ticklabel_format(useOffset=False)  # disable scientific notation offset
            # ax.set_xticks(np.arange(np.round(x[0]), x[-1], 5))  # x-tick every 5 seconds
            ax.grid(True, which='both')

        if legend:
            # If legend is given as a string, that'll be the location
            handles, labels = axes[0].get_legend_handles_labels()
            loc = legend if isinstance(legend, str) else None
            fig.legend(handles, labels, loc=loc, ncol=len(y_params))

        if remove_x_labels:
            for ax in axes[:-1]:
                ax.set_xticklabels([])

        axes[-1].set_xlabel(x_label)
        # axes[-1].set_xlim([0, None])

        return fig, axes

    def interactive_plot(
            self,
            parameters: list=None, plot_vectors: dict=None, vector_directions: list = None,
            vector_scaling: dict=None, animate=False, y_labels=None, gradient_color: tuple = None):
        """Interactive plot. To make the slider work, you must keep a reference to the figure and slider in the
        __main__ thread. I.e. you must call this method like: fig, slider = timeseries.interactive_plot()"""

        if parameters is None:
            parameters = PLOT_PARAMETERS
        if plot_vectors is None:
            plot_vectors = {}
        if y_labels is None:
            y_labels = {}

        if vector_directions is None:
            vector_directions = np.ones((3, 1))

        if vector_scaling is None:
            vector_scaling = {v: 0.05 for v in plot_vectors}

        if len(parameters) < 4:
            raise ValueError(f'Interactive plot needs at least 4 parameters to plot')

        try:
            dt = self.states[1].time - self.states[0].time
            fps = 1 / dt
        except Exception as e:
            print('Could not determine frame rate to animate, using default value.')
            fps = 24

        # Grid is such that the 3d plot spans half of the timeplots, 2d plot the other half and the slider as well.
        halfway_point = int(np.ceil(len(parameters) / 2))
        grid = (
                [['3d', p] for p in parameters[:halfway_point]] +
                [['2d', p] for p in parameters[halfway_point:-1]] +
                [['slider', parameters[-1]]]
        )

        # Create figs + axes
        fig, axs = plt.subplot_mosaic(
            grid,
            # figsize=(screen_width/100, screen_height/100),  # 100 dpi
            figsize=(15, 7),
            per_subplot_kw={'3d': {'projection': '3d'}},
            gridspec_kw={'width_ratios': [1, 2],'wspace': 0.3, 'hspace': 0.2}
        )

        # Fig size + ax size
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        # axs['3d'].set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))

        # Plot 3d plot, 2d plot, and time plots in the right axes.
        param_axs = [ax for k, ax in axs.items() if k not in ['3d', '2d', 'slider']]
        self.plot_trace_on_plane(ax=axs['2d'], gradient_color=gradient_color, plot_kwargs={'legend': True})
        self.trajectory_plot3d(ax=axs['3d'], gradient_color=gradient_color)
        self.plot_traces(
            y_params=parameters,
            axes=param_axs,
            y_labels=y_labels,
            plot_kwargs={'unwrap': True, 'remove_x_labels': True}
        )

        x = self.return_variable('x')
        y = self.return_variable('y')
        z = self.return_variable('z')
        t = self.return_variable('t')
        extract_state = {}
        for p in parameters:
            extract_state[p] = self.return_variable(p)
        # Plot time markers
        state = self.states[0]
        markers = {}
        for p in parameters:
            value = extract_state[p][0] if 'angle' not in p else np.degrees(extract_state[p][0])
            markers[p] = axs[p].plot(state["t"], value, color='tab:red', linewidth=2, marker='o')[0]

        markers['3d'] = axs['3d'].plot([x[0]],[y[0]],[z[0]], color='tab:red', marker='o', linewidth=1)[0]
        markers['2d'] = axs['2d'].plot(x[0], z[0], color='tab:red', marker='o', linewidth=1)[0]

        # Plot vectors
        vectors = {}

        # Create slider, and cache time vector
        self.__cached_time = [round(s["t"], 3) for s in self.states]

        time_slider = Slider(
            ax=axs['slider'],
            label='Time [s]',
            # valmin = self.states[0]["t"],
            # valmax=self.states[-1]["t"],
            valstep=self.__cached_time,
            valmin=self.__cached_time[0],
            valmax=self.__cached_time[-1],
            valinit=self.__cached_time[0],
        )

        # The function to be called anytime a slider's value changes
        def update(time):
            index = self.__cached_time.index(round(time, 3))

            # Update markers
            for p in parameters:
                val = extract_state[p][index] if 'angle' not in p else np.degrees(extract_state[p][index])
                markers[p].set_data([time], [val])

            markers['3d'].set_data_3d([x[index]], [y[index]], [z[index]])
            markers['2d'].set_data([x[index]], [z[index]])

            # For each vector, plot components required directions
            for v, c in plot_vectors.items():
                for i, d in enumerate(vector_directions):
                    # Vector component in Cartesian coordinates:
                    vec = np.matmul(state.transformation_C_from_W.T, getattr(state, v) * d * vector_scaling[v])
                    vec_length = np.linalg.norm(vec)
                    try:
                        vectors[v + str(i)].remove()
                    except KeyError:
                        pass
                    vectors[v + str(i)] = axs['3d'].quiver(*state.position_W, *vec, length=vec_length, color=c, arrow_length_ratio=0.2)

            fig.canvas.draw_idle()

        time_slider.on_changed(update)

        if animate:
            # Rotate the axes and update plot.
            def init():
                axs['3d'].view_init(30, -40)
                return [fig]

            def animate(i):
                axs['3d'].view_init(30, -40 + i/5)
                time_slider.set_val(self.__cached_time[i])
                return [fig]

            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=len(self.__cached_time), interval=1, blit=True)

            writervideo = animation.FFMpegWriter(fps=fps, bitrate=1e5)
            anim.save('interactive_plot.mp4', writer=writervideo)

        # mng = plt.get_current_fig_manager()
        # mng.window.state('zoomed')
        # mng.resize(screen_width, screen_height)
        return fig, time_slider