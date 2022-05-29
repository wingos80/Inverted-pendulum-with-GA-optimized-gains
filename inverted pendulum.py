from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
import timeit

def theta_dd(theta, dtheta, x_dd):
    return (g*l*np.sin(theta)+x_dd*l*np.cos(theta))/(ixx+l**2)-dtheta/100


### Genetic Alogorithm ###
fig1, subfig1 = plt.subplots(1,2)
subfig2 = subfig1[0].twinx()
fig1.suptitle('katyusha:) Algorithm')

debugger_dims = (5,7)
fig_debug, subfig_debug = plt.subplots(debugger_dims[0],debugger_dims[1], sharex=True, sharey=True)
fig_debug.suptitle('Debugger.... ')

# function used for python sorted()
def take_last(elem):
    return elem[5]

class GeneticAlgo():
    def __init__(self, max_itr, elite_cutoff, init_gains, time_list, init_conds, tgt):
        self.max_itr = max_itr
        self.elite_cutoff = elite_cutoff
        self.theta_kp = init_gains[0]
        self.theta_kd = init_gains[1]
        self.x_kp = init_gains[2]
        self.x_kd = init_gains[3]
        self.time_list = time_list

        self.theta0, self.dtheta0, self.ddtheta0 = init_conds[0], init_conds[1], init_conds[4]
        self.x0, self.dx0, self.ddx0 = init_conds[2], init_conds[3], init_conds[5]
        self.x_tgt, self.dx_tgt = tgt[0], tgt[1]

        self.best_fitness = 9999999
        self.best_gains = np.zeros(6)
        self.best_gains_history = []
        self.best_gains_index = 0

    def PopInit(self, size, mutation_rate):
        """
        Function returning an array of PD populations, VERIFIED
        :param size: size of population
        :return: np array of everyones' gains, fitness, and states history
        """
        init_pop_start_time = timeit.default_timer()
        print(f'{init_pop_start_time}: Initializing population...')
        self.size = size
        self.mutation_rate = mutation_rate

        # initialize gains
        self.gains = np.random.rand(6, self.size)
        self.gains[0, :] = (self.gains[0, :] - 0.5) * self.theta_kp + self.theta_kp
        self.gains[1, :] = (self.gains[1, :] - 0.5) * self.theta_kd + self.theta_kd
        self.gains[2, :] = (self.gains[2, :] - 0.5) * self.x_kp + self.x_kp
        self.gains[3, :] = (self.gains[3, :] - 0.5) * self.x_kd + self.x_kd
        self.gains[4, :] = np.linspace(0, self.size-1, self.size)

        # initialize fitness
        self.fitnesses = np.zeros((self.size))

        # initialize states
        self.sim_states = np.zeros((self.size, 4)) # theta, dtheta, x, dx
        self.sim_states[:, 0] = self.theta0
        self.sim_states[:, 1] = self.dtheta0
        self.sim_states[:, 2] = self.x0
        self.sim_states[:, 3] = self.dx0

        # initialize states history
        self.all_states = np.zeros((self.time_list.shape[0], self.size, 6))     # theta, dtheta, x, dx, ddtheta, ddx
        self.all_states[0, :, 0:4] = self.sim_states

        init_pop_end_time = timeit.default_timer()
        print(f'{init_pop_end_time}: Population initialized ({(init_pop_end_time- init_pop_start_time)/1000} ms)')

    # function to simulate the system
    def Simulate(self):
        """
        Function simulating the pendulum for a given time index. VERIFIED
        :return:
        """
        sim_start_time = timeit.default_timer()
        print(f'{sim_start_time}: Simulating...')
        dt = timestep
        for i in range(len(self.time_list)-1):
            err = self.sim_states-np.array([[0, 0, x_tgt, dx_tgt]])                         # finding PD errors

            ddx = (err @ self.gains[0:4,:]).diagonal()                                      # calculating ddx from gains
            self.sim_states[:,3] += ddx*dt                                                  # updating speed
            self.sim_states[:,2] += self.sim_states[:,3]*dt                                 # updating position

            ddtheta = theta_dd(self.sim_states[:,0], self.sim_states[:,1], ddx)             # calculating ddtheta from ddx
            self.sim_states[:,1] += ddtheta * dt                                            # updating angular speed
            self.sim_states[:,0] += self.sim_states[:,1] * dt                               # updating angle
            self.sim_states[:,0] = (self.sim_states[:,0] + np.pi) % (2 * np.pi) - np.pi     # normalize angle to be between -pi and pi

            # Logging sim states to all states
            self.all_states[i+1, :, 0:4] = self.sim_states
            self.all_states[i+1, :, 4] = ddtheta
            self.all_states[i+1, :, 5] = ddx

        # resetting sim states
        self.sim_states[:, 0] = self.theta0
        self.sim_states[:, 1] = self.dtheta0
        self.sim_states[:, 2] = self.x0
        self.sim_states[:, 3] = self.dx0

        sim_end_time = timeit.default_timer()
        print(f'{sim_end_time}: Simulation finished ({(sim_end_time- sim_start_time)/1000} ms)')

    # function to calculate fitness and sort individuals
    def Sorting(self):
        """
        Function to calculate fitness of each individual based on integral of error, then sorting.
        :return: None
        """
        sort_start_time = timeit.default_timer()
        print(f'{sort_start_time}: Sorting...')
        self.fitnesses = (np.sum(np.sqrt(self.all_states[:, :, 0]**2), axis = 0) \
                         + np.sum((self.all_states[:, :, 2]-self.x_tgt)**2, axis = 0))\
                         + 1
        self.gains[5, :] = self.fitnesses

        self.sorted_gains = np.transpose(np.array(sorted(np.transpose(self.gains), key=take_last)))
        sort_end_time = timeit.default_timer()
        print(f'{sort_end_time}: Sorting finished ({(sort_end_time- sort_start_time)/1000} ms)')
    # function to breed the best individuals
    def Breed(self, mum, dad):
        """
        Function to breed the best individuals, VERIFIED
        :param mum: gains of mum
        :param dad: gains of dad
        :return: offspring with gains from both parents plus some mutation
        """
        # crossover
        child = (mum+dad)/2
        # mutation
        child += np.random.normal(0, self.mutation_rate, (4))
        return child

    # function to evolve the population
    def Evolve(self):
        """
        Function to evolve the population, VERIFIED
        :return:
        """
        pop_cutoff = round(self.elite_cutoff*self.size)
        parents = self.sorted_gains.copy()
        for i in range(self.size):
            a = np.random.randint(pop_cutoff)
            b = np.random.randint(pop_cutoff)
            # print(f'optimizer cut off indices: {a, b}')
            parent_a = parents[0:4, a]
            parent_b = parents[0:4, b]
            new_child = self.Breed(parent_a, parent_b)
            self.gains[0:4, i] = new_child

    # debugger function
    def PlotDebugger(self, epoch):
        """
        Function to plot the results
        :return:
        """
        plot_start_time = timeit.default_timer()
        print(f'{plot_start_time}: Debug plotting...')
        fig_debug.suptitle(f'Debugger epoch {epoch}')
        for i in range(self.size):
            subfig_debug[i%debugger_dims[0]][i%debugger_dims[1]].cla()
            subfig_debug[i%debugger_dims[0]][i%debugger_dims[1]].plot(self.time_list, self.all_states[:, i, 0])
            subfig_debug[i%debugger_dims[0]][i%debugger_dims[1]].set_ylim([-0.2, 0.2])

        for i in range(debugger_dims[0]):
            subfig_debug[i][0].set_ylabel('Position')

        for i in range(debugger_dims[1]):
            subfig_debug[-1][i].set_xlabel('Time')
        # plt.legend()
        plt.pause(0.001)
        plot_end_time = timeit.default_timer()
        print(f'{plot_end_time}: Debug plotting finished ({(plot_end_time- plot_start_time)/1000} ms)')

    # function to display the GA results
    def Display(self, i):
        """
        Function to display the best individual
        :return:
        """
        best_one_rn = self.sorted_gains[:, 0].transpose()
        self.best_gains_history.append(list(best_one_rn))
        if best_one_rn[-1] < self.best_fitness:
            # print(f'New best fitness!\ngains: {best_one_rn[0:4]}')
            subfig1[0].cla()
            subfig2.cla()
            self.best_fitness = best_one_rn[-1]
            self.best_gains = best_one_rn[0:4]
            self.best_gains_index = best_one_rn[-2]
            self.index = int(self.best_gains_index)
            # print(index)

            subfig1[0].set_title(f'Best individual sofar')
            color = 'tab:red'
            subfig1[0].plot(self.time_list, self.all_states[:, self.index, 0], 'orangered', label='angle')
            # print(self.all_states[:, index, 0])
            # plt.plot(self.x_tgt, 0, 'ro')
            subfig1[0].set_xlabel('time [s]')
            subfig1[0].set_ylabel('theta', color=color)
            subfig1[0].tick_params(axis='y', labelcolor=color)

            color = 'tab:blue'
            subfig2.plot(self.time_list, self.all_states[:, self.index, 2], 'dodgerblue', label='position')
            subfig2.set_ylabel('position', color=color)
            subfig2.tick_params(axis='y', labelcolor=color)
            fig1.tight_layout()  # otherwise the right y-label is slightly clipped

        print(f'Best fitness: {self.best_fitness}')

        # plotting fitness history
        subfig1[1].cla()
        subfig1[1].set_title('Fitness history')
        ndarray_gains = np.array(self.best_gains_history)
        fitness_to_plot = (ndarray_gains[:, -1])
        subfig1[1].plot(np.linspace(1, i+1, i+1), fitness_to_plot)
        subfig1[1].set_xlabel('Epoch')
        subfig1[1].set_ylabel('Fitness')
        plt.pause(0.1)



# model constants
g = 9.80665
l = 1
ixx = 1
m_cart = 1

# simulation constants
theta, dtheta, ddtheta = 0.2, 0, 0      # initial theta conditions
x, dx, ddx = 0, 0, 0                    # initial position conditions

x_tgt, dx_tgt = 0, 0                    # target position
timestep, t = 0.001, 0                  # timestep and time

# lists to store states
theta_list = []
x_list = []
ddx_list = []
x_tgt_list = []
t_list = []

# initial PD controller gains
theta_kp0 = -g*6
theta_kd0 = -2
x_kp0 = 1.3
x_kd0 = 0.3

# GA parameters
maximum_iterations = 100
pop_size = 35
mutate_rate = 0.4
elite_cutoff = 0.25
gains = [theta_kp0, theta_kd0, x_kp0, x_kd0]
time_list = np.arange(0, 12, timestep)
initial_conditions = [theta, dtheta, x, dx, ddtheta, ddx]
tgt = [x_tgt, dx_tgt]

# initializing the GA
optimizer = GeneticAlgo(maximum_iterations, elite_cutoff, gains, time_list, initial_conditions, tgt)
optimizer.PopInit(pop_size, mutate_rate)

### Genetic Algorithm Loop###
for j in range(maximum_iterations):
    epoc_start_time = timeit.default_timer()
    print(f'\n\n---------------Epoch: {j}---------------')
    optimizer.Simulate()
    optimizer.Sorting()
    optimizer.Evolve()
    optimizer.PlotDebugger(j)
    optimizer.Display(j)
    epoc_end_time = timeit.default_timer()
    print(f'Epoch duration: {epoc_end_time - epoc_start_time}')

best_gains = optimizer.best_gains
print(f'\n\nFinished optimizing\nBest gains: {best_gains}')
plt.show()



# pg viewer intialization
res = [700, 300]
origin = (res[0]/2, res[1]/2)
screen = pg.display.set_mode(res)
pg.display.set_caption('pendulum')
pg.init()

# define colors
white = (255, 255, 255)
red = (255, 0, 50)
light_red = (255, 194, 194)
blue = (0, 50, 255)
light_blue = (194, 194, 255)
green = (50, 255, 0)
light_green = (194, 255, 194)
gold = (255, 223, 0)


# simulation consts
theta_kp = best_gains[0]
theta_kd = best_gains[1]
x_kp = best_gains[2]
x_kd = best_gains[3]

## Pygame Simulation Loop###
running = 1
play = 0
while running:
    toc = timeit.default_timer()
    # Clear screen
    screen.fill((0, 0, 0))

    dt = timestep*play
    t += dt
    t_list.append(t)

    # dynamics stuff
    ddx = theta*theta_kp+dtheta*theta_kd+(x-x_tgt)*x_kp+(dx-dx_tgt)*x_kd
    # ddx = -theta*g*2-dtheta*4+(x-x_tgt)*0.6+(dx-dx_tgt)*0.8

    # integration stuff
    dx += ddx*dt
    x += dx*dt

    ddtheta = theta_dd(theta, dtheta, ddx)
    dtheta += ddtheta*dt
    theta += dtheta*dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi


    x_list.append(x)
    x_tgt_list.append(x_tgt)
    theta_list.append(theta)

    # real coords
    o_x = x
    o_y = 0
    tip_x = -np.sin(theta) + x
    tip_y = np.cos(theta)

    # converting real coords into window coords
    win_o_x = res[0]/2 + 100*o_x
    win_o_y = res[1]/2 - 100*o_y
    win_tip_x = res[0]/2 + 100*tip_x
    win_tip_y = res[1]/2 - 100*tip_y
    win_tgt_x = res[0]/2 + 100*x_tgt
    win_tgt_y = res[1]/2

    pg.draw.aaline(screen, white, (win_o_x, win_o_y), (win_tip_x, win_tip_y))
    pg.draw.circle(screen, red, (win_tgt_x, win_tgt_y), 2)
    for i in range(10):
        win_tick_x = res[0]/2 + 100*i
        win_tick_y = res[1]/2
        pg.draw.circle(screen, white, (win_tick_x, win_tick_y), 0.5)

    for event in pg.event.get():
        # Stay in main loop until pygame.quit event is sent
        if event.type == pg.QUIT:
            running = 0
        # If a key on the keyboard is pressed
        elif event.type == pg.KEYDOWN:
            # Escape key, end game
            if event.key == pg.K_ESCAPE:
                running = 0
            if event.key == pg.K_RIGHT:
                x_tgt += 1.5
            if event.key == pg.K_LEFT:
                x_tgt -= 1.5
            if event.key == pg.K_UP:
                theta += 0.2
            if event.key == pg.K_DOWN:
                theta -= 0.2
            if event.key == pg.K_SPACE:
                if play == 0:
                    play = 1
                else:
                    play = 0
    pg.display.flip()

    tic = timeit.default_timer()

    print(f'frame time: {1000*round(tic-toc, 8)} ms')

pg.quit()


fig, axs = plt.subplots(2, 1, sharex='col')
fig.suptitle('Inverted pendulum :0000')

axs[0].plot(t_list, x_list, label='x')
axs[0].plot(t_list, x_tgt_list, label='x tgt')
axs[0].legend(loc='upper right')
axs[0].grid()
axs[0].set_ylabel('[m]')
axs[1].plot(t_list, theta_list, label='theta')
axs[1].legend(loc='upper right')
axs[1].grid()
axs[1].set_ylabel('[rad]')
plt.show()

