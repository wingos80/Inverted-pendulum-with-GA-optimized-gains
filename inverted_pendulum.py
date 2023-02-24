from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
import timeit

np.random.seed(123)

def theta_dd(theta, dtheta, x_dd):
    return (g*l*np.sin(theta)+x_dd*l*np.cos(theta))/(ixx+l**2)-dtheta/100


### Genetic Alogorithm ###
fig1, subfig1 = plt.subplots(1,2)
subfig2 = subfig1[0].twinx()
# fig, ax = plt.subplots(1,1)
fig1.suptitle('Inverted pendulum')
# fig.suptitle('Gains and shit')

debugger_dims = (4, 5)


# function used for python sorted()
def take_last(elem):
    return elem[7]

class GeneticAlgo():
    def __init__(self, max_itr, elite_cutoff, init_gains, time_list, init_conds, tgt, debug):
        self.debug= debug
        if debug:
            self.fig_debug, self.subfig_debug = plt.subplots(debugger_dims[0], debugger_dims[1], sharex=True, sharey=True)
            self.fig_debug.suptitle('Debugger.... ')
        self.max_itr = max_itr
        self.elite_cutoff = elite_cutoff
        self.theta_kp = init_gains[0]
        self.theta_kd = init_gains[1]
        self.x_kp = init_gains[2]
        self.x_kd = init_gains[3]
        self.t_tgt_k = init_gains[4]
        self.t_tgt_cap = init_gains[5]
        self.time_list = time_list

        self.theta0, self.dtheta0, self.ddtheta0 = init_conds[0], init_conds[1], init_conds[4]
        self.x0, self.dx0, self.ddx0 = init_conds[2], init_conds[3], init_conds[5]
        self.theta_tgt, self.dtheta_tgt, self.x_tgt, self.dx_tgt = tgt[0], tgt[1], tgt[2], tgt[3]

        self.update_best = 0
        self.best_fitness = 9999999
        self.best_gains = np.zeros(6)
        self.best_gains_history = []
        self.best_gains_index = 0


    # function to initialize population
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

        # initialize gains  (gain1, gain2, gain3, gain4, gain5, gain6, gain id, gain fitness)
        self.gains = np.random.rand(8, self.size)
        self.gains[0, :] = (self.gains[0, :] - 0.5) * self.theta_kp + self.theta_kp
        self.gains[1, :] = (self.gains[1, :] - 0.5) * self.theta_kd + self.theta_kd
        self.gains[2, :] = (self.gains[2, :] - 0.5) * self.x_kp + self.x_kp
        self.gains[3, :] = (self.gains[3, :] - 0.5) * self.x_kd + self.x_kd
        self.gains[4, :] = (self.gains[4, :] - 0.5) * self.t_tgt_k/1000  + self.t_tgt_k
        self.gains[5, :] = (self.gains[5, :] - 0.5) * self.t_tgt_cap/1000 + self.t_tgt_cap
        self.gains[6, :] = np.linspace(0, self.size-1, self.size)                       # index id of each gain


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
        print(f'{init_pop_end_time}: Population initialized ({(init_pop_end_time- init_pop_start_time)*1000} ms)')

    # function to simulate the system
    def Simulate(self):
        """
        Function simulating the pendulum for a given time index. VERIFIED
        :return:
        """
        sim_start_time = timeit.default_timer()
        print(f'{sim_start_time}: Simulating...')
        dt = timestep

        # initialize random target states for each epoch
        tgt_arr = np.zeros((self.size, 4))
        self.epoch_rand = x_init_sigma*np.random.randn()
        tgt_arr[:,2] = self.epoch_rand*np.ones(self.size)

        for i in range(len(self.time_list)-1):
            # tgt_arr is for making theta target vary from 0 as the pendulum moves
            self.t_tgt = np.maximum(np.minimum(((self.sim_states[:,2]-self.x_tgt).reshape(self.size, 1)**3@(self.gains[4,:]).reshape(1, self.size)).diagonal(), self.gains[5,:]), -self.gains[5,:])

            tgt_arr[:,0] = self.t_tgt

            err = self.sim_states-tgt_arr               # finding PD errors

            ddx = (err @ self.gains[0:4,:]).diagonal()                                      # calculating ddx from gains
            self.sim_states[:,3] += ddx*dt                                                  # updating speed
            self.sim_states[:,2] += self.sim_states[:,3]*dt                                 # updating position
            # (g * l * np.sin(theta) + x_dd * l * np.cos(theta)) / (ixx + l ** 2) - dtheta / 100
            # ddtheta = theta_dd(self.sim_states[:,0], self.sim_states[:,1], ddx)             # calculating ddtheta from ddx
            ddtheta = (g * l * np.sin(self.sim_states[:,0]) + ddx * l * np.cos(self.sim_states[:,0])) / (ixx + l ** 2) - self.sim_states[:,1] / 100
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
        print(f'{sim_end_time}: Simulation finished ({(sim_end_time- sim_start_time)*1000} ms)')

    # function to calculate fitness and sort individuals
    def Sorting(self):
        """
        Function to calculate fitness of each individual based on integral of error, then sorting.
        :return: None
        """
        sort_start_time = timeit.default_timer()
        print(f'{sort_start_time}: Sorting...')
        self.fitnesses = 1*np.sum(np.sqrt(self.all_states[:, :, 0]**2), axis = 0) \
                         + 1*np.sum(np.sqrt((self.all_states[:, :, 2]-self.epoch_rand)**2), axis = 0)
        # print('shape of fitness maybe idfk', np.shape(self.fitnesses)[0])
        # self.avg_fit = np.sum(self.fitnesses)/np.shape(self.fitnesses)[0]
        # print('avg_fitness', self.avg_fit)
        self.gains[-1, :] = self.fitnesses*timestep

        self.sorted_gains = np.transpose(np.array(sorted(np.transpose(self.gains), key=take_last)))
        # print(f'gains:{self.gains}\n\n sorted gains: {self.sorted_gains}')
        # print(f'worst one rn: {worst_one_rn}\nworst index: {self.worst_index}')
        self.avg_fit, self.best_fit, self.worst_fit = np.sum(self.fitnesses)/np.shape(self.fitnesses)[0], self.sorted_gains[-1,0], self.sorted_gains[-1,-1]
        sort_end_time = timeit.default_timer()
        print(f'{sort_end_time}: Sorting finished ({(sort_end_time- sort_start_time)*1000} ms)')

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
        mutate = np.random.normal(0, self.mutation_rate, (6))
        mutate[4:] = np.random.normal(0,self.mutation_rate/100)
        child += mutate
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
            parent_a = parents[0:6, a]
            parent_b = parents[0:6, b]
            new_child = self.Breed(parent_a, parent_b)
            self.gains[0:6, i] = new_child

    def FindingBest(self):
        """
        Function to find the best gains so that they can be plotted
        :return:
        """
        # interim = self.sorted_gains.copy()
        # best_one_rn = np.transpose(interim)[0]
        # print(f'unsorted: {self.gains}\n sorted: {self.sorted_gains}')
        best_one_rn = self.sorted_gains[:, 0].transpose()
        # print(f'0th gain{self.sorted_gains[:, 0]}\nlast-th gain {self.sorted_gains[:, -1]}')
        self.best_gains_history.append(list(best_one_rn))
        self.index = int(best_one_rn[-2])
        if best_one_rn[-1] < self.best_fitness: # << some trouble here, best theta seems to be updating when it shouldnt
            self.update_best = 1
            self.best_fitness = best_one_rn[-1]
            self.best_gains = best_one_rn[0:-2]
            self.best_gains_index = int(best_one_rn[-2])

            self.best_theta = self.all_states[:, self.best_gains_index, 0].copy()
            self.best_pos = self.all_states[:, self.best_gains_index, 2].copy()


    # function to display the GA results
    def Display(self, i):
        """
        Function to display the best individual
        :return:
        """
        if self.debug:
            self.PlotDebugger(i)

        find_best_one_start_time = timeit.default_timer()
        print(f'{find_best_one_start_time}: Plotting best individual...')
        if self.update_best:
            subfig1[0].cla()
            subfig2.cla()
            subfig1[0].set_title(f'Best individual sofar')
            subfig1[0].plot(self.time_list, self.best_theta, 'orangered', label='angle')
            subfig1[0].set_xlabel('Time [s]')
            subfig1[0].set_ylabel('Angle [rad]', color='tab:red')
            subfig1[0].tick_params(axis='y', labelcolor='tab:red')
            subfig1[0].grid()
            # subfig1[0].set_ylim([-0.33,0.33])

            subfig2.plot(self.time_list, self.best_pos, 'dodgerblue', label='pos')
            subfig2.axhline(y=self.epoch_rand, xmin=0, xmax=ga_time_cap, c='black', linestyle='dotted')
            subfig2.set_ylabel('Pos [m]', color='tab:blue')
            subfig2.tick_params(axis='y', labelcolor='tab:blue')
            # subfig2.grid()
            # subfig2.set_ylim([-0.33, 0.33])

            fig1.tight_layout()  # otherwise the right y-label is slightly clipped

            self.update_best = 0


        # plotting fitness history
        if i < (self.max_itr-1):                                        # checking if its the last epoch yet
            subfig1[1].cla()


        # ax.cla()
        # ax.boxplot(self.gains[0:-2].T, 0, '')
        subfig1[1].set_title('Improvement history')
        ndarray_gains_his = np.array(self.best_gains_history)
        fitness_to_plot = (ndarray_gains_his[:, -1])
        print(f'Best gain: {self.best_gains_history[-1][0:6]}')
        print(f'Best fitness: {self.best_gains_history[0][-1]/self.best_fitness}')
        fitness_to_plot[:] = fitness_to_plot[0]/fitness_to_plot[:]      # normalizing fitness
        subfig1[1].plot(np.linspace(1, i+1, i+1), fitness_to_plot)
        subfig1[1].set_xlabel('Epoch')
        subfig1[1].set_ylabel('Imporvement')
        subfig1[1].grid()

        find_the_best_one_end_time = timeit.default_timer()
        print(f'{find_the_best_one_end_time}: Plotting best individual finished ({(find_the_best_one_end_time- find_best_one_start_time)*1000} ms)')


    def oldDisplay(self, i):
        """
        Function to display the best individual
        :return:
        """
        if self.debug:
            self.PlotDebugger(i)

        find_best_one_start_time = timeit.default_timer()
        print(f'{find_best_one_start_time}: Plotting best individual...')
        if self.update_best:
            self.testt1 = self.all_states[:, self.index, 0]  # useless!!!
            self.testt2 = self.all_states[:, self.best_gains_index, 0]  # useless!!!
            self.testt3 = self.best_theta
            color = 'tab:red'
            subfig1[1,0].cla()
            subfig1[1,0].set_title(f'Best individual sofar')
            subfig1[1,0].plot(self.time_list, self.best_theta, 'orangered', label='all time best angle')
            subfig1[1,0].set_xlabel('Time [s]')
            subfig1[1,0].set_ylabel('Angle [rad]', color='tab:red')
            subfig1[1,0].tick_params(axis='y', labelcolor='tab:red')
            subfig1[1,0].grid()
            subfig1[1,0].set_ylim([-0.33,0.33])

            subfig1[0,0].cla()
            subfig1[0,0].plot(self.time_list, self.best_pos, 'dodgerblue', label='all time best position')
            subfig1[0,0].set_xlabel('Time [s]')
            subfig1[0,0].set_ylabel('Position [m]', color='tab:blue')
            subfig1[0,0].tick_params(axis='y', labelcolor='tab:blue')
            subfig1[0,0].grid()
            subfig1[0,0].set_ylim([-2.1,0.5])

            fig1.tight_layout()  # otherwise the right y-label is slightly clipped

            self.update_best = 0

        subfig2.cla()
        color = 'tab:blue'
        subfig2.plot(self.time_list, self.all_states[:, self.index, 0], 'orangered', label='instant best theta', linestyle='dotted')
        subfig2.set_ylabel('Angle [rad]', color='tab:red')
        subfig2.tick_params(axis='y', labelcolor='tab:red')
        subfig2.grid()
        subfig2.set_ylim([-0.33, 0.33])

        subfig3.cla()
        subfig3.plot(self.time_list, self.all_states[:, self.index, 2], 'dodgerblue',label='instant best position', linestyle='dotted')
        subfig3.set_xlabel('Time [s]')
        subfig3.set_ylabel('Position [m]', color='tab:blue')
        subfig3.tick_params(axis='y', labelcolor='tab:blue')
        subfig3.grid()
        subfig3.set_ylim([-2.1,0.5])

        # plotting fitness history
        if i < (self.max_itr-1):                                        # checking if its the last epoch yet
            subfig1[0,1].cla()


        # ax.cla()
        # ax.boxplot(self.gains[0:-2].T, 0, '')
        subfig1[0,1].set_title('Fitness history')
        ndarray_gains = np.array(self.best_gains_history)
        fitness_to_plot = (ndarray_gains[:, -1])

        print(f'Best fitness: {self.best_fitness}')
        # fitness_to_plot[:] = fitness_to_plot[0]/fitness_to_plot[:]      # normalizing fitness
        subfig1[0,1].plot(np.linspace(1, i+1, i+1), fitness_to_plot)
        subfig1[0,1].set_xlabel('Epoch')
        subfig1[0,1].set_ylabel('Fitness')
        subfig1[0,1].grid()

        find_the_best_one_end_time = timeit.default_timer()
        print(f'{find_the_best_one_end_time}: Plotting best individual finished ({(find_the_best_one_end_time- find_best_one_start_time)*1000} ms)')

        fig1.legend()

    # debugger function
    def PlotDebugger(self, epoch):
        """
        Function to plot the results
        :return:
        """
        if epoch % 1 == 0:
            plot_start_time = timeit.default_timer()
            print(f'{plot_start_time}: Debug plotting...')
            self.fig_debug.suptitle(f'Debugger epoch {epoch}')
            for i in range(debugger_dims[0]*debugger_dims[1]):
                self.subfig_debug[i%debugger_dims[0]][i%debugger_dims[1]].cla()
                self.subfig_debug[i%debugger_dims[0]][i%debugger_dims[1]].plot(self.time_list, self.all_states[:, i, 0], label=f'{self.gains[-1, i]}')
                self.subfig_debug[i % debugger_dims[0]][i % debugger_dims[1]].plot(self.time_list, self.all_states[:, i, 2])
                self.subfig_debug[i%debugger_dims[0]][i%debugger_dims[1]].set_ylim([-1.1, 1.1])

            # best and worst gains plotting
            self.subfig_debug[0][0].cla()
            self.subfig_debug[0][0].plot(self.time_list, self.all_states[:, self.index, 0], label=f'{self.gains[-1, self.index]}')
            self.subfig_debug[0][0].plot(self.time_list, self.all_states[:, self.index, 2],)
            self.subfig_debug[0][0].set_ylim([-1.1, 1.1])
            self.subfig_debug[0][0].set_title('Best gains')

            worst_one_rn = self.sorted_gains[:, -1].transpose()
            self.worst_index = int(worst_one_rn[-2])
            self.subfig_debug[0][1].cla()
            self.subfig_debug[0][1].plot(self.time_list, self.all_states[:, self.worst_index, 0], label=f'{self.gains[-1, self.worst_index]}')
            self.subfig_debug[0][1].plot(self.time_list, self.all_states[:, self.worst_index, 2], )
            self.subfig_debug[0][1].set_ylim([-1.1, 1.1])
            self.subfig_debug[0][1].set_title('Worst gains')

            for i in range(debugger_dims[0]):
                self.subfig_debug[i][0].set_ylabel('Angle [rad]')

            for i in range(debugger_dims[1]):
                self.subfig_debug[-1][i].set_xlabel('Time [s]')
            # plt.legend()
            plot_end_time = timeit.default_timer()
            print(f'{plot_end_time}: Debug plotting finished ({(plot_end_time- plot_start_time)*1000} ms)')

# model constants
g = 9.80665
l = 1
ixx = 1
m_cart = 1

# simulation constants
theta, dtheta, ddtheta = 0.5, 0, 0      # initial theta conditions
x, dx, ddx = 0.0, 0, 0                    # initial position conditions

theta_tgt, dtheta_tgt, x_tgt, dx_tgt = 0, 0, 0, 0                    # target position
timestep, t = 0.02, 0                  # timestep and time


# # initial PD controller gains
# theta_kp0 = -g*4
# theta_kd0 = -2
# x_kp0 = 0.5
# x_kd0 = 1.2

# initial PD controller gains
theta_kp0 = -30
theta_kd0 = -3
x_kp0 = 1
x_kd0 = 2
t_tgt_k = 1/30
t_tgt_cap = 0.2

best_gains = [theta_kp0, theta_kd0, x_kp0, x_kd0, t_tgt_k, t_tgt_cap]

# GA parameters
maximum_iterations = 88
pop_size = int(0.2*10**3)
mutate_rate = 0.9
x_init_sigma = 0.0
elite_cutoff = 0.1
ga_time_cap = 13.5
gains = [theta_kp0, theta_kd0, x_kp0, x_kd0, t_tgt_k, t_tgt_cap]
time_list = np.arange(0, ga_time_cap, timestep)
initial_conditions = [theta, dtheta, x, dx, ddtheta, ddx]
tgt = [theta_tgt, dtheta_tgt, x_tgt, dx_tgt]

# initializing the GA
optimizer = GeneticAlgo(maximum_iterations, elite_cutoff, gains, time_list, initial_conditions, tgt, debug=0)
optimizer.PopInit(pop_size, mutate_rate)


### Genetic Algorithm Loop###
for j in range(maximum_iterations):
    epoc_start_time = timeit.default_timer()
    print(f'\n\n---------------Epoch: {j}---------------')
    optimizer.Simulate()
    optimizer.Sorting()
    optimizer.Evolve()
    optimizer.FindingBest()
    optimizer.Display(j)
    plt.pause(0.001)
    epoc_end_time = timeit.default_timer()
    print(f'Epoch duration: {epoc_end_time - epoc_start_time} s')
plt.show()



best_gains = optimizer.best_gains
print(f'\n\n>>>>>>Finished optimizing<<<<<<\nBest gains: {best_gains}\n\n')
plt.show()

file1 = open("best_gains.txt", "w")
file1.write(f'{best_gains, timeit.default_timer()} \n')
file1.close()


# pg viewer intialization
res = [700, 300]
target_fps = 90
origin = (res[0]/2, res[1]/2)
screen = pg.display.set_mode(res)
pg.display.set_caption('pendulum')
pg.init()

# Inititalize main sim clock
clock = pg.time.Clock()

# define colors
white = (255, 255, 255)
red = (255, 0, 50)
light_red = (255, 194, 194)
blue = (0, 50, 255)
light_blue = (194, 194, 255)
green = (50, 255, 0)
light_green = (194, 255, 194)
gold = (255, 223, 0)


# lists to store states
theta_list = []
theta_tgt_list = []
dtheta_list = []
dtheta_tgt_list = []
x_list = []
ddx_list = []
x_tgt_list = []
t_list = []


# simulation consts
theta_kp = best_gains[0]
theta_kd = best_gains[1]
x_kp = best_gains[2]
x_kd = best_gains[3]
t_tgt_k = best_gains[4]
t_tgt_cap = best_gains[5]
theta_tgt, dtheta_tgt, x_tgt, dx_tgt = 0, 0, 0, 0                    # target position
timestep = 0.1

print("\n\ngains used:  ", best_gains, "\n\n")

## Pygame Simulation Loop###
frame_time = 0
frame_list = []
running = 1
play = 0
while running:
    toc = timeit.default_timer()
    # dt_frame = timestep
    dt_frame = clock.tick(target_fps) / 1000
    # Clear screen

    screen.fill((0, 0, 0))

    dt = dt_frame*play
    t += dt
    t_list.append(t)

    # dynamics stuff
    theta_tgt = max(min((x-x_tgt)**3*t_tgt_k, t_tgt_cap), -t_tgt_cap)
    ddx = (theta-theta_tgt)*theta_kp+(dtheta)*theta_kd+(x-x_tgt)*x_kp+(dx-dx_tgt)*x_kd
    # ddx = -theta*g*2-dtheta*4+(x-x_tgt)*0.6+(dx-dx_tgt)*0.8
    ddx_list.append(ddx)
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
    theta_tgt_list.append(theta_tgt)

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


    frame_list.append(1000*(tic-toc))
    if len(frame_list) > 200:
        print(f'frame time: {round(sum(frame_list)/len(frame_list), 8)} ms')
        frame_list = []


pg.quit()


fig, axs = plt.subplots(2, 2, sharex=True)
fig.suptitle('Inverted pendulum :0000')

axs[0,0].plot(t_list, x_list, label='x')
axs[0,0].plot(t_list, x_tgt_list, label='x tgt')
axs[0,0].legend(loc='upper right')
axs[0,0].grid()
axs[0,0].set_ylabel('[m]')
axs[1,0].plot(t_list, theta_list, label='theta')
axs[1,0].plot(t_list, theta_tgt_list, label='theta tgt')
axs[1,0].legend(loc='upper right')
axs[1,0].grid()
axs[1,0].set_ylabel('[rad]')
axs[0,1].plot(t_list, ddx_list, label='ddx')
axs[0,1].legend(loc='upper right')
axs[0,1].grid()
plt.show()

if t_list[-1] > ga_time_cap:
    time1 = [x for x in t_list if x < ga_time_cap]
else:
    time1 = t_list

plt.title('PG vs GA loop theta history comparison')
plt.plot(time1, theta_list[0:len(time1)], label='PG loop')
plt.plot(time_list, optimizer.best_theta, label='GA loop')
plt.legend()
plt.grid()
plt.show()