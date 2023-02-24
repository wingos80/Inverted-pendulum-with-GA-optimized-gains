import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
import timeit

res = [700, 300]
origin = (res[0]/2, res[1]/2)
# screen = pg.display.set_mode(res)
# pg.display.set_caption('pendulum')
# pg.init()

# define colors
white = (255, 255, 255)
red = (255, 0, 50)
light_red = (255, 194, 194)
blue = (0, 50, 255)
light_blue = (194, 194, 255)
green = (50, 255, 0)
light_green = (194, 255, 194)
gold = (255, 223, 0)

# model constants
g = 9.80665
l = 1
ixx = 1
m_cart = 1

# simulation constants
theta, dtheta, ddtheta = 0.2, 0, 0
x, dx = 0, 0
x_tgt, dx_tgt = 0, 0
timestep, t = 0.001, 0

theta_list = []
x_list = []
ddx_list = []
x_tgt_list = []
t_list = []

# initial PD controller gains
theta_kp = -g*4
theta_kd = -2
x_kp = 1.3
x_kd = 0.3

def theta_dd(theta, dtheta, x_dd):
    return (g*l*np.sin(theta)+x_dd*l*np.cos(theta))/(ixx+l**2)-dtheta/100


### Genetic Alogorithm ###
fig1, subfig1 = plt.subplots(2,3, sharex=True, sharey=True)
# subfig2 = subfig[0].twinx()
fig1.suptitle('Optimizer results.... ')

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

        self.best_fitness = 0
        self.best_gains = np.zeros(6)
        self.best_gains_history = []
        self.best_gains_index = 0

    def PopInit(self, size, mutation_rate):
        """
        Function returning an array of PD populations
        :param size: size of population
        :return: np array of everyones' gains, fitness, and states history
        """
        init_pop_start_time = timeit.default_timer()
        print(f'{init_pop_start_time}: Initializing population...')
        self.size = size
        self.mutation_rate = mutation_rate

        self.gains = np.random.rand(6, self.size)
        self.gains[0, :] = (self.gains[0, :] - 0.5) * self.theta_kp + self.theta_kp
        self.gains[1, :] = (self.gains[1, :] - 0.5) * self.theta_kd + self.theta_kd
        self.gains[2, :] = (self.gains[2, :] - 0.5) * self.x_kp + self.x_kp
        self.gains[3, :] = (self.gains[3, :] - 0.5) * self.x_kd + self.x_kd
        self.gains[4, :] = np.linspace(0, self.size-1, self.size)

        self.fitnesses = np.zeros((self.size))

        self.sim_states = np.zeros((self.size, 4)) # theta, dtheta, x, dx
        self.sim_states[:, 0] = self.theta0
        self.sim_states[:, 1] = self.dtheta0
        self.sim_states[:, 2] = self.x0
        self.sim_states[:, 3] = self.dx0

        self.all_states = np.zeros((self.time_list.shape[0], self.size, 6))     # theta, dtheta, x, dx, ddtheta, ddx
        self.all_states[0, :, 0:4] = self.sim_states

        init_pop_end_time = timeit.default_timer()
        print(f'{init_pop_end_time}: Population initialized ({(init_pop_end_time- init_pop_start_time)/1000} ms)')

    def Simulate(self):
        """
        Function simulating the pendulum for a given time index
        :return:
        """
        dt = timestep
        err = self.sim_states - np.array([[0, 0, x_tgt, dx_tgt]])
        ddx = (err @ self.gains[0:4, :]).diagonal()
        print(f'from the optimizer!!!{err}')
        for i in range(len(self.time_list)-1):
            err = self.sim_states-np.array([[0, 0, x_tgt, dx_tgt]])                         # finding PD errors
            # print(np.shape(err), '\n\n', np.shape(self.gains[0:4,:]))
            ddx = (err @ self.gains[0:4,:]).diagonal()                                      # calculating ddx from gains
            self.sim_states[:,3] += ddx*dt                                                  # updating speed
            self.sim_states[:,2] += self.sim_states[:,3]*dt                                 # updating position

            ddtheta = theta_dd(self.sim_states[:,0], self.sim_states[:,1], ddx)             # calculating ddtheta from ddx
            self.sim_states[:,1] += ddtheta * dt                                            # updating angular speed
            self.sim_states[:,0] += self.sim_states[:,1] * dt                               # updating angle
            self.sim_states[:,0] = (self.sim_states[:,0] + np.pi) % (2 * np.pi) - np.pi     # normalize angle to be between 0 and 2 pi
            # print(np.shape(ddtheta), '\n\n', np.shape(ddx), '\n\n', np.shape(self.all_states[i+1, :, 4:6]))
            # Logging sim states to all states
            self.all_states[i+1, :, 0:4] = self.sim_states
            self.all_states[i+1, :, 4] = ddtheta
            self.all_states[i+1, :, 5] = ddx

        # resetting sim states
        self.sim_states[:, 0] = self.theta0
        self.sim_states[:, 1] = self.dtheta0
        self.sim_states[:, 2] = self.x0
        self.sim_states[:, 3] = self.dx0


    def Sorting(self):
        """
        Function to calculate fitness of each individual
        :return: None
        """
        self.fitnesses = 1/(np.sum(np.sqrt(self.all_states[:, :, 0]**2), axis = 0) \
                         + 5*np.sum((self.all_states[:, :, 2]-self.x_tgt)**2, axis = 0))
        self.gains[5, :] = self.fitnesses

        self.sorted_gains = np.transpose(np.array(sorted(np.transpose(self.gains), key=take_last)))

    # function to breed the best individuals
    def Breed(self, mum, dad):
        # crossover
        child = (mum+dad)/2
        # mutation
        child += np.random.normal(0, self.mutation_rate, (4))
        return child

    def Evolve(self):
        """
        Function to evolve the population
        :return:
        """
        pop_cutoff = round(self.elite_cutoff*self.size)
        parents = self.sorted_gains.copy()
        for i in range(self.size):
            parent_a = parents[0:4, np.random.randint(pop_cutoff)]
            parent_b = parents[0:4, np.random.randint(pop_cutoff)]
            new_child = self.Breed(parent_a, parent_b)
            self.gains[0:4, i] = new_child

    def Plot(self):
        """
        Function to plot the results
        :return:
        """
        for i in range(self.size):
            subfig1[i%2][i%3].cla()
            subfig1[i%2][i%3].plot(self.time_list, self.all_states[:, i, 0], label=f'gain{i} pos')
            subfig1[i % 2][i % 3].set_title(f'Gain {i}')
            subfig1[i % 2][i % 3].set_ylabel('Position')
            subfig1[i % 2][i % 3].set_xlabel('Time')

    def Display(self, i):
        """
        Function to display the best individual
        :return:
        """
        best_one_rn = self.sorted_gains[:, -1].transpose()
        self.best_gains_history.append(list(best_one_rn))
        if best_one_rn[-1] > self.best_fitness:
            print(f'New best fitness!\ngains: {best_one_rn[0:4]}')
            subfig[0].cla()
            subfig2.cla()
            self.best_fitness = best_one_rn[-1]
            self.best_gains = best_one_rn[0:4]
            self.best_gains_index = best_one_rn[-2]
            self.index = int(self.best_gains_index)
            # print(index)

            subfig[0].set_title('Best individual')
            color = 'tab:red'
            subfig[0].plot(self.time_list, self.all_states[:, self.index, 0], 'orangered', label='angle')
            # print(self.all_states[:, index, 0])
            # plt.plot(self.x_tgt, 0, 'ro')
            subfig[0].set_xlabel('time [s]')
            subfig[0].set_ylabel('theta', color=color)
            subfig[0].tick_params(axis='y', labelcolor=color)


            color = 'tab:blue'
            subfig2.plot(self.time_list, self.all_states[:, self.index, 2], 'dodgerblue', label='position')
            subfig2.set_ylabel('position', color=color)
            subfig2.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped

        print(f'Best fitness: {self.best_fitness}')
        # print(self.best_gains_history[0][-1])
        # plotting fitness history
        subfig[1].cla()
        ndarray_gains = np.array(self.best_gains_history)
        fitness_to_plot = (ndarray_gains[:, -1])
        subfig[1].plot(np.linspace(1, i+1, i+1), fitness_to_plot)
        plt.pause(0.1)


# initializing the GA
maximum_iterations = 10
elite_cutoff = 0.5
gains = [theta_kp, theta_kd, x_kp, x_kd]
time_list = np.arange(0, 20, timestep)
initial_conditions = [theta, dtheta, x, dx, 0, 0]
tgt = [x_tgt, dx_tgt]
optimizer = GeneticAlgo(maximum_iterations, elite_cutoff, gains, time_list, initial_conditions, tgt)

pop_size = 6
optimizer.PopInit(pop_size, 0.05)

#running the GA
fig2, subfig2 = plt.subplots(2, 3, sharex=True, sharey=True)
# subfig2 = subfig[0].twinx()
fig2.suptitle('Verifying optimizer results ')
for j in range(maximum_iterations):
    optimizer.Simulate()
    test_gains = np.zeros((5, pop_size))
    for i in range(pop_size):
        test_gains[0, i] = optimizer.gains[0, i]
        test_gains[1, i] = optimizer.gains[1, i]
        test_gains[2, i] = optimizer.gains[2, i]
        test_gains[3, i] = optimizer.gains[3, i]
        # theta_kp = optimizer.gains[0, i]
        # theta_kd = optimizer.gains[1, i]
        # x_kp = optimizer.gains[2, i]
        # x_kd = optimizer.gains[3, i]
        # print(f'gains from sorted gains: {theta_kp}, {theta_kd}, {x_kp}, {x_kd} \n\n {theta}')

    test_sim_states = np.zeros((pop_size, 4))  # theta, dtheta, x, dx
    test_sim_states[:, 0] = theta
    test_sim_states[:, 1] = dtheta
    test_sim_states[:, 2] = x
    test_sim_states[:, 3] = dx

    test_all_states = np.zeros((time_list.shape[0], pop_size, 6))  # theta, dtheta, x, dx, ddtheta, ddx
    test_all_states[0, :, 0:4] = test_sim_states

    err = test_sim_states - np.array([[0, 0, x_tgt, dx_tgt]])
    ddx = (err @ test_gains[0:4, :]).diagonal()
    print(f'from the verifizer!!!{err}\n')
    for k in range(len(time_list)-1):
        err = test_sim_states - np.array([[0, 0, x_tgt, dx_tgt]])  # finding PD errors
        ddx = (err @ test_gains[0:4, :]).diagonal()  # calculating ddx from gains
        test_sim_states[:, 3] += ddx * timestep  # updating speed
        test_sim_states[:, 2] += test_sim_states[:, 3] * timestep  # updating position

        ddtheta = theta_dd(test_sim_states[:, 0], test_sim_states[:, 1], ddx)  # calculating ddtheta from ddx
        test_sim_states[:, 1] += ddtheta * timestep  # updating angular speed
        test_sim_states[:, 0] += test_sim_states[:, 1] * timestep  # updating angle
        test_sim_states[:, 0] = (test_sim_states[:, 0] + np.pi) % (
                    2 * np.pi) - np.pi  # normalize angle to be between 0 and 2 pi
        # Logging sim states to all states
        test_all_states[k + 1, :, 0:4] = test_sim_states
        test_all_states[k + 1, :, 4] = ddtheta
        test_all_states[k + 1, :, 5] = ddx

    for k in range(pop_size):
        # plotting test all states
        subfig2[k%2, k%3].cla()
        subfig2[k%2, k%3].plot(time_list, test_all_states[:, k, 0], label=f'gain{k} pos')

    optimizer.Sorting()
    # optimizer.Display(j)
    optimizer.Evolve()
    optimizer.Plot()

    plt.legend()
    plt.pause(0.2)

best_gains = optimizer.best_gains
print(f'Optimized gains (t_kp, t_kd, x_kp, x_kd): {best_gains}')
plt.show()



best_index = optimizer.index
angles = optimizer.all_states[:, best_index, 0]
plt.plot(time_list, angles)
print(f'gains from the optimizer: {optimizer.gains[0:4,best_index]}')
plt.show()
#
# for i in range(6):
#     theta_kp = optimizer.gains[0,i]
#     theta_kd = optimizer.gains[1,i]
#     x_kp = optimizer.gains[2,i]
#     x_kd = optimizer.gains[3,i]
#     print(f'gains from sorted gains: {theta_kp}, {theta_kd}, {x_kp}, {x_kd} \n\n {theta}')
#
#     for i in range(len(time_list-1)):
#         t += timestep
#         t_list.append(t)
#         ddx = theta*theta_kp + dtheta*theta_kd + (x-x_tgt)*x_kp + (dx-dx_tgt)*x_kd
#         dx += ddx*timestep
#         x += dx*timestep
#
#         ddtheta = theta_dd(theta, dtheta, ddx)
#         dtheta += ddtheta*timestep
#         theta += dtheta*timestep
#
#         x_list.append(x)
#         theta_list.append(theta)
#
#     plt.plot(t_list, x_list, label='x')
#     plt.plot(t_list, theta_list, label='theta')
#     plt.xlabel('time [s]')
#     plt.ylabel('position and angle')
#     plt.legend()
#     plt.show()


### Simulation Loop###
# running = 1
# play = 0
# while running:
#     toc = timeit.default_timer()
#     # Clear screen
#     screen.fill((0, 0, 0))
#
#     dt = timestep*play
#     t += dt
#     t_list.append(t)
#
#     # dynamics stuff
#     ddx = theta*theta_kp+dtheta*theta_kd+(x-x_tgt)*x_kp+(dx-dx_tgt)*x_kd
#     # ddx = -theta*g*2-dtheta*4+(x-x_tgt)*0.6+(dx-dx_tgt)*0.8
#
#     # integration stuff
#     dx += ddx*dt
#     x += dx*dt
#
#     ddtheta = theta_dd(theta, dtheta, ddx)
#     dtheta += ddtheta*dt
#     theta += dtheta*dt
#     theta = (theta + np.pi) % (2 * np.pi) - np.pi
#
#
#     x_list.append(x)
#     x_tgt_list.append(x_tgt)
#     theta_list.append(theta)
#
#     # real coords
#     o_x = x
#     o_y = 0
#     tip_x = -np.sin(theta) + x
#     tip_y = np.cos(theta)
#
#     # converting real coords into window coords
#     win_o_x = res[0]/2 + 100*o_x
#     win_o_y = res[1]/2 - 100*o_y
#     win_tip_x = res[0]/2 + 100*tip_x
#     win_tip_y = res[1]/2 - 100*tip_y
#     win_tgt_x = res[0]/2 + 100*x_tgt
#     win_tgt_y = res[1]/2
#
#     pg.draw.aaline(screen, white, (win_o_x, win_o_y), (win_tip_x, win_tip_y))
#     pg.draw.circle(screen, red, (win_tgt_x, win_tgt_y), 2)
#     for i in range(10):
#         win_tick_x = res[0]/2 + 100*i
#         win_tick_y = res[1]/2
#         pg.draw.circle(screen, white, (win_tick_x, win_tick_y), 0.5)
#
#     for event in pg.event.get():
#         # Stay in main loop until pygame.quit event is sent
#         if event.type == pg.QUIT:
#             running = 0
#         # If a key on the keyboard is pressed
#         elif event.type == pg.KEYDOWN:
#             # Escape key, end game
#             if event.key == pg.K_ESCAPE:
#                 running = 0
#             if event.key == pg.K_RIGHT:
#                 x_tgt += 1.5
#             if event.key == pg.K_LEFT:
#                 x_tgt -= 1.5
#             if event.key == pg.K_UP:
#                 theta += 0.2
#             if event.key == pg.K_DOWN:
#                 theta -= 0.2
#             if event.key == pg.K_SPACE:
#                 if play == 0:
#                     play = 1
#                 else:
#                     play = 0
#     pg.display.flip()
#
#     tic = timeit.default_timer()
#
#     print(f'frame time: {1000*round(tic-toc, 8)} ms')
#
# pg.quit()
#
#
# fig, axs = plt.subplots(2, 1, sharex='col')
# fig.suptitle('Inverted pendulum :0000')
#
# axs[0].plot(t_list, x_list, label='x')
# axs[0].plot(t_list, x_tgt_list, label='x tgt')
# axs[0].legend(loc='upper right')
# axs[0].grid()
# axs[0].set_ylabel('[m]')
# axs[1].plot(t_list, theta_list, label='theta')
# axs[1].legend(loc='upper right')
# axs[1].grid()
# axs[1].set_ylabel('[rad]')
# plt.show()
#
