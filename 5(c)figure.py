import pickle
import matplotlib.pyplot as plt


def load_variable(filename):
	f = open('.\\' + filename + '.txt', 'rb')
	data = pickle.load(f)
	f.close()
	return data


# load data
mc_naive = load_variable('MC_naive')
mc_complex = load_variable('MC_complex')
q_naive = load_variable('Q_naive')
q_complex = load_variable('Q_complex')

# plot figure 1
plt.figure(1)
plt.subplot(411)
plt.plot(mc_naive, color='green', label='Monte Carlo / reward structure 1')
plt.legend()
plt.subplot(412)
plt.plot(mc_complex, color='blue', label='Monte Carlo / reward structure 2')
plt.legend()
plt.subplot(413)
plt.plot(q_naive, color='red', label='Q Learning / reward structure 1')
plt.legend()
plt.subplot(414)
plt.plot(q_complex, color='black', label='Q Learning / reward structure 2')
plt.legend()
plt.show()

# plot figure 2
plt.figure(2)
plt.plot(mc_naive, color='green', label='Monte Carlo / reward structure 1')
plt.plot(mc_complex, color='blue', label='Monte Carlo / reward structure 2')
plt.plot(q_naive, color='red', label='Q Learning / reward structure 1')
plt.plot(q_complex, color='black', label='Q Learning / reward structure 2')
plt.legend()
plt.title('The Total Return in Each Episode')
plt.xlabel('number of episode')
plt.ylabel('total return')
plt.show()
