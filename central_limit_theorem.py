import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.animation import FuncAnimation

class TrialDataGenerator:
    def __init__(self, number_of_trials=100):
        self.number_of_trials = number_of_trials
        self.fig = plt.figure(figsize=(15, 10))

        # Define subplots for different visualizations
        self.ax1 = plt.subplot2grid((2, 3), (0, 0))
        self.ax2 = plt.subplot2grid((2, 3), (0, 1))
        self.ax4 = plt.subplot2grid((2, 3), (1, 1))  
        self.ax3 = plt.subplot2grid((2, 3), (1, 0))             
        self.ax5 = plt.subplot2grid((2, 3), (0, 2,),colspan=2)              

        self.trials = np.random.randint(1, 7, (self.number_of_trials, 7))
        self.means = []
        self.p_values = []

        self.anim = FuncAnimation(self.fig, self.update_all, frames=np.arange(self.number_of_trials),
                                  init_func=self.init_animation, interval=500, repeat=False)

    def init_animation(self):
        pass

    def update_all(self, frame_number):
        self.trials[frame_number] = np.random.randint(1, 7, 7)
        self.update_histogram()
        self.update_qq_plot()
        self.update_shapiro_wilk_test(self.means)
        self.update_distribution()

    def update_histogram(self):
        self.ax1.clear()
        self.means = np.mean(self.trials[:len(self.means) + 1], axis=1)
        self.ax1.hist(self.means, bins=20, alpha=0.7, color='blue', density=True)
        self.ax1.set_title('Histogram of Means')
        self.ax1.set_xlabel('Mean')
        self.ax1.set_ylabel('Density')
        self.ax1.set_xlim(1, 6)
        self.ax1.set_ylim(0, 0.6)
        self.ax1.text(0.5, 0.95, f'Trials: {len(self.means)}', fontsize=10, ha='center', transform=self.ax1.transAxes)

    def update_qq_plot(self):
        self.ax2.clear()
        stats.probplot(self.means, plot=self.ax2)
        self.ax2.set_title('QQ Plot (Quantile-Quantile plot)')

    def update_shapiro_wilk_test(self, means):
        if len(means) >= 3:
            w_stat, p_value = stats.shapiro(means)
            self.p_values.append(p_value)
            skewness = stats.skew(means)
            kurtosis = stats.kurtosis(means)
            self.ax3.clear()
            self.ax3.text(0.5, 0.7, f'p-value: {p_value:.4f}\nW-statistic: {w_stat:.4f}\nSkewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}',
                          fontsize=12, ha='center', va='center', transform=self.ax3.transAxes)
            self.ax3.axis('off')
        else:
            self.ax3.clear()
            self.ax3.axis('off')

        self.ax4.clear()
        if len(means) >= 3:
            self.ax4.plot(range(len(self.p_values)), self.p_values, color='red')
            self.ax4.set_title('Historical Shapiro-Wilk p-values')
            self.ax4.set_xlabel('Trials')
            self.ax4.set_ylabel('p-value')

    def update_distribution(self):
        self.ax5.clear()
        self.ax5.hist(self.trials[:len(self.means) + 1].flatten(), bins=6, range=(0.5, 7.5), alpha=0.7, color='blue', rwidth=0.95)
        self.ax5.set_xticks(np.arange(1, 8))
        self.ax5.set_title('Distribution of Outputs (Original Dice Rolls)')

generator = TrialDataGenerator(number_of_trials=100)
plt.show()