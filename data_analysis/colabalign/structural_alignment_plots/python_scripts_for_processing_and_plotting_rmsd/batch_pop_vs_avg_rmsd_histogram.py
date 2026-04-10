import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class Population_vs_RMSD_Histogram():
    def __init__(self, data):
        self.data = data
        self.x = self.data["RMSD"]
        self.y = self.data["Atoms_Compared"]

    def draw(self):
        #num_points = len(self.x)
        plt.figure(figsize=(10, 8))
        plt.hist(self.x, bins=40, color = 'skyblue', edgecolor = 'black', alpha = 0.7)
        #n, bin_edges, patches = plt.hist(self.x, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Sample Frequency')

        # Calculate weighted mean and standard deviation for drawing normal distribution curve
        #mu = np.average(self.x, weights=self.y)
        #sigma = np.sqrt(np.average((self.x - mu)**2, weights=self.y))

        # Create the normal distribution curve
        #x_range = np.linspace(min(self.x), max(self.x), 1000)
        #normal_dist = norm.pdf(x_range, mu, sigma)

        #bin_width = bin_edges[1] - bin_edges[0]
        #scaled_p = normal_dist * self.y.sum() * bin_width

        #plt.plot(x_range, scaled_p, color='red', linewidth=3, linestyle='dotted',label=f'Normal Fit\n$\mu={mu:.2f}$\n$\sigma={sigma:.2f}$')
        
        plt.title('Frequency Distribution of the Average RMSD Values for 80 Selected Proteins', fontsize=16)
        plt.xlabel("Average RMSD / $\mathrm{\AA}$", fontsize=14)
        plt.ylabel("Frequency (Number of Samples)", fontsize=14)
        #plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"Population_vs_Avg_RMSD_Histogram.png", dpi = 500)

data_file = "batch_rmsd_summary_new.csv"
data = pd.read_csv(data_file)
population_vs_rmsd_histogram = Population_vs_RMSD_Histogram(data)
print(data.columns)
population_vs_rmsd_histogram.draw()

        



    

