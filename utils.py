import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_results(results, initialization=""):
    results = pd.DataFrame(results[:-4], columns=["Iteration", "Reduced Cost", "MultiTour", "Cycle Time", "Total Cost"])
    results['Iteration'] += 1
    ax = results.plot(x="Iteration", y="Reduced Cost", marker='s', color='black', linestyle='-', label='Reduced Cost', linewidth=0.9, fillstyle='none')
    results.plot(x="Iteration", y="Total Cost", marker='o', color='black', linestyle='--', label='Total Cost', ax=ax, linewidth=0.9)

    # Add a horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.5)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.set_title('Reduced Cost and Total Cost'+initialization)

    plt.show()