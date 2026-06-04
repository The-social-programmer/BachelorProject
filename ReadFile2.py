import matplotlib.pyplot as plt
import csv
import numpy as np

# reader for files mean_module_repulsion, UCBvsLOGEI, DFTvsPETMAD and MATTERNVSRBF

OPT_VALS = {
    "Cu2O": -31.458280563354492,
    "Si16": -94.24507904052734,
    "TiO2": -56.89365768432617,
    "Si2": -11.780633926391602,
}

def get_data(path):
    with open(path, mode="r", newline="") as file:
        data = csv.reader(file)
        index = 1
        material = ["Si2", "TiO2", "Si16", "Cu2O"]
        master_curve = []
        for line in data:
            # Empty line between between every system in csv file
            if len(line) == 0:
                
                index += 1
                master_curve = []
            try:
                float(line[0])
            except:
                continue

            for i in range(0, len(line)):
                line[i] = float(line[i])
            master_curve.append(line[0:100])
            
            if len(master_curve) == 10:
                system = material[index-1]
                opt_val = OPT_VALS[system]
                objective_values = np.array(master_curve)
                regret_values = objective_values - opt_val
                vals_to_plot = np.log(regret_values)
                plot_data(vals_to_plot, index, system)

def plot_data(data, index, system):
    #fig, axes = plt.subplots(2,2)
    mean = data.mean(axis=0)
    median = np.median(data, axis=0)
    qlow = np.quantile(data, 0, axis=0)
    qhigh = np.quantile(data, 1, axis=0)
    x = np.arange(len(mean))

    plt.subplot(2,2,index)
    plt.title(system)
    plt.fill_between(x, qlow, qhigh, color=color, alpha=0.2)
    plt.ylabel("log regret")
    plt.plot(median, color=color, label=label)
    plt.legend()
    
for j in range(0,2):
    if j == 0:
        color = "tab:orange"
        label = "Prior"
        data = get_data("/home/andres/BachelorProject/mean_module_repulsion.csv")
    else:
        color = "b"
        label = "No repulsion"
        data = get_data("/home/andres/BachelorProject/StandardBO.csv")

plt.tight_layout()
plt.show()
