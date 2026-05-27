import matplotlib.pyplot as plt
import csv
import numpy as np

# reader for files mean_module_repulsion, UCBvsLOGEI, DFTvsPETMAD and kernel

for j in range(0,2):
    if j == 0:
        path = "/home/andres/BachelorProject/UCBvsLOGEI.csv"
    else:
        path = "/home/andres/BachelorProject/StandardBO.csv"
    with open(path, mode="r", newline="") as file:
        data = csv.reader(file)
        color = "b"
        label = "nothing"
        index = 1
        firstTime = 0
        compare = "UCB" # Change here
        material = ["Si2", "TiO2", "Si16", "Cu2O"]
        masterCurve = []
        for line in data:
            if len(line) == 0:
                plt.legend()
                firstTime = 0
                index += 1
                masterCurve = []
            try:
                if line[0] == compare:
                    color = "tab:orange"
                    label = compare
                elif line[0] == "BO":
                    color = "b"
                    label = "LOGEI" # Change here
                elif line[0] == "True energy":
                    color = "0"
                    label = "True energy"

                x = float(line[0])
                if isinstance(x, float) : #and len(line) != 1
                    for i in range(0, len(line)):
                        line[i] = float(line[i])
                    if len(line) == 1:
                        for i in range(0, 100):
                            line.append(line[0])
                        if firstTime < 1 and j == 1:
                            plt.subplot(2,2,index)
                            plt.title(material[index-1])
                            plt.plot(line, color=color, label=label, linestyle = "dashed")
                            firstTime += 1
                    else:
                        masterCurve.append(line[0:100])
                        if len(masterCurve) == 10:
                            mean = np.array(masterCurve).mean(axis=0)
                            std = np.array(masterCurve).std(axis=0)
                            x = np.arange(len(mean))
                            plt.subplot(2,2,index)
                            plt.title(material[index-1])
                            plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
                            plt.plot(mean, color=color, label=label)
                        
            except:
                pass

plt.legend()
plt.show()
