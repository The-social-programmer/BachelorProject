import matplotlib.pyplot as plt
import csv

with open("output_pen.csv", mode="r", newline="") as file:
    data = csv.reader(file)
    color = "b"
    label = "nothing"
    index = 1
    firstTime = 0
    compare = "BFGS"
    material = ["Si2", "TiO2", "Si16", "Cu2O"]
    for line in data:
        if len(line) == 0:
            plt.legend()
            firstTime = 0
            index += 1
        try:
            if line[0] == compare:
                color = "tab:orange"
                label = compare
            elif line[0] == "BO":
                color = "b"
                label = "BO"
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
                    if firstTime < 2:
                        plt.subplot(2,2,index)
                        plt.title(material[index-1])
                        plt.plot(line, color=color, label=label, linestyle = "dashed")
                        firstTime += 1
                else:
                    plt.subplot(2,2,index)
                    plt.title(material[index-1])
                    if firstTime > 2:
                        plt.plot(line, color=color)
                    else:
                        plt.plot(line, color=color, label=label)
                        firstTime += 1
        except:
            pass

plt.legend()
plt.show()
