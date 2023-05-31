import pandas as pd
import matplotlib.pyplot as plt

input_file = "../out/output.csv"
data = pd.read_csv(input_file)

plt.plot(data["Vertices"], data["SequentialTime"], label="Sequential")
plt.plot(data["Vertices"], data["OpenMpTime"], label="OpenMP")
plt.plot(data["Vertices"], data["CudaTime"], label="CUDA")

plt.title("Czas obliczeń dla różnych algorytmów")
plt.xlabel("Vertices")
plt.ylabel("Czas obliczeń [ms]")

plt.legend()
# plt.show()
plt.savefig("wykres.png")

