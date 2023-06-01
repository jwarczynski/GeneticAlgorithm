import os
import subprocess
import csv
import pandas as pd

input_folder = "../input"
input_folder = "input"
bin_folder = "../bin"
bin_folder = "bin"
output_file = "../out/output2.csv"
output_file = "out/output2.csv"

files = os.listdir(input_folder)
sorted_files = sorted(files, key=int)
print(sorted_files)

data = pd.DataFrame(index=files, columns=["OpenMpTime", "SequentialTime", "CudaTime", "OpenMpRes", "SequentialRes", "CudaRes"])

with open(output_file, mode='w', newline='') as output:
    writer = csv.writer(output)
    headers = ["Vertices", "CudaTime", "OpenMpTime", "SequentialTime", "CudaRes", "OpenMpRes", "SequentialRes"]
    writer.writerow(headers)
    for file in sorted_files:
        if (int(file) < 100):
            continue
        print(f'filename: {file}')
        main_output = subprocess.check_output([os.path.join(bin_folder, "main"), os.path.join(input_folder, file), "40"])
        main_output = main_output.decode().strip().split("\n")
        
        row = [file, main_output[0].split()[0], main_output[1].split()[0], main_output[2], main_output[0].split()[1], main_output[1].split()[1], main_output[1]]
        writer.writerow(row)
        print("results: ", row)
