import os
import subprocess
import csv
import pandas as pd

input_folder = "../input"
bin_folder = "../bin"
output_file = "../out/output.csv"

files = os.listdir(input_folder)
print(files)

data = pd.DataFrame(index=files, columns=["OpenMpTime", "SequentialTime", "CudaTime", "OpenMpRes", "SequentialRes", "CudaRes"])

with open(output_file, mode='w', newline='') as output:
    writer = csv.writer(output)
    headers = ["Vertices", "SequentialTime", "OpenMpTime", "CudaTime", "SequentialRes", "OpenMpRes", "CudaRes"]
    writer.writerow(headers)
    for file in reversed(files):
        # Wywołanie pliku main
        main_output = subprocess.check_output([os.path.join(bin_folder, "main"), os.path.join(input_folder, file), "20"])
        main_output = main_output.decode().strip().split("\n")
        
        # Wywołanie pliku cuda
        cuda_output = subprocess.check_output([os.path.join(bin_folder, "cuda"), os.path.join(input_folder, file), "20"])
        cuda_output = cuda_output.decode().strip().split()

        row = [file, main_output[1].split()[0], main_output[0].split()[0], cuda_output[0], main_output[1].split()[1], main_output[0].split()[1], cuda_output[1]]
        writer.writerow(row)
        print(row)
