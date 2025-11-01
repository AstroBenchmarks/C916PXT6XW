import subprocess
import time
import pyvista as pv
import json
import h5py
import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt

start_time = time.time()
subprocess.run(
    ["../athenak/_build/src/athena", "-i", "orszag_tang.athinput"], check=True
)
end_time = time.time()

runtime = end_time - start_time

print(f"Execution time: {runtime:.2f} seconds")


# read OrszagTang.mhd_w.00001.vtk
snapshot = pv.read("vtk/OrszagTang.mhd_w.00001.vtk")


rho = snapshot["dens"].reshape((512, 512)).T

plt.imshow(rho)
plt.colorbar()
plt.show()

num_cycles = input("Enter number of cycles (default=1640): ")
if num_cycles == "":
    num_cycles = 1640
else:
    num_cycles = int(num_cycles)
print(f"Number of cycles: {num_cycles}")


# Make JSON file and hdf5 data:

# template
# {
#  "version": "version or commit_hash",
#  "date": "20XX-XX-XX XX:XX:XX+00:00",
#  "runtime_(seconds)": 0.0,
#  "n_timesteps": 100,
#  "setup": "https://<link-to-setup>"
# }

# Get the git version of the repository
version = (
    subprocess.check_output(["git", "-C", "../athenak", "rev-parse", "HEAD"])
    .strip()
    .decode("utf-8")
)
version = version[:7]  # short version

# Prepare the data for the JSON file
metadata = {
    "version": version,
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
    "runtime_(seconds)": round(runtime, 1),
    "n_timesteps": num_cycles,
    "setup": "https://github.com/AstroBenchmarks/C916PXT6XW/tree/main/athenak/orszag-tang",
}

# Create directory if it doesn't exist
os.makedirs(version, exist_ok=True)

# Write to JSON file
with open(f"{version}/result.json", "w") as f:
    json.dump(metadata, f, indent=2)

# save "rho" into data.h5

with h5py.File(f"{version}/data.h5", "w") as f:
    f.create_dataset("rho", data=rho)
