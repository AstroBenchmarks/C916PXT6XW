# numerics
import jax
import jax.numpy as jnp

import adirondax as adx
from adirondax.hydro.common2d import get_curl, get_avg

# utilities
import json
import h5py
from datetime import datetime, timezone
import subprocess
import time
import os


def set_up_simulation():
    # Define the parameters for the simulation
    n = 512
    nt = 100 * int(n / 32)
    t_stop = 0.5
    dt = t_stop / nt
    gamma = 5.0 / 3.0
    box_size = 1.0
    dx = box_size / n

    params = {
        "physics": {
            "hydro": True,
            "magnetic": True,
            "quantum": False,
            "gravity": False,
        },
        "mesh": {
            "type": "cartesian",
            "resolution": [n, n],
            "boxsize": [box_size, box_size],
        },
        "simulation": {
            "stop_time": t_stop,
            "timestep": dt,
            "n_timestep": nt,
        },
        "hydro": {
            "eos": {"type": "ideal", "gamma": gamma},
        },
    }

    # Initialize the simulation
    sim = adx.Simulation(params)

    # Set initial conditions
    sim.state["t"] = jnp.array(0.0)
    X, Y = sim.mesh
    sim.state["rho"] = (gamma**2 / (4.0 * jnp.pi)) * jnp.ones(X.shape)
    sim.state["vx"] = -jnp.sin(2.0 * jnp.pi * Y)
    sim.state["vy"] = jnp.sin(2.0 * jnp.pi * X)
    P_gas = (gamma / (4.0 * jnp.pi)) * jnp.ones(X.shape)
    # (Az is at top-right node of each cell)
    xlin_node = jnp.linspace(dx, box_size, n)
    Xn, Yn = jnp.meshgrid(xlin_node, xlin_node, indexing="ij")
    Az = jnp.cos(4.0 * jnp.pi * Xn) / (4.0 * jnp.pi * jnp.sqrt(4.0 * jnp.pi)) + jnp.cos(
        2.0 * jnp.pi * Yn
    ) / (2.0 * jnp.pi * jnp.sqrt(4.0 * jnp.pi))
    bx, by = get_curl(Az, dx)
    Bx, By = get_avg(bx, by)
    P_tot = P_gas + 0.5 * (Bx**2 + By**2)
    sim.state["P"] = P_tot
    sim.state["bx"] = bx
    sim.state["by"] = by

    return sim


# Run the simulation::
print("Starting the simulation...")
sim = set_up_simulation()

start_time = time.time()
sim.run()
jax.block_until_ready(sim.state["rho"])
runtime = time.time() - start_time

print("Simulation completed in {:.2f} seconds.".format(runtime))

rho = sim.state["rho"]


# Make JSON file and hdf5 data:

# template
# {
#  "version": "version or commit_hash",
#  "date": "20XX-XX-XX XX:XX:XX+00:00",
#  "runtime_(seconds)": 0.0,
#  "n_timesteps": 100,
#  "setup": "https://<link-to-setup>"
# }

# Get version info (you may need to adjust this based on your package setup)
try:
    version = subprocess.check_output(["pip", "show", "adirondax"]).decode().split("\n")
    version = next(
        line.split(": ")[1] for line in version if line.startswith("Version:")
    )
except Exception:
    version = "unknown"

# Create metadata dictionary
metadata = {
    "version": version,
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
    "runtime_(seconds)": round(runtime, 1),
    "n_timesteps": int(sim._nt),
    "setup": "https://github.com/AstroBenchmarks/C916PXT6XW/tree/main/adirondax/orszag-tang",
}

# Create directory if it doesn't exist
os.makedirs(version, exist_ok=True)

# Write to JSON file
with open(f"{version}/result.json", "w") as f:
    json.dump(metadata, f, indent=2)

# save "rho" into data.h5

with h5py.File(f"{version}/data.h5", "w") as f:
    f.create_dataset("rho", data=rho)
