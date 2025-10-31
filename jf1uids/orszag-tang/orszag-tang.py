# numerics
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# fluids
from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import finalize_config

# utilities
import json
import h5py
from datetime import datetime, timezone
import subprocess
import time
import os

# Initialization::

from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    FORWARDS,
    HLL,
    HLLC,
    MINMOD,
    OSHER,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

print("👷 Setting up simulation...")

# simulation settings
gamma = 5 / 3

# spatial domain
box_size = 1.0
num_cells = 512

fixed_timestep = False
scale_time = False
dt_max = 0.1
num_timesteps = 10

# setup simulation config
config = SimulationConfig(
    runtime_debugging=False,
    progress_bar=True,
    first_order_fallback=False,
    return_snapshots=True,
    use_specific_snapshot_timepoints=True,
    num_snapshots=1,
    mhd=True,
    dimensionality=2,
    box_size=box_size,
    num_cells=num_cells,
    fixed_timestep=fixed_timestep,
    differentiation_mode=FORWARDS,
    num_timesteps=num_timesteps,
    limiter=MINMOD,
    riemann_solver=HLL,
    boundary_settings=BoundarySettings(
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
    ),
)

helper_data = get_helper_data(config)

params = SimulationParams(t_end=0.5, C_cfl=0.4, snapshot_timepoints=jnp.array([0.5]))

registered_variables = get_registered_variables(config)


# Setting the initial state::

# Grid size and configuration
num_cells = config.num_cells
x = jnp.linspace(0, box_size, num_cells)
y = jnp.linspace(0, box_size, num_cells)
X, Y = jnp.meshgrid(x, y, indexing="ij")

# Initialize state
rho = jnp.ones_like(X) * gamma**2 / (4 * jnp.pi)
P = jnp.ones_like(X) * gamma / (4 * jnp.pi)

V_x = -jnp.sin(2 * jnp.pi * Y)
V_y = jnp.sin(2 * jnp.pi * X)

B_0 = 1 / jnp.sqrt(4 * jnp.pi)
B_x = -B_0 * jnp.sin(2 * jnp.pi * Y)
B_y = B_0 * jnp.sin(4 * jnp.pi * X)
# B_x = jnp.zeros_like(X)
# B_y = jnp.zeros_like(X)
B_z = jnp.zeros_like(X)

initial_magnetic_field = jnp.stack([B_x, B_y, B_z], axis=0)

dx = 1 / (num_cells - 1)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=V_x,
    velocity_y=V_y,
    magnetic_field_x=B_x,
    magnetic_field_y=B_y,
    magnetic_field_z=B_z,
    gas_pressure=P,
)

config = finalize_config(config, initial_state.shape)


# Run the simulation::
print("Starting the simulation...")

start_time = time.time()
snapshots = time_integration(
    initial_state, config, params, helper_data, registered_variables
)
jax.block_until_ready(snapshots)
runtime = time.time() - start_time

print("Simulation completed in {:.2f} seconds.".format(runtime))

final_state = snapshots.states[-1]

rho = final_state[0, ...]


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
    version = subprocess.check_output(["pip", "show", "jf1uids"]).decode().split("\n")
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
    "n_timesteps": int(snapshots.num_iterations),
    "setup": "https://github.com/AstroBenchmarks/C916PXT6XW/tree/main/jf1uids/orszag-tang",
}

# Create directory if it doesn't exist
os.makedirs(version, exist_ok=True)

# Write to JSON file
with open(f"{version}/result.json", "w") as f:
    json.dump(metadata, f, indent=2)

# save "rho" into data.h5

with h5py.File(f"{version}/data.h5", "w") as f:
    f.create_dataset("rho", data=rho)
