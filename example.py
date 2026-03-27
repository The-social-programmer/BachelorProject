"""
# A simple crystal structure search example with ASE, GPAW, and BoTorch

We follow the BEACON tutorial here for the problem setup:
https://gitlab.com/gpatom/ase-gpatom/-/wikis/How-to-use-BEACON
"""

import numpy as np
import ase
import ase.build
from ase.optimize import BFGS
from ase.spacegroup import crystal
from gpaw import GPAW
from upet.calculator import UPETCalculator
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf


################################################################################
# 1: The system and calculator
################################################################################
def make_calculator(kind="pet-mad-s"):
    if kind == "GPAW":
        return GPAW(
            mode="pw",
            txt="gpaw.txt",
            xc="LDA",
            # kpts=(4, 4, 4),
        )
    elif kind == "pet-mad-s":
        return UPETCalculator(
            model="pet-mad-s",
            version="1.5.0",
            device="cpu",
        )
    elif kind == "pet-oam-xl":
        return UPETCalculator(
            model="pet-oam-xl",
            version="1.0.0",
            device="cpu",
        )


# Some example systems to choose from
system = "TiO2"

if system == "Si2":
    atoms_true = ase.build.bulk("Si", "diamond", a=5.43)
elif system == "Si16":
    # Si16 in a fixed diamond cell (same setup as in the BEACON paper)
    atoms_true = ase.build.bulk("Si", "diamond", a=5.43) * (2, 2, 2)
elif system == "TiO2":
    # Rutile TiO2 (space group 136, 6 atoms)
    atoms_true = crystal(
        ["Ti", "O"],
        basis=[(0, 0, 0), (0.305, 0.305, 0)],
        spacegroup=136,
        cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
    )
elif system == "Cu2O":
    # Cuprite Cu2O (space group 224, 6 atoms)
    atoms_true = crystal(
        ["Cu", "O"],
        basis=[(0.25, 0.25, 0.25), (0, 0, 0)],
        spacegroup=224,
        cellpar=[4.2696, 4.2696, 4.2696, 90, 90, 90],
    )

cell = atoms_true.cell.copy()
symbols = atoms_true.get_chemical_symbols()
n_atoms = len(symbols)


def make_atoms(scaled_positions):
    atoms = ase.Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True,
    )
    atoms.calc = make_calculator()
    return atoms


################################################################################
# 2: The true ground state
################################################################################
print(f"\n=== True ground state ===")
atoms_true.calc = make_calculator()
dyn = BFGS(atoms_true)
dyn.run(fmax=0.001)
E_true = atoms_true.get_potential_energy()
print("Result:")
print(f"{n_atoms} Si atoms, energy: {E_true:.4f} eV ({E_true / n_atoms:.4f} eV/atom)")


################################################################################
# 3: Local relaxation from a random start (baseline)
################################################################################
np.random.seed(42)
scaled_positions_init = np.random.rand(n_atoms, 3)
scaled_positions_init[0] = 0.0  # pin first atom at origin

atoms_loc = make_atoms(scaled_positions_init)
print(f"\n=== Local relaxation from random start ===")
print(f"Initial energy: {atoms_loc.get_potential_energy():.4f} eV")
dyn = BFGS(atoms_loc)
dyn.run(fmax=0.001)
E_loc = atoms_loc.get_potential_energy()
print(f"Relaxed energy: {E_loc:.4f} eV ({E_loc / n_atoms:.4f} eV/atom)")
print(f"Gap to ground state: {E_loc - E_true:.4f} eV")


################################################################################
# 4: Bayesian Optimization
################################################################################
# We now implement a Bayesian optimization approach using botorch
def objective(x):
    # botorch works with torch but ASE needs numpy:
    x_np = x.detach().cpu().numpy()
    pos = x_np.reshape(-1, 3)
    scaled_positions = np.vstack([np.zeros((1, 3)), pos])  # pin first atom at origin

    atoms = make_atoms(scaled_positions)
    energy = atoms.get_potential_energy()

    return -torch.tensor(energy, dtype=x.dtype)


# We need to define the bounds for the optimization
n_pos = 3 * (n_atoms - 1)
bounds = torch.tensor(
    [
        [0.05] * n_pos,
        [0.95] * n_pos,
    ],
    dtype=torch.double,
)

# Initialization: Here we just start with a single initial guess
# THIS IS MOST LIKELY NOT IDEAL!
x_init = torch.tensor(np.array(scaled_positions_init[1:]).ravel(), dtype=torch.double)
# Let's try the objective function:
y_init = objective(x_init)
print(f"\n=== Bayesian optimization ===")
print(f"Objective (negative energy): {y_init.item()}")
print(f"(recall the initial energy from before: {E_loc:.4f} eV)")

# Create the data for the GP
train_X = x_init.unsqueeze(0)  # shape: (1, d)
train_Y = y_init.unsqueeze(0).unsqueeze(0)  # shape: (1, 1)
d = train_X[0].numel()

# The actual BO loop
for i in range(100):
    # i) fit a GP
    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)),
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # ii) define an acquisition function and optimize it
    logEI = LogExpectedImprovement(model=gp, best_f=train_Y.max())
    candidate, acq_value = optimize_acqf(
        logEI,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )

    # iii) evaluate the objective on the candidate and add to data
    new_y = objective(candidate[0])
    train_X = torch.cat([train_X, candidate], dim=0)
    train_Y = torch.cat([train_Y, new_y.unsqueeze(0).unsqueeze(0)], dim=0)

    print(
        f"Iter {i:3d} | E={-new_y.item():.4f} eV | best={-train_Y.max().item():.4f} eV"
    )

# Results
best_idx = train_Y.argmax()
print(f"\n=== Comparison ===")
print(f"Local relaxation (BFGS): {E_loc:.4f} eV")
print(f"Bayesian optimization:   {-train_Y[best_idx].item():.4f} eV")
print(f"Known ground state:      {E_true:.4f} eV")
