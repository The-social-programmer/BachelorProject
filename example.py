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
from ase.calculators import lj
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import matplotlib.pyplot as plt
import csv
from gpytorch.means import Mean


################################################################################
# 1: The system and calculator
################################################################################
torch.set_default_dtype(torch.double)

def make_calculator(kind="pet-mad-s"):
    if kind == "GPAW":
        return GPAW(
            mode="pw",
            txt="gpaw.txt",
            xc="LDA",
            kpts=(4, 4, 4),
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
    elif kind == "LJ":
        return lj.LennardJones()


# Some example systems to choose from
system = "Cu2O"

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


kindCal = "petmad"
def make_atoms(scaled_positions):
    atoms = ase.Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True,
    )
    if kindCal == "petmad":
        atoms.calc = make_calculator()
    elif kindCal == "gpaw":
        atoms.calc = make_calculator("GPAW")
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
print(f"{n_atoms} Si atoms, energy: {E_true:.15f} eV ({E_true / n_atoms:.15f} eV/atom)")

################################################################################
# 3: Local relaxation from a random start (baseline)
################################################################################
np.random.seed(42)
scaled_positions_init = np.random.rand(n_atoms, 3) # atoms_true.get_scaled_positions()
scaled_positions_init[0] = 0.0  # pin first atom at origin
print(scaled_positions_init)

atoms_loc = make_atoms(scaled_positions_init)
print(f"\n=== Local relaxation from random start ===")
print(f"Initial energy: {atoms_loc.get_potential_energy():.4f} eV")

"""
energies = []

# Callback function
def save_energy():
    energy = atoms_loc.get_potential_energy()
    energies.append(energy)

# Create optimizer
dyn = BFGS(atoms_loc)

# Save energy every iteration
dyn.attach(save_energy, interval=1)

# Run optimization
dyn.run(fmax=0.001)

# Convert to NumPy array
energies = np.array(energies)


with open("DFTvsPETMAD.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["True energy"])
    writer.writerow([E_true])
    writer.writerow(["BFGS"])
    writer.writerow(energies)
    
plt.plot(energies)

"""
E_loc = atoms_loc.get_potential_energy()
print(f"Relaxed energy: {E_loc:.4f} eV ({E_loc / n_atoms:.4f} eV/atom)")
print(f"Gap to ground state: {E_loc - E_true:.4f} eV")


################################################################################
# 4: Bayesian Optimization
################################################################################
# We now implement a Bayesian optimization approach using botorch
def overlap_penalty(positions):
    penalty = 0.0
    n = len(positions)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 1.5 and d != 0:
                penalty += (1.5 / d) ** 12
    if penalty > 1000000:
        return 1000000
    else:
        return penalty

class RepulsiveMean(Mean):

    def __init__(self, cell, symbols, min_dist):
        super().__init__()

        self.cell = cell
        self.symbols = symbols
        self.min_dist = min_dist

    def forward(self, x):

        original_shape = x.shape[:-1]

        # flatten all batch dimensions
        x_flat = x.reshape(-1, x.shape[-1])
        means = []

        for row in x_flat:
            x_np = row.detach().cpu().numpy()
            pos = x_np.reshape(-1, 3)
            scaled_positions = np.vstack([np.zeros((1, 3)),pos])

            atoms = ase.Atoms(
                symbols=self.symbols,
                scaled_positions=scaled_positions,
                cell=self.cell,
                pbc=True,
            )
            distances = atoms.get_all_distances(mic=True)
            repulsion = 0.0

            n_atoms = len(atoms)
        
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    d = distances[i, j]
                    d = max(d, 0.5)

                    if d < self.min_dist:
                        repulsion += (self.min_dist / d) ** 12

            means.append(-repulsion)
        
        means = torch.tensor(
            means,
            dtype=x.dtype,
            device=x.device,
        )
        means = means.reshape(*original_shape)
        return means

def objective(x):
    # botorch works with torch but ASE needs numpy:
    
    x_np = x.detach().cpu().numpy()
    pos = x_np.reshape(-1, 3)
    scaled_positions = np.vstack([np.zeros((1, 3)), pos])  # pin first atom at origin
    
    atoms = make_atoms(scaled_positions)
    positions = atoms.get_positions()

    E_penalty = overlap_penalty(positions)

    energy = atoms.get_potential_energy()
    tot_energy = energy #+ E_penalty

    #Only the tot_energy is negative as we want to pass it to BO which works with maximization
    return torch.tensor([-tot_energy, energy], dtype=x.dtype)


# We need to define the bounds for the optimization
n_pos = 3 * (n_atoms - 1)
bounds = torch.tensor(
    [
        [0.00] * n_pos,
        [1.00] * n_pos,
    ],
    dtype=torch.double,
)

# Initialization: Here we just start with a single initial guess
# THIS IS MOST LIKELY NOT IDEAL!
for j in range(6,10):
    np.random.seed(42 + j)
    scaled_positions_init = np.random.rand(n_atoms, 3) # atoms_true.get_scaled_positions()
    scaled_positions_init[0] = 0.0  # pin first atom at origin
    x_init = torch.tensor(np.array(scaled_positions_init[1:]).ravel(), dtype=torch.double)

    # Let's try the objective function:
    #if j == 1:
    #    kindCal = "gpaw"
    
    obj = objective(x_init)
    y_init = obj[0]
    print(f"\n=== Bayesian optimization ===")
    print(f"Objective (negative energy): {y_init.item()}")
    print(f"(recall the initial energy from before: {E_loc:.4f} eV)")

    # Create the data for the GP
    train_X = x_init.unsqueeze(0)  # shape: (1, d)
    train_Y = y_init.unsqueeze(0).unsqueeze(0)  # shape: (1, 1)
    d = train_X[0].numel()

    # The actual BO loop
    iterations = 100
    y_points = np.array([obj[1]])
    minimum = obj[1]

    for i in range(iterations):
    # i) fit a GP
        gp = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            mean_module=RepulsiveMean(cell=cell, symbols=symbols, min_dist=1.5),
            covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)),
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # ii) define an acquisition function and optimize it
        #UCB = UpperConfidenceBound(model=gp, beta=2.0)
        logEI = LogExpectedImprovement(model=gp, best_f=train_Y.max())
        candidate, acq_value = optimize_acqf(
            logEI,
            bounds=bounds,
            q=1, # defines the size of candidate (q,d)
            num_restarts=5,
            raw_samples=20,
        )

        # iii) evaluate the objective on the candidate and add to data
        obj = objective(candidate[0])
        new_y = obj[0]
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_y.unsqueeze(0).unsqueeze(0)], dim=0)
        if obj[1] < minimum:
            a = np.append(y_points, obj[1])
            minimum = obj[1]
        else:
            a = np.append(y_points, minimum)
        y_points = a

        print(
            f"Iter {i:3d} | E={-new_y.item():.4f} eV | best={-train_Y.max().item():.4f} eV"
        )
        
    with open("mean_module_repulsion.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Seed", 42+j])
        writer.writerow(["BO"])
        writer.writerow(y_points)
        
    plt.plot(y_points)


# Results
best_idx = train_Y.argmax()
print(f"\n=== Comparison ===")
print(f"Local relaxation (BFGS): {E_loc:.4f} eV")
print(f"Bayesian optimization:   {-train_Y[best_idx].item():.4f} eV")
print(f"Known ground state:      {E_true:.4f} eV")

plt.xlabel("Iteration")
plt.ylabel("Energy [eV]")
plt.show()
