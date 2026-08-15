## Download the data
Download and save the configuration files in a directory named `ensemble`.

## Run the benchmark
Create a file `benchmark.py` with the content
```python
from pyquda_utils import core, io

ensemble = {
    "C24P29": {
        "Ls": 24,
        "Lt": 72,
        "beta": 6.20,
        "mass_l": -0.2770,
        "mass_s": -0.2400,
        "cfg": 48000,
        "clover_coeff": 1.160920226,
        "tol": 1e-12,
        "maxiter": 1000,
        "multigrid": [[6, 6, 6, 4], [4, 4, 4, 9]],
    },
    "C32P29": {
        "Ls": 32,
        "Lt": 64,
        "beta": 6.20,
        "mass_l": -0.2770,
        "mass_s": -0.2400,
        "cfg": 38000,
        "clover_coeff": 1.160920226,
        "tol": 1e-12,
        "maxiter": 1000,
        "multigrid": [[4, 4, 4, 4], [2, 2, 2, 2], [4, 4, 4, 4]],
    },
    "C48P14": {
        "Ls": 48,
        "Lt": 96,
        "beta": 6.20,
        "mass_l": -0.2825,
        "mass_s": -0.2310,
        "cfg": 3000,
        "clover_coeff": 1.160587196,
        "tol": 1e-12,
        "maxiter": 1000,
        "multigrid": [[6, 6, 6, 3], [2, 2, 2, 4], [4, 4, 4, 4]],
    },
    "F32P30": {
        "Ls": 32,
        "Lt": 96,
        "beta": 6.41,
        "mass_l": -0.2295,
        "mass_s": -0.2050,
        "cfg": 9000,
        "clover_coeff": 1.141151096,
        "tol": 1e-12,
        "maxiter": 1000,
        "multigrid": [[4, 4, 4, 6], [2, 2, 2, 2], [4, 4, 4, 4]],
    },
    "H48P32": {
        "Ls": 48,
        "Lt": 144,
        "beta": 6.41,
        "mass_l": -0.1850,
        "mass_s": -0.1700,
        "cfg": 3640,
        "clover_coeff": 1.11927241,
        "tol": 1e-12,
        "maxiter": 1000,
        "multigrid": [[6, 6, 6, 4], [2, 2, 2, 3], [4, 4, 4, 6]],
    },
}


def benchmark(name, cuda, sloppy, precondition):
    info = ensemble[name]
    Ls = info["Ls"]
    Lt = info["Lt"]
    beta = info["beta"]
    mu = info["mass_l"]
    ms = info["mass_s"]
    cfg = info["cfg"]

    latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)

    gauge = io.readChromaQIOGauge(
        f"./ensemble/{name}/beta{beta:.02f}_mu{mu:.04f}_ms{ms:.04f}_L{Ls}x{Lt}_cfg_{cfg}.lime"
    )
    gauge.stoutSmear(1, 0.125, 4)

    dirac_l = core.getDirac(
        latt_info=latt_info,
        mass=info["mass_l"],
        tol=info["tol"],
        maxiter=info["maxiter"] // 10,
        clover_coeff_t=info["clover_coeff"],
        multigrid=info["multigrid"],
    )
    dirac_l.setPrecision(cuda=cuda, sloppy=sloppy, precondition=precondition)
    dirac_l.loadGauge(gauge)
    core.invert(dirac_l, "wall", 0)
    dirac_l.destroy()

    dirac_s = core.getDirac(
        latt_info=latt_info,
        mass=info["mass_s"],
        tol=info["tol"],
        maxiter=info["maxiter"],
        clover_coeff_t=info["clover_coeff"],
    )
    dirac_s.setPrecision(cuda=cuda, sloppy=sloppy, precondition=precondition)
    dirac_s.loadGauge(gauge)
    core.invert(dirac_s, "wall", 0)
    dirac_s.destroy()


benchmark("C24P29", 8, 2, 2)
```

Then execute the command to run the benchmark. Here we need to create a directory `.cache` to save the cached tuning parameters.
```bash
mkdir -p .cache
mpiexec -n 2 python3 -m pyquda -g 1 1 1 2 -p .cache benchmark.py
```
Here we use two GPUs to run the benchmark. Numbers after `-g` indicate how to split the lattice. The product of the four numbers should be equal to the number of processes. The benchmark requires the configuration file in the `ensemble` directory, and you should choose a downloaded `*.lime` file to fill the first parameter of the `benchmark` function. The following parameters are precisions of different levels.