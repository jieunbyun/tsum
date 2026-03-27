# CLAUDE.md — TSUM Project

## DC-OPF Blackout Model Integration

The **adaptMCS-benchmarks** project at `/mnt/c/Projects/adaptMCS-benchmarks` provides a DC-OPF (DC Optimal Power Flow) blackout model that can be used as a TSUM system function (`sfun`).

### Key files in adaptMCS-benchmarks

| File | Purpose |
|------|---------|
| `DC-opf model/func_dcopt_py.py` | Pure Python DC-OPF solver (scipy linprog, no MATLAB dependency) |
| `DC-opf model/sfun_dcopt.py` | TSUM-compatible `sfun` wrapper — `make_dcopt_sfun()` |
| `DC-opf model/matpower2tsum.py` | Converts MATPOWER `.m` case files to TSUM input (nodes.json, edges.json, probs.json) |
| `DC-opf model/case14_tsum/` | Pre-generated TSUM input for IEEE 14-bus (branch failures only, 20 binary components) |
| `DC-opf model/case14_tsum_bus/` | Pre-generated TSUM input for IEEE 14-bus (branch + bus failures, 34 components, 5 multi-state generators) |

### sfun interface

```python
from sfun_dcopt import make_dcopt_sfun

sfun = make_dcopt_sfun(
    case_path='/mnt/c/Projects/matpower8.1/data/case14.m',
    blackout_threshold=54.8,  # % load shed that defines system failure
    alpha=2.0,                # branch capacity scaling
)

# comps_st: Dict[str, int] mapping component IDs to state indices
# Returns: (blackout_size_percent, sys_st, None)
#   sys_st = 1 if blackout < threshold, 0 otherwise
fval, sys_st, _ = sfun(comps_st)
```

### Component naming convention

- **Branches**: `br1`, `br2`, ..., `br{nl}` — binary states: 0=failed, 1=operational
- **Buses** (when `--include_bus_failures`): `vbus{id}` — modeled as virtual edges
  - Generator buses: 4-state (0=removed, 1=40% cap, 2=80% cap, 3=full)
  - Ordinary buses: binary (0=failed, 1=operational)

### Running with TSUM

```python
import sys, json, torch
sys.path.insert(0, '/mnt/c/Projects/adaptMCS-benchmarks/DC-opf model')

from sfun_dcopt import make_dcopt_sfun
from tsum import tsum

# Load TSUM inputs
data_dir = '/mnt/c/Projects/adaptMCS-benchmarks/DC-opf model/case14_tsum'
edges = json.loads(open(f'{data_dir}/edges.json').read())
probs_dict = json.loads(open(f'{data_dir}/probs.json').read())

# Setup sfun
sfun = make_dcopt_sfun(case_path='/mnt/c/Projects/matpower8.1/data/case14.m')

# Setup TSUM inputs
row_names = list(edges.keys())  # component names matching probs.json keys
n_state = 2  # binary for branch-only; use max(len(v) for v in probs_dict.values()) for mixed
probs = [[probs_dict[n][str(s)]['p'] for s in range(n_state)] for n in row_names]
probs = torch.tensor(probs, dtype=torch.float32)

# Run rule extraction
result = tsum.run_rule_extraction_by_mcs(
    sfun=sfun,
    probs=probs,
    row_names=row_names,
    n_state=n_state,
    sys_surv_st=1,
    unk_prob_thres=1e-2,
    output_dir='tsum_res_case14',
)
```

### For bus failures (mixed multi-state)

When using `case14_tsum_bus/`, components have different numbers of states (generators have 4, others have 2). `n_state` should be set to the maximum (4), and the probs tensor must be padded:

```python
data_dir = '/mnt/c/Projects/adaptMCS-benchmarks/DC-opf model/case14_tsum_bus'
probs_dict = json.loads(open(f'{data_dir}/probs.json').read())
row_names = list(probs_dict.keys())
n_state = max(len(v) for v in probs_dict.values())  # 4

probs = []
for name in row_names:
    p = probs_dict[name]
    row = [p[str(s)]['p'] if str(s) in p else 0.0 for s in range(n_state)]
    probs.append(row)
probs = torch.tensor(probs, dtype=torch.float32)
```

### Performance

- Pure Python DC-OPF: ~1.6ms per call (case14)
- No MATLAB/Octave dependency — suitable for cluster deployment
- Dependencies: numpy, scipy

### MATPOWER case file location

MATPOWER v8.1 is at `/mnt/c/Projects/matpower8.1`. Case files are in `/mnt/c/Projects/matpower8.1/data/`.

To generate TSUM inputs for other cases:
```bash
cd "/mnt/c/Projects/adaptMCS-benchmarks/DC-opf model"
python matpower2tsum.py /mnt/c/Projects/matpower8.1/data/case30.m --output_dir ./case30_tsum
python matpower2tsum.py /mnt/c/Projects/matpower8.1/data/case30.m --output_dir ./case30_tsum_bus --include_bus_failures
```
