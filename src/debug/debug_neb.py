"""NEB 诊断脚本：检查位点距离、收敛情况、能量分布"""
import numpy as np
from chgnet.model import StructOptimizer
from ase.mep import SingleCalculatorNEB
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from mp_api.client import MPRester

API_KEY = "ldv96e4HWbzrle58oBzOUfp5OpDTJ6UQ"
MP_ID = "mp-867252"  # LiZn2Ni — 上次势垒只有 0.026 eV

print(f"=== NEB 诊断: {MP_ID} ===\n")

# 加载
relaxer = StructOptimizer()
adaptor = AseAtomsAdaptor()
calc = relaxer.calculator

with MPRester(API_KEY) as mpr:
    structure = mpr.get_structure_by_material_id(MP_ID)

# 弛豫体相
print("1. 弛豫体相...")
result_bulk = relaxer.relax(structure, fmax=0.1, steps=200, relax_cell=True, verbose=False)
bulk = result_bulk["final_structure"]

# 构建 slab
print("2. 构建 slab...")
slabgen = SlabGenerator(bulk, (1, 1, 0), 10, 15, center_slab=True, primitive=True)
slab = slabgen.get_slabs()[0]
ase_slab = adaptor.get_atoms(slab)

# 固定底部
z_vals = ase_slab.get_positions()[:, 2]
z_min, z_max = z_vals.min(), z_vals.max()
threshold = z_min + (z_max - z_min) * 0.33
fixed = [i for i, z in enumerate(z_vals) if z < threshold]
ase_slab.set_constraint(FixAtoms(indices=fixed))
print(f"   slab 原子数: {len(ase_slab)}, 固定: {len(fixed)}")

# 弛豫清洁表面
print("3. 弛豫表面...")
slab_pmg = adaptor.get_structure(ase_slab)
result_clean = relaxer.relax(slab_pmg, fmax=0.05, steps=200, relax_cell=False, verbose=False)
clean_slab = result_clean["final_structure"]

# 找吸附位点
print("4. 找吸附位点...")
asf = AdsorbateSiteFinder(clean_slab)
ads_sites = asf.find_adsorption_sites(distance=2.0)

for stype in ["top", "bridge", "hollow"]:
    sites = ads_sites.get(stype, [])
    print(f"   {stype}: {len(sites)} 个")
    for i, s in enumerate(sites[:3]):
        print(f"     [{i}] {np.array(s).round(3)}")

# 弛豫 Li 在各位点
print("\n5. 弛豫 Li 原子...")
ase_clean = adaptor.get_atoms(clean_slab)

def relax_li(site, label):
    atoms = ase_clean.copy()
    atoms.append("Li")
    atoms.positions[-1] = site
    atoms.set_constraint(FixAtoms(indices=list(range(len(atoms) - 1))))
    atoms.calc = calc
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=0.05, steps=100)
    li_pos = atoms[-1].position
    e = atoms.get_potential_energy()
    print(f"   {label}: Li pos = {li_pos.round(3)}, E = {e:.4f} eV")
    return atoms

results = {}
for stype in ["top", "bridge", "hollow"]:
    sites = ads_sites.get(stype, [])
    if sites:
        results[stype] = relax_li(sites[0], stype)

# 检查位点间距离
print("\n6. 位点间 Li 距离:")
pairs = [("top", "bridge"), ("bridge", "hollow"), ("hollow", "top")]
for a, b in pairs:
    if a in results and b in results:
        pos_a = results[a][-1].position
        pos_b = results[b][-1].position
        dist = np.linalg.norm(pos_a - pos_b)
        print(f"   {a} → {b}: {dist:.3f} Å {'⚠️ 太短!' if dist < 0.5 else '✓'}")

# 跑 NEB (top → bridge)
if "top" in results and "bridge" in results:
    init = results["top"]
    final = results["bridge"]
    final.set_cell(init.get_cell(), scale_atoms=True)
    li_dist = np.linalg.norm(init[-1].position - final[-1].position)
    print(f"\n7. NEB (top→bridge, Li距离={li_dist:.3f}Å):")

    images = [init]
    for _ in range(5):
        img = init.copy()
        img.calc = calc
        images.append(img)
    images.append(final)

    neb = SingleCalculatorNEB(images, climb=True, method="improvedtangent")
    neb.interpolate(method="idpp")

    # 打印 interpolation 后的能量
    energies_before = [img.get_potential_energy() for img in images]
    print(f"   interpolation 后能量: {[f'{e:.3f}' for e in energies_before]}")

    opt = BFGS(neb, logfile=None)
    converged = opt.run(fmax=0.10, steps=200)
    print(f"   收敛: {converged}, 步数: {opt.nsteps}")

    energies = [img.get_potential_energy() for img in images]
    print(f"   优化后能量:")
    for i, e in enumerate(energies):
        marker = " ← max" if e == max(energies) else ""
        print(f"     image {i}: {e:.4f} eV{marker}")
    barrier = max(energies) - energies[0]
    print(f"   势垒: {barrier:.4f} eV")
