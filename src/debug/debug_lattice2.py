import numpy as np
from mp_api.client import MPRester
from pymatgen.core.surface import SlabGenerator, generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from chgnet.model import StructOptimizer

API_KEY = "ldv96e4HWbzrle58oBzOUfp5OpDTJ6UQ"
relaxer = StructOptimizer()

def match_check(li_slab, film_slab, max_scale=8, angle_tol=5):
    """Debug version of mismatch"""
    a_f = np.array(film_slab.lattice.matrix[0][:2])
    b_f = np.array(film_slab.lattice.matrix[1][:2])
    a_li = np.array(li_slab.lattice.matrix[0][:2])
    b_li = np.array(li_slab.lattice.matrix[1][:2])
    
    print(f"  film a=({a_f[0]:.3f},{a_f[1]:.3f}) |a|={np.linalg.norm(a_f):.3f}")
    print(f"  film b=({b_f[0]:.3f},{b_f[1]:.3f}) |b|={np.linalg.norm(b_f):.3f}")
    print(f"  Li   a=({a_li[0]:.3f},{a_li[1]:.3f}) |a|={np.linalg.norm(a_li):.3f}")
    print(f"  Li   b=({b_li[0]:.3f},{b_li[1]:.3f}) |b|={np.linalg.norm(b_li):.3f}")
    
    # Angles
    def get_angle(v1, v2):
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    
    ang_f = get_angle(a_f, b_f)
    ang_li = get_angle(a_li, b_li)
    diff = abs(ang_f - ang_li)
    print(f"  film angle={ang_f:.1f} Li angle={ang_li:.1f} diff={diff:.1f}")
    
    # Try simple scaling first
    best = float('inf')
    for n1 in range(1, max_scale+1):
        for m1 in range(1, max_scale+1):
            ma = abs(n1*np.linalg.norm(a_f) - m1*np.linalg.norm(a_li)) / (m1*np.linalg.norm(a_li)) * 100
            for n2 in range(1, max_scale+1):
                for m2 in range(1, max_scale+1):
                    mb = abs(n2*np.linalg.norm(b_f) - m2*np.linalg.norm(b_li)) / (m2*np.linalg.norm(b_li)) * 100
                    mm = max(ma, mb)
                    if mm < best and diff <= angle_tol:
                        best = mm
                        if mm < 20:
                            print(f"    MATCH: n1={n1} m1={m1} n2={n2} m2={m2} ma={ma:.1f}% mb={mb:.1f}%")
    print(f"  best mismatch: {best:.1f}%")
    return best if best < 900 else 999

with MPRester(API_KEY) as mpr:
    # Li(110) without relaxation
    li_entries = mpr.get_entries_in_chemsys(["Li"])
    li_struct = sorted(li_entries, key=lambda e: e.energy_per_atom)[0].structure
    print(f"Li bulk: a={li_struct.lattice.a:.3f} b={li_struct.lattice.b:.3f} c={li_struct.lattice.c:.3f}")
    
    slabgen = SlabGenerator(li_struct, (1,1,0), 12, 15, center_slab=True, primitive=True)
    li_slab = slabgen.get_slabs()[0]
    
    for mp_id in ["mp-31168"]:
        s = mpr.get_structure_by_material_id(mp_id)
        s = SpacegroupAnalyzer(s).get_conventional_standard_structure()
        result = relaxer.relax(s, fmax=0.05, steps=500, relax_cell=True, verbose=False)
        s = result["final_structure"]
        print(f"\n{mp_id}: a={s.lattice.a:.3f} b={s.lattice.b:.3f} c={s.lattice.c:.3f}")
        
        slabs = generate_all_slabs(s, max_index=1, min_slab_size=12, min_vacuum_size=15,
                                   center_slab=True, primitive=True)
        for sl in slabs[:3]:
            print(f"  Miller {sl.miller_index}:")
            match_check(li_slab, sl)
