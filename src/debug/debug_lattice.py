import numpy as np
from mp_api.client import MPRester
from pymatgen.core.surface import SlabGenerator, generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from chgnet.model import StructOptimizer

API_KEY = "ldv96e4HWbzrle58oBzOUfp5OpDTJ6UQ"
relaxer = StructOptimizer()

with MPRester(API_KEY) as mpr:
    # Get Li(110)
    li_entries = mpr.get_entries_in_chemsys(["Li"])
    li_struct = sorted(li_entries, key=lambda e: e.energy_per_atom)[0].structure
    result = relaxer.relax(li_struct, fmax=0.05, steps=500, relax_cell=True, verbose=False)
    li_struct = result["final_structure"]
    print(f"Li(110): a={li_struct.lattice.a:.3f}, b={li_struct.lattice.b:.3f}, c={li_struct.lattice.c:.3f}, "
          f"alpha={li_struct.lattice.alpha:.1f}, beta={li_struct.lattice.beta:.1f}, gamma={li_struct.lattice.gamma:.1f}")
    
    slabgen = SlabGenerator(li_struct, (1,1,0), 12.0, 15.0, center_slab=True, primitive=True)
    li_slab = slabgen.get_slabs()[0]
    li_cell = li_slab.lattice.matrix
    print(f"\nLi(110) slab 2D vectors:")
    print(f"  a = ({li_cell[0][0]:.3f}, {li_cell[0][1]:.3f})  |a|={np.linalg.norm(li_cell[0][:2]):.3f}")
    print(f"  b = ({li_cell[1][0]:.3f}, {li_cell[1][1]:.3f})  |b|={np.linalg.norm(li_cell[1][:2]):.3f}")
    
    for mp_id in ["mp-31168", "mp-862318", "mp-867252"]:
        s = mpr.get_structure_by_material_id(mp_id)
        s = SpacegroupAnalyzer(s).get_conventional_standard_structure()
        result = relaxer.relax(s, fmax=0.05, steps=500, relax_cell=True, verbose=False)
        s = result["final_structure"]
        print(f"\n{mp_id} ({s.composition.reduced_formula}):")
        print(f"  bulk: a={s.lattice.a:.3f}, b={s.lattice.b:.3f}, c={s.lattice.c:.3f}, "
              f"alpha={s.lattice.alpha:.1f}, beta={s.lattice.beta:.1f}, gamma={s.lattice.gamma:.1f}")
        
        try:
            slabs = generate_all_slabs(s, max_index=1, min_slab_size=12, min_vacuum_size=15,
                                       center_slab=True, primitive=True)
            print(f"  {len(slabs)} slab(s):")
            for sl in slabs:
                m = sl.miller_index
                sl_cell = sl.lattice.matrix
                a_len = np.linalg.norm(sl_cell[0][:2])
                b_len = np.linalg.norm(sl_cell[1][:2])
                a_2d = sl_cell[0][:2]
                b_2d = sl_cell[1][:2]
                cos_a = np.dot(a_2d, b_2d) / (a_len * b_len)
                angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                print(f"    ({m[0]}{m[1]}{m[2]}): |a|={a_len:.2f}, |b|={b_len:.2f}, angle={angle:.1f}")
        except Exception as e:
            print(f"  Error: {e}")
