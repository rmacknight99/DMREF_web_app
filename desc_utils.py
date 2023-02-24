import os, json, sys, shutil
import pandas as pd
import numpy as np
from morfeus import conformer, Dispersion, read_xyz, XTB, LocalForce, SASA, utils
from rdkit.Chem import AllChem as Chem
import autode as ade

def gen_crest(xyz_files, out_files):
    # generate crest conformer ensemble (crest_conformers.xyz, cre_members, crest.energies)
    # from a group of xyz files for the same molecule (list_xyz_files)
    fail = False
    # initialize lists
    cre_members = []
    energies = []
    xyz = []
    raw_files = []
    cre_members.append(len(xyz_files)) # cre_members starts with number of conformers
    col_0 = 1 # always 1 degeneracy ?
    col_1 = 0 # initialize cumulative sum
    col_2 = 0 # initialize cumulative sum
    for i, xyz_file in enumerate(xyz_files):
        with open(xyz_file, "r") as f: # open xyz files
            lines = f.readlines()
            num_atoms = lines[0] # first line as number of atoms
        with open(out_files[i], "r") as orca_out: # open orca output
            orca_lines = orca_out.readlines() 
            for orca_line in orca_lines:
                if "FINAL SINGLE POINT" in orca_line:
                    energy = float(orca_line.split()[-1]) # get energy for this conformer
        try:
            energies.append(energy) # append energy
            xyz_lines = [] # for coordinates
            for line in lines[2:]:
                xyz_lines.append(line) # append coordinated
            col_0 = 1
            col_1 += 1
            col_2 += 1
            cre_members_string = f"\t{col_0}\t{col_1}\t{col_2}"
            cre_members.append(cre_members_string) # append to cre_members
            xyz.append([str(num_atoms)] + [str(energy)+"\n"] + xyz_lines) # make xyz_file lines
            raw_files.append(xyz_file) # append to list of all files
        except:
            fail = True
    energies = [(e - min(energies))*627.5 for e in energies] # convert to kcal/mol from Hartree
    zipped = zip(energies, xyz, raw_files) # zipped list for sorting
    zipped = list(zipped)
    res = sorted(zipped, key = lambda x: x[0])
    energies, xyz, raw_files = [list(tup) for tup in zip(*res)] # unpack sorted by energy
    # write the files
    for index, xyz_f in enumerate(xyz):
        xyz_f[1] = str(energies[index])+"\n"
    try:
        os.remove(f"crest_conformers.xyz")
        os.remove(f"cre_members")
        os.remove(f"crest.energies")
    except:
        pass
    with open(f"crest_conformers.xyz", "w") as conformers:
        for xyz_f in xyz:
            for line in xyz_f:
                conformers.write(line)
    with open(f"cre_members", "w") as cre_memb:
        for line in cre_members:
            cre_memb.write(str(line)+"\n")
    with open(f"crest.energies", "w") as e:
        for index, energy in enumerate(energies):
            e.write(str(index+1)+"\t"+str(energy)+"\n")

    return energies

def get_ORCA_descriptors(orca_dir, n=58, ORCA=True, withh=True, dispersion=True, SASA=True, XTB=True, LFCs=False):
    conf_dict = {}
    desc_dict = {}
    for i in list(range(n)):
        print(f"----- Salt #{i} -----")
#        orca_dir = f"salt_{i}/with_chlorine/CONFORMERS/ORCA/"
        xyz_files = [orca_dir+j for j in sorted(os.listdir(orca_dir)) if j.endswith("optimised_orca.xyz")]
        out_files = [orca_dir+j for j in sorted(os.listdir(orca_dir)) if j.endswith(".out")]
        energies = gen_crest(xyz_files, out_files)
        ce = conformer.ConformerEnsemble.from_crest(".")
        conf_dict[i] = (ce, ce.n_conformers)
        gen_hess(i, 64, ORCA=True, withh=True)
        bw = ce.boltzmann_weights().T
        ensemble_desc, calc_dict = get_ensemble_descriptors(i, ORCA=ORCA, withh=withh, dispersion=dispersion, SASA=SASA, XTB=XTB, LFCs=LFCs)
        #n_conformers, n_descriptors = len(list(ensemble_desc)), len(list(ensemble_desc[0].values()))
        bw_desc = boltz_weight_desc(ensemble_desc, bw)
        desc_dict[i] = bw_desc

    return desc_dict, conf_dict, calc_dict

def boltz_weight_desc(desc, weights):

    n_conformers, n_descriptors = len(list(desc)), len(list(desc[0].values()))
    keys = list(desc[0].keys())
    with_desc_array = np.zeros((n_conformers, n_descriptors))
    for conf_index, conf_descriptors in enumerate(desc):
        for val_index, val in enumerate(conf_descriptors.values()):
            with_desc_array[conf_index, val_index] = round(val, 4)
    bw_desc = (with_desc_array.T * weights).T.sum(axis=0)
    print(bw_desc.shape)
    bw_desc = {key: bw_desc[i] for i, key in enumerate(keys)}
    return bw_desc

def get_dispersion(filename, desc_dict):

    elements, coordinates = read_xyz(filename)

    disp = Dispersion(elements, coordinates)
    disp.compute_coefficients()
    disp.compute_p_int()

    desc_dict["disp_area"] = disp.area
    desc_dict["disp_volume"] = disp.volume
    desc_dict["disp_p_int"] = disp.p_int
    desc_dict["disp_p_max"] = disp.p_max
    desc_dict["disp_p_min"] = disp.p_min

    return desc_dict, disp

def get_SASA(filename, desc_dict):

    elements, coordinates = read_xyz(filename)
    sasa = SASA(elements, coordinates)
    desc_dict["max_atom_area"] = max(sasa.atom_areas)
    desc_dict["sasa_area"] = sasa.area
    desc_dict["sasa_volume"] = sasa.volume
    
    return desc_dict, sasa

def get_XTB(filename, desc_dict, charge, unpaired, solvent="water"):

    elements, coordinates = read_xyz(filename)
    xtb = XTB(elements, coordinates, version="2", charge=charge, n_unpaired=unpaired, solvent=solvent)

    desc_dict["ip"] = xtb.get_ip()
    desc_dict["ea"] = xtb.get_ea()
    dipole = xtb.get_dipole()
    desc_dict["dipole_moment"] = np.sqrt(np.sum(dipole**2))
    desc_dict["e_philicity"] = xtb.get_global_descriptor("electrophilicity", corrected=True)
    desc_dict["n_philicity"] = xtb.get_global_descriptor("nucleophilicity", corrected=True)

    return desc_dict, xtb

def get_LFCs(filename, desc_dict):

    elements, coordinates = read_xyz(filename)
    LF = LocalForce(elements, coordinates)
    hess_path = f"hessian_files/{filename[:-4]}/hessian"
    LF.load_file(hess_path, "xtb", "hessian")
    LF.normal_mode_analysis()
    LF.detect_bonds()
    LF.compute_local()
    LF.compute_frequencies()
    LF.compute_compliance()

    desc_dict["LFCs"] = LF.local_force_constants
    desc_dict["LFqCs"] = LF.local_frequencies

    return desc_dict, LF

def find_bond(elements, int_coords, search):

    indices = []
    for index, bond in enumerate(int_coords):
        idx_1 = bond.i - 1
        idx_2 = bond.j - 1
        key = elements[idx_1] + "_" + elements[idx_2]
        if key == search:
            indices.append(index)
    return indices

def resolve_LFCs(filename, desc_dict, LF, search):

    elements, coordinates = read_xyz(filename)
    indices = find_bond(elements, LF.internal_coordinates, search)
    
    for i, index in enumerate(indices):
        desc_dict[f"LFCs_{search}_{i}"] = round(desc_dict["LFCs"][index], 3)
        desc_dict[f"LFqCs_{search}_{i}"] = round(desc_dict["LFqCs"][index], 3)
    desc_dict.pop("LFCs")
    desc_dict.pop("LFqCs")

    return desc_dict, indices

def get_ensemble_descriptors(geom_dir, salt_number, ORCA=True, withh=True, dispersion=True, SASA=True, XTB=True, LFCs=None):

    descriptors = []
    HOME = os.getcwd()
    if withh:
        if ORCA:
            search = "optimised_orca.xyz"
#            geom_dir = f"salt_{salt_number}/with_chlorine/CONFORMERS/ORCA/"
        else:
            search = ".xyz"
#            geom_dir = f"salt_{salt_number}/with_chlorine/CONFORMERS/"
    else:
        if ORCA:
            search = "optimised_orca.xyz"
#            geom_dir = f"salt_{salt_number}/with_chlorine/CONFORMERS/ORCA/"
        else:
            search = ".xyz"
#            geom_dir = f"salt_{salt_number}/with_chlorine/CONFORMERS/"
    os.chdir(geom_dir)
    print(f"\tworking on {geom_dir}")
    for j, i in enumerate(os.listdir(".")):
        if search in i:
            desc_dict = {}
            filename = i
            if dispersion:
                desc_dict, disp = get_dispersion(filename, desc_dict)
            if SASA:
                desc_dict, sasa = get_SASA(filename, desc_dict)
            if XTB:
                charge = 0 if withh else 1
                unpaired = 0
                desc_dict, xtb = get_XTB(filename, desc_dict, charge, unpaired)
            if LFCs is not None:
                desc_dict, LF = get_LFCs(filename, desc_dict)
                desc_dict, indices = resolve_LFCs(filename, desc_dict, LF, LFCs)
            else:
                LF = None
            descriptors.append(desc_dict)
    os.chdir(HOME)
    for i, mol in enumerate(descriptors):
        for key, value in mol.items():
            descriptors[i][key] = round(value, 3)
        
    return descriptors, {"dispersion": disp, "SASA": sasa, "xTB": xtb, "LFs": LF}

def gen_hess(geom_dir, salt_number, CORES, ORCA=False, withh=True):
    
    HOME = os.getcwd()

    if withh:
        chrg, uhf = 0, 0
        if ORCA:
            search = "optimised_orca.xyz"
#            geom_dir = f"salt_{salt_number}/with_chlorine/CONFORMERS/ORCA/"
        else:
            search = ".xyz"
#            geom_dir = f"salt_{salt_number}/with_chlorine/CONFORMERS/"
    else:
        chrg, uhf = 1, 0
        if ORCA:
            search = "optimised_orca.xyz"
#            geom_dir = f"salt_{salt_number}/with_chlorine/CONFORMERS/ORCA/"
        else:
            search = ".xyz"
#            geom_dir = f"salt_{salt_number}/with_chlorine/CONFORMERS/"

    os.chdir(geom_dir)
    os.system("mkdir -p hessian_files/") # make hessian files
    ROOT = os.getcwd()
    for xyz_file in os.listdir("."): # iterate over xyz files
        if search in xyz_file:
            name = xyz_file[:-4]
            if not os.path.exists(f"hessian_files/{name}/hessian"):
                os.system(f"mkdir -p hessian_files/{name}/")
                os.system(f"cp {xyz_file} hessian_files/{name}/")
                os.chdir(f"hessian_files/{name}/")
                os.system(f"xtb {xyz_file} --hess --alpb water -c {chrg} -u {uhf} -P {CORES} >/dev/null 2>&1")
                os.chdir(ROOT)
    os.chdir(HOME)

def dump_conformers(dump_dir, ce, salt_number, withh=True):

    os.system(f"mkdir -p {dump_dir}/CONFORMERS")
    ce.write_xyz(f"{dump_dir}/CONFORMERS/conformer", separate=True)
        
def trim_conformers(ce, rmsd_thresh=0.35, energy_thresh=3.0):

    ce.prune_energy(threshold=energy_thresh)
    ce.prune_rmsd(method="obrms-batch", thres=rmsd_thresh)
    ce.sort()

    return ce

def make_best_ensemble(path_with, path_without=None):

    with_ce, with_dict = load_ensemble(f"{path_with}/CREST/")
    if path_without is not None:
        without_ce, without_dict = load_ensemble(f"{path_without}/CREST/")
    else:
        without_ce, without_dict = load_ensemble(f"{path_with}/CREST/")
    elements = utils.convert_elements(without_dict["best_elements"], output="symbols")

    if list(with_dict["best_elements"]) == list(without_dict["best_elements"]):
        best_ensemble = conformer.ConformerEnsemble(elements=elements)
        best_ensemble.add_conformers([with_dict["best_coords"]], None, None, None)
        best_ensemble.add_conformers([without_dict["best_coords"]], None, None, None)
        
    return with_ce, without_ce, best_ensemble

def remove_counterion(ce_elements):

    unique, counts = np.unique(ce_elements, return_counts=True)
    n_chlorines = dict(zip(unique, counts))[17]

    if n_chlorines > 1:
        ii = np.where(ce_elements == 17)[0][-1]
        ce_elements = list(ce_elements)
        ce_elements.pop(ii)
    else:
        ce_elements = list(ce_elements)
        ce_elements.remove(17)

    return ce_elements

def load_ensemble(path):

    ce = conformer.ConformerEnsemble.from_crest(path)
    ce_best_elements = ce.elements
    if "with_" in path:
        ce_best_elements = remove_counterion(ce_best_elements)
        ce_best_coords = np.delete(ce.get_coordinates()[0], (5), axis=0)
    else:
        ce_best_coords = ce.get_coordinates()[0]
        ce_best_elements = list(ce_best_elements)

    ddict = {"ce": ce, "best_coords": ce_best_coords, "best_degen": None, "best_elements": ce_best_elements, "best_energy": None}
    
    return ce, ddict

def run_crest_pipeline(input_file, cores):

    smiles = sorted([string + ".[Cl-]" for string in pd.read_csv(input_file, names=["x"])["x"].tolist()])
    for index, smiles_string in enumerate(smiles):
        print(f"-----On salt #{index}-----")
        with_chlorine = smiles_string
        n_atoms = len(Chem.AddHs(Chem.MolFromSmiles(with_chlorine)).GetAtoms())
        without_chlorine = smiles_string.split(".")[0]
        print(with_chlorine)
        pipeline(with_chlorine, f"salt_{index}/with_chlorine", cores, n_atoms, 0, 1)
        pipeline(without_chlorine, f"salt_{index}/without_chlorine", cores, n_atoms, 1, 1)

def get_constraints(ID):

    os.system(f"obabel -ixyz {ID}/init.xyz -omol -O {ID}/init.mol")
    mol = Chem.MolFromMolFile(f"{ID}/init.mol", sanitize=False)
    os.remove(f"{ID}/init.mol")
    constrain_indices = []
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for index, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == "N":
            constrain_indices.append(index)
            bonded_atoms = [sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])[1] for b in atom.GetBonds()]
            bonded_atoms = [a for a in bonded_atoms if symbols[a] == "H"]
            constrain_indices += bonded_atoms
        elif atom.GetSymbol() == "Cl":
            constrain_indices.append(index)

    return constrain_indices

def write_xtb_inp():

    with open("xtb.inp", "w") as xtb_inp:
        xtb_inp.write("$fix\n")
        xtb_inp.write(f"\telements: N,H,O,Cl\n")
        xtb_inp.write("$end\n")

def check_xTB(ID):

    os.system("mkdir -p xTB_errors")
    with open(f"{ID}/XTB/xtb.out", "r") as f:
        lastline = f.readlines()[-1]
        lastline.replace("\n", "")
    if "#" in lastline:
        try:
            shutil.move(f"{ID}", "xTB_errors")
            print(f"\tmoved {ID} folders to xTB_errors", flush=True)
        except:
            pass

def pipeline(SMILES, ID, cores, n_atoms, charge, multiplicity=1):

    ROOT = os.getcwd()
    CREST_CORES = cores
    XTB_CORES = min(cores, 16)

    # Generate initial XYZ file from SMILES with openbabel
    if not os.path.exists(f"{ID}/init.xyz"):
#        print(f"-----Generating Initial XYZ Structure for {SMILES}-----\n", flush=True)
        os.system(f"mkdir -p {ID}")
        os.system(f"obabel -:'{SMILES}' --addhs --gen3d -O {ID}/init.xyz > /dev/null 2>&1")
        # Record the SMILES
        os.system(f"obabel -ixyz {ID}/init.xyz -osmi -O {ID}/smiles.txt > /dev/null 2>&1")

    # Initial geometry optimization with xTB
    if not os.path.exists(f"{ID}/XTB/"):
#        print(f"-----Optimizing Initial XYZ Structure with xTB-----", flush=True)
        try:
            os.chdir(ID)
            uhf = multiplicity - 1
            #print(f"\tSMILES: {SMILES}\n\tCHARGE: {charge}\n\tUHF: {uhf}\n", flush=True)
            cmd = f"xtb init.xyz --opt -c {charge} -u {uhf} -P {XTB_CORES} --alpb water > xtb.out"
            os.system("mkdir -p XTB")
            shutil.copy("init.xyz", "XTB/init.xyz")
            os.chdir("XTB")
            # Write constraint input
            os.system(cmd)
            os.chdir(ROOT) # always return to root upon completion
        except:
            os.chdir(ROOT) # always return to root upon completion  
    
    if not os.path.exists(f"{ID}/XTB/xtbopt.xyz"):
#        print(f"-----RERUNNING Initial Optimization at GFNFF level of theory-----", flush=True)
        try:
            os.chdir(ID)
            uhf = multiplicity - 1
            #print(f"\tSMILES: {SMILES}\n\tCHARGE: {charge}\n\tUHF: {uhf}\n", flush=True)
            cmd = f"xtb init.xyz --opt -c {charge} -u {uhf} -P {XTB_CORES} --gfnff --alpb water > xtb.out"
            os.system("mkdir -p XTB")
            shutil.copy("init.xyz", "XTB/init.xyz")
            os.chdir("XTB")
            os.system(cmd)
            os.chdir(ROOT) # always return to root upon completion
        except:
            os.chdir(ROOT) # always return to root upon completion  
    
    # check for xTB errors and move
    check_xTB(ID)

    # CREST conformer generation with gfn2//gfnff
    if not os.path.exists(f"{ID}/CREST/"):
        #print(f"-----CREST Conformer Generation-----\n", flush=True)
        try:
            os.chdir(ID)
            os.system("mkdir -p CREST")
            shutil.copy("XTB/xtbopt.xyz", "CREST/xtbopt.xyz")
            os.chdir("CREST")
            if n_atoms > 35:
                os.system(f"crest xtbopt.xyz --gfn2//gfnff --chrg {charge} --uhf {uhf} --cbonds --alpb water -T {CREST_CORES+64} --quick > crest.out")
            else:
                os.system(f"crest xtbopt.xyz --gfn2//gfnff --chrg {charge} --uhf {uhf} --cbonds --alpb water -T {CREST_CORES} > crest.out")
            os.chdir(ROOT) # always return to root upon completion  
        except:
            os.chdir(ROOT) # always return to root upon completion

    # If CREST not converged use GNF2
    if not os.path.exists(f"{ID}/CREST/cre_members"):
        #print(f"-----RERUNNING CREST with GFN2-----", flush=True)
        try:
            os.chdir(ID)
            os.system("mkdir -p CREST")
            shutil.copy("XTB/xtbopt.xyz", "CREST/xtbopt.xyz")
            os.chdir("CREST")
            os.system(f"crest xtbopt.xyz --gfn2 --chrg {charge} --uhf {uhf} --cbonds --alpb water --quick -T {CREST_CORES} > crest.out")
            os.chdir(ROOT) # always return to root upon completion
        except:
            os.chdir(ROOT) # always return to root upon completion

def run_dft(path, salt_number, CORES, withh=True):

    ade.Config.n_cores = CORES
    ade.Config.max_core = 8000
    ade.Config.lcode = "XTB"
    ade.Config.hcode = "ORCA"

    ade.Config.ORCA.keywords.set_opt_basis_set("def2-TZVP")
    ade.Config.ORCA.keywords.set_functional("PBE0")

    ROOT = os.getcwd()
#    path = f"salt_{salt_number}/with_chlorine/CONFORMERS/" if withh else f"salt_{salt_number}/without_chlorine/CONFORMERS/"
    charge = 0 if withh else 1
    os.chdir(path)
    os.system("mkdir -p ORCA")
    home = os.getcwd()
    cn = 0 

    files = [i for i in os.listdir(".") if i.endswith(".xyz")]
    n_conformers = len(files)
    for f in files:
        print(f"  Optimizing conf {cn+1}/{n_conformers} at DFT level of theory")
        conf_geom = f
        conf = ade.Molecule(conf_geom, name=f"salt_{salt_number}_conf_{cn}", charge=charge, mult=1, solvent_name="water")
        os.chdir("ORCA")
        conf.optimise(method=ade.methods.get_hmethod())
        os.chdir(home)
        cn +=1

    os.chdir(ROOT)
