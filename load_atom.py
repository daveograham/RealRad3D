from ruamel.yaml import YAML
from crtaf import Atom, NaturalBroadening, ScaledExponents
import numpy as np
import astropy.units as u

# Clone https://github.com/Goobley/crtaf-py somewhere and install it to provide the Python helpers
# Also check out the documentation for the standard: https://github.com/Goobley/CommonRTAtomicFormat

atomfile = "atomGen/CaII.yaml"

def aload(afile, check64=False):
    #(cmo): Load the YAML into a dict.
    yaml = YAML()
    with open(afile, 'r') as f:
        model_dict = yaml.load(f)

    model = Atom.model_validate(model_dict)

    level_labels = list(model.levels.keys())

    adata = dict(
        mass=model.element.atomic_mass,
        abundance=10**(model.element.abundance - 12.0), # logarithmic relative to 12 as per convention
        Z=model.element.Z, # Z is typically used as our atomic identifier when we don't have access to strings
        energy=np.array([l.energy.to('eV', equivalencies=u.spectral()).value for l in model.levels.values()]), # .to and .value come from the fact these are astropy.units. Read their docs!
        g=np.array([l.g for l in model.levels.values()]),
        stage=np.array([l.stage for l in model.levels.values()]),
        line_i=np.array([level_labels.index(l.transition[1]) for l in model.lines]), # lower level of line -- the transition pair on a transition is always sorted j, i
        line_j=np.array([level_labels.index(l.transition[0]) for l in model.lines]), # upper level of line
        line_f=np.array([l.f_value for l in model.lines]), # oscillator strength
        line_lambda0=np.array([l.lambda0.to('nm').value for l in model.lines]), # rest wavelength
        line_Aji=np.array([l.Aji.value for l in model.lines]), # Einstein A
        line_Bji=np.array([l.Bji_wavelength.to('m2 nm / kJ').value for l in model.lines]),
        line_Bij=np.array([l.Bij_wavelength.to('m2 nm / kJ').value for l in model.lines]),
        g_natural=np.zeros(len(model.lines)), # natural broadening width
    )

    for l_idx, l in enumerate(model.lines):
        for b in l.broadening:
            if isinstance(b, NaturalBroadening):
                adata['g_natural'][l_idx] = b.value.to('1/s').value
                break
        else:
            raise ValueError(f'Did not find natural broadening for line {l}')

    max_broadening_terms = max([len(l.broadening) - 1 for l in model.lines])
    num_lines = len(model.lines)
    adata['broad_scaling'] = np.zeros((num_lines, max_broadening_terms)) # scaling terms for other broadenings
    adata['broad_ele_exp'] = np.zeros((num_lines, max_broadening_terms)) # electron exponent for other broadening
    adata['broad_temp_exp'] = np.zeros((num_lines, max_broadening_terms)) # temperature exponent for other broadening
    adata['broad_nh0_exp'] = np.zeros((num_lines, max_broadening_terms))  # neutral H exponent for other broadening

    for l_idx, l in enumerate(model.lines):
        b_idx = 0
        for b in l.broadening:
            if isinstance(b, NaturalBroadening):
                continue
            elif isinstance(b, ScaledExponents):
                adata['broad_scaling'][l_idx, b_idx] = b.scaling
                adata['broad_ele_exp'][l_idx, b_idx] = b.electron_exponent
                adata['broad_temp_exp'][l_idx, b_idx] = b.temperature_exponent
                adata['broad_nh0_exp'][l_idx, b_idx] = b.hydrogen_exponent
                b_idx += 1
            else:
                raise ValueError(f"Got unexpected broadening type {type(b)}, please simplify the atom first!")
            
    if check64 == True:
        akeys = adata.keys()
        for key in akeys:
            if isinstance(adata[key],np.ndarray):
                if adata[key].dtype == 'float64':
                    array32 = adata[key].astype('float32')
                    adata[key] = array32
                    print('changed',key,'to f32')
    
    return adata

if __name__ == "__main__":
    ATOM_PATH = "CaII.yaml"
    atompy = aload(ATOM_PATH, check64=True)