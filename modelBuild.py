import taichi as ti
import taichi.math as tm
import numpy as np
from taichi.math import vec2, vec3, vec4, ivec2, ivec3, ivec4

TI_ARCH = ti.gpu
TI_FP = ti.f32
TI_INT = ti.i32
PY_FP = np.float32

from config import *

@ti.data_oriented
class Constants:
    def __init__(self):
        self.c = 299792458.0
        self.h = 6.62607015e-34
        self.hc = 1.9864458571489286e-25
        self.k_b = 1.380649e-23
        self.ev = 1.602176633999999894e-19
        self.hion = 13.5984 # [eV]
        self.m_e = 9.1093837139e-31
        self.amu = 1.99264688270e-26
        #average mass unit per proton
        self.umass = self.amu/12.0

        self.log_T_np = np.array([3.000000e+00, 3.133333e+00, 3.266667e+00,
                    3.400000e+00, 3.533333e+00, 3.666667e+00,
                    3.800000e+00, 3.933333e+00, 4.066667e+00,
                    4.200000e+00, 4.333333e+00, 4.466667e+00,
                    4.600000e+00, 4.733333e+00, 4.866667e+00,
                    5.000000e+00, 5.133333e+00, 5.266667e+00,
                    5.400000e+00, 5.533333e+00, 5.666667e+00,
                    5.800000e+00, 5.933333e+00, 6.066667e+00,
                    6.200000e+00, 6.333333e+00, 6.466667e+00,
                    6.600000e+00, 6.733333e+00, 6.866667e+00,
                    7.000000e+00],dtype=np.float32)
        #partition function
        self.h_partfn_np = np.array([2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00,
                       2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000012e+00,
                       2.000628e+00, 2.013445e+00, 2.136720e+00, 2.776522e+00,
                       4.825934e+00, 9.357075e+00, 1.691968e+01, 2.713700e+01,
                       3.892452e+01, 5.101649e+01, 6.238607e+01, 7.240922e+01,
                       8.083457e+01, 8.767237e+01, 9.307985e+01, 9.727538e+01,
                       1.004851e+02, 1.029155e+02, 1.047417e+02, 1.061062e+02,
                       1.071217e+02, 1.078750e+02, 1.084327e+02],dtype=np.float32)
        
        self.log_T_grid = ti.field(dtype=TI_FP, shape=self.log_T_np.shape)
        self.log_T_grid.from_numpy(self.log_T_np.astype(PY_FP))
        self.h_partfn_grid = ti.field(dtype=TI_FP, shape=self.h_partfn_np.shape)
        self.h_partfn_grid.from_numpy(self.h_partfn_np.astype(PY_FP))

@ti.data_oriented
class TaiAtom:
    def __init__(self, atom):
        self.mass = atom["mass"]
        self.abundance =  atom["abundance"]
        self.Z = atom["Z"]
        self.energy = ti.field(dtype=TI_FP, shape=atom["energy"].shape)
        self.energy.from_numpy(atom["energy"].astype(PY_FP))
        self.g = ti.field(dtype=TI_INT, shape=atom["g"].shape)
        self.g.from_numpy(atom["g"])
        self.stage = ti.field(dtype=TI_INT, shape=atom["stage"].shape)
        self.stage.from_numpy(atom["stage"])

        #lower and upper level numbers
        self.line_i = ti.field(dtype=TI_INT, shape=atom["line_i"].shape)
        self.line_i.from_numpy(atom["line_i"])
        self.line_j = ti.field(dtype=TI_INT, shape=atom["line_j"].shape)
        self.line_j.from_numpy(atom["line_j"])
        #oscillator strength
        self.line_f = ti.field(dtype=TI_FP, shape=atom["line_f"].shape)
        self.line_f.from_numpy(atom["line_f"].astype(PY_FP))
        #rest line wavelength
        self.line_lambda0 = ti.field(dtype=TI_FP, shape=atom["line_lambda0"].shape)
        self.line_lambda0.from_numpy(atom["line_lambda0"].astype(PY_FP))
        #einstein A and B coefficients
        self.line_Aji = ti.field(dtype=TI_FP, shape=atom["line_Aji"].shape)
        self.line_Aji.from_numpy(atom["line_Aji"].astype(PY_FP))
        self.line_Bji = ti.field(dtype=TI_FP, shape=atom["line_Bji"].shape)
        self.line_Bji.from_numpy(atom["line_Bji"].astype(PY_FP))
        self.line_Bij = ti.field(dtype=TI_FP, shape=atom["line_Bij"].shape)
        self.line_Bij.from_numpy(atom["line_Bij"].astype(PY_FP))

        #natural broadening
        self.g_natural = ti.field(dtype=TI_FP, shape=atom["g_natural"].shape)
        self.g_natural.from_numpy(atom["g_natural"].astype(PY_FP))
        #broadening scale factor
        self.broad_scaling = ti.field(dtype=TI_FP, shape=atom["broad_scaling"].shape)
        self.broad_scaling.from_numpy(atom["broad_scaling"].astype(PY_FP))
        #temp broadening factor
        self.broad_temp_exp = ti.field(dtype=TI_FP, shape=atom["broad_temp_exp"].shape)
        self.broad_temp_exp.from_numpy(atom["broad_temp_exp"].astype(PY_FP))
        #n electron broadening factor
        self.broad_ele_exp = ti.field(dtype=TI_FP, shape=atom["broad_ele_exp"].shape)
        self.broad_ele_exp.from_numpy(atom["broad_ele_exp"].astype(PY_FP))
        #n hydrogen broadening factor
        self.broad_nh0_exp = ti.field(dtype=TI_FP, shape=atom["broad_nh0_exp"].shape)
        self.broad_nh0_exp.from_numpy(atom["broad_nh0_exp"].astype(PY_FP))

@ti.data_oriented
class ModelSetup:
    def __init__(self, ds, ps, constants, atomdata, lineindex, ATOM):
        if ds.program != "dexrt (3d)":
            print("Program tag does not appear to be \"dexrt (3d)\", are you sure this is the right file?")
        if ds.output_format != "sparse":
            raise ValueError("Expected a sparse file")
        self.block_size = int(ds.block_size)        #morton blocksize
        self.block_stride = self.block_size**3
        self.log2_block_size = int(np.log2(self.block_size))
        self.num_active_tiles = int(ds.num_active_tiles.shape[0])   #number of morton tiles

        self.voxel_scale = float(ds.voxel_scale)
        self.offset_x = float(ds.offset_x)
        self.offset_y = float(ds.offset_y)
        self.offset_z = float(ds.offset_z)
        self.offset = vec3(ds.offset_x, ds.offset_y, ds.offset_z)

        self.const = constants
        self.atom = atomdata
        self.restwav = atomdata.line_lambda0[lineindex]
        self.nlevels = atomdata.g.shape[0]
        self.lineindex = lineindex
        ###
        #set up wavelength array
        self.wavenp = self.wave_array(self.restwav, WAVESTEPS, WAVEDELTA)

        self.wdim = self.wavenp.shape[0]
        self.wav_profile = ti.field(dtype=TI_FP, shape=self.wdim)
        self.wav_profile.from_numpy(self.wavenp.astype(PY_FP))
        self.wav_basis = ti.field(dtype=TI_FP, shape=self.wdim)
        self.wav_basis.from_numpy(np.arange(self.wdim, dtype=PY_FP))
        ###

        self.temp = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.temp.from_numpy(ds.temperature.values.astype(PY_FP))
        self.ne = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.ne.from_numpy(ds.ne.values.astype(PY_FP))
        self.nh_tot = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.nh_tot.from_numpy(ds.nh_tot.values.astype(PY_FP))
        self.vx = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.vx.from_numpy(ds.vx.values.astype(PY_FP))
        self.vy = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.vy.from_numpy(ds.vy.values.astype(PY_FP))
        self.vz = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.vz.from_numpy(ds.vz.values.astype(PY_FP))
        self.vturb = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.vturb.from_numpy(ds.vturb.values.astype(PY_FP))

        self.nh0 = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.get_nh0()
        self.wdop = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.get_doppler_width()
        self.gamma = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.get_gamma(lineindex)

        #POPULATIONS
        #FOR CA II LEVELS ARE 20 ONWARDS [9 H, 11 Mg, 6 Ca]
        self.pops = ti.field(dtype=TI_FP, shape=(self.nlevels, self.num_active_tiles * self.block_size**3,))
        # if ATOM == 'CaII':
        #     self.pops.from_numpy(ps.pops.values[Ca:,:].astype(PY_FP))
        self.pops.from_numpy(ps.pops.values[ION_I0:ION_I1,:].astype(PY_FP))
        #SOLAR BOUNDARY LOOKUP TABLE
        self.lookup_ax_lambda = ti.field(dtype=TI_FP, shape=(ds.prom_bc_wavelength.values.shape[0]))
        self.lookup_ax_mu = np.linspace(ds.prom_bc_mu_min.values, ds.prom_bc_mu_max.values, len(ds.prom_bc_mu.values))
        self.mu_max = ds.prom_bc_mu_max.values
        self.mu_min = ds.prom_bc_mu_min.values

        wavidx = self.wave_lookup(self.wavenp, ds.prom_bc_wavelength.values)
        self.wav_to_lookup_index = ti.field(dtype=TI_INT, shape=len(wavidx))
        self.wav_to_lookup_index.from_numpy(wavidx.astype(PY_FP))

        #intensity lookup = atmosdata.prom_bc_I.values
        self.boundary_lookup = ti.field(dtype=TI_FP, shape=(ds.prom_bc_I.values.shape[0], ds.prom_bc_I.values.shape[1]))
        self.boundary_lookup.from_numpy(ds.prom_bc_I.values.astype(PY_FP))


    def wave_lookup(self, wavarr, wav_lookup):
        '''Interpolate tracer wavelength range onto boundary intensity lookup table axis.
        Returns fractional indices that map the tracing wavelength range to the lookup table index'''
        frac_index = np.zeros(len(wavarr))
        lli = np.where((wav_lookup >= wavarr[0]) & (wav_lookup <= wavarr[-1]))
        if len(lli) == 0:
            print('You probably have the wrong boundary table in your model file')
        lli = lli[0]
        lli = np.append(lli, lli[-1]+1) #add one more buffer index
        #subset of lookup lambda array to match input array
        wav_sub = wav_lookup[lli]

        for i,wl in enumerate(wavarr):
            w_frac = np.interp(wl, wav_sub, np.arange(len(wav_sub)))
            frac_index[i] = w_frac + lli[0]
        return frac_index

    def wave_array(self, rest_wave: float, w_points: int, width: float):
        '''Create wavelength base array centred at rest wavelength for odd grids of <w_points> with <width> in nm
        - on even grids rest lies between two closest mid points'''
        left = rest_wave - (width/2.0)
        return left+(np.arange(w_points)*width/(w_points-1))

    #DIRECTION INDEPENDENT
    @ti.kernel
    def get_nh0(self):
        '''Finds LTE ground state fraction of H given temp, n_e, and nh_tot'''
        for i in self.nh0:
            #interp to find partition fn at input t
            hion_k_b = self.const.hion * self.const.ev / self.const.k_b   # H ionisation energy in eV / kB
            #get partition function result from pre comuted partition functions
            tlog = tm.log2(self.temp[i])/tm.log2(10)
            u_hi = lerp(tlog, self.const.log_T_grid, self.const.h_partfn_grid)
            u_hii = 1.0

            saha_const = (2.0 * tm.pi * self.const.k_b) / (self.const.h / (self.const.h / self.const.m_e) )
            ratio = (2.0 * u_hii / (self.ne[i] * u_hi) * ti.pow(saha_const * self.temp[i], 1.5)) * ti.exp(-hion_k_b / self.temp[i])
            self.nh0[i] = self.nh_tot[i] / (1.0 + ratio)

    @ti.kernel
    def get_doppler_width(self):
        '''Returns doppler width in nm in a given line for a temperature t with v turbulence, rest wavlength in nm, atomic mass z in amu'''
        for i in self.wdop:
            vterm2 = ((2.0 * self.const.k_b * self.temp[i]) / (self.const.umass * self.atom.mass))
            self.wdop[i] = (self.restwav / self.const.c) * ti.sqrt(vterm2 + ti.pow(self.vturb[i], 2.0))

    @ti.kernel
    def get_gamma(self, lineindex:TI_INT):
        '''get sum of broadening gamma terms for each line using gamma = bscale * T^a * nh0^b * n_e^c
        for line and broadening term in the broad_field (dims 0 and 1 respectively)'''
        for i in self.gamma:
            nterms = self.atom.broad_scaling.shape[1]
            gamma = self.atom.g_natural[lineindex]

            for term in range(nterms):
            #for each term in atom.broad_scaling sum gamma terms
                a = self.atom.broad_temp_exp[lineindex, term]
                b = self.atom.broad_nh0_exp[lineindex, term]
                c = self.atom.broad_ele_exp[lineindex, term]
                powed = ti.pow(self.temp[i], a) * ti.pow(self.nh0[i], b) * ti.pow(self.ne[i], c)
            
                gamma += self.atom.broad_scaling[lineindex, term] * powed
                self.gamma[i] = gamma

@ti.data_oriented
class Slit:
    def __init__(self, slitres, slitwidth, cen):
        '''setup slit and intensity fields - Slit resolution (slitres), Slit pixel width (slitwidth),
          line profile pixel width (wavdim) model x and y dimensions (xdim ydim) - all model coordinates'''
        self.res = slitres
        self.width = slitwidth
        self.slit_field = ti.field(dtype=vec2, shape=self.res)
        #SETUP DEFAULT SLIT
        self.dir = vec2(1.0,0.0)
        self.cen = vec2(cen[0], cen[1])
        self.set_pos(self.dir, self.cen)
        self.contact = ti.field(dtype=vec3, shape=slitres) #store first ray hit location

    @ti.kernel
    def set_pos(self, dir:vec2, center:vec2):
        '''sets slit with new direction and center'''
        spacing = self.width/self.res
        ndir = tm.normalize(dir)
        sdot = tm.dot(tm.vec2(1,0),ndir)
        angle = tm.acos(sdot)
        #fix as there are multiple places the dot is 0
        if dir[1] < 0:
            angle = 2*tm.pi - tm.acos(sdot)

        rotmat = tm.rotation2d(angle)
            
        for i in self.slit_field:
            #fill points vertically onto a slit around the origin
            y = spacing*i - (self.width/2)
            #rotate to chosen direction
            p = tm.vec2(0.0,y)
            prot = tm.vec2(0.0,0.0)
            prot = rotmat @ p
            #move to slit center location
            startpix = prot + center
            self.slit_field[i] = startpix

@ti.func
def ihighbound(x: float, xbase: ti.template()):
    '''Find input value <x> position on 1D field <xbase> - returns lower bound'''
    imin = TI_INT(0)
    imax = TI_INT(0)
    imax = xbase.shape[0]

    while imin < imax:
        midpt = ti.i32((imax + imin) // 2)

        if x < xbase[midpt]:
            imax = midpt
        else:
            imin = midpt + 1
    return imin

@ti.func
def lerp(x: float, xbase: ti.template(), newbase: ti.template()):
    '''1D interpolation function, x - input value, xbase - current base axis, newbase - base axis to interpolate x onto'''
    xl = xbase.shape[0]
    nl = xbase.shape[0]
    xnew = TI_FP(0.0)

    if x <= xbase[0]:
        xnew = newbase[0]
    else:
        if x >= xbase[xl-1]:
            xnew = newbase[xl-1]
        else:
            ihigh = ihighbound(x, xbase)
            frac = (x - xbase[ihigh-1]) / (xbase[ihigh] - xbase[ihigh-1])
            xnew = tm.mix(newbase[ihigh-1], newbase[ihigh], frac)

    return xnew