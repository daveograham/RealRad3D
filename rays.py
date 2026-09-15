import taichi as ti
import taichi.math as tm
import numpy as np

import taichi as ti
import taichi.math as tm
from taichi.math import vec2, vec3, vec4, ivec2, ivec3, ivec4

TI_ARCH = ti.gpu
TI_FP = ti.f32
TI_INT = ti.i32
PY_FP = np.float32

from modelBuild import ModelSetup
from camera import OrthoCamera
from modelBuild import Slit
from morton import decode_morton_3
from voigt import Voigt
from config import *

@ti.dataclass
class HddaState:
    t: ti.f32
    dt: ti.f32
    step_size: ti.i32
    step_axis: ti.i32
    curr_coord: ivec3
    step: ivec3
    next_hit: vec3
    delta: vec3

    @ti.func
    def compute_axis_and_dt(self: ti.template(), ts):       #ts vec2  t at the start of the ray ( first aabb hit), t at end of array (leaving box t)
        if (self.next_hit[0] <= self.next_hit[1] and self.next_hit[0] <= self.next_hit[2]):
            self.step_axis = 0
        elif (self.next_hit[1] <= self.next_hit[2]):
            self.step_axis = 1
        else:
            self.step_axis = 2

        next_t = self.next_hit[self.step_axis]
        if (next_t <= self.t):
            self.next_hit[self.step_axis] += self.t - 0.999999 * self.next_hit[self.step_axis] + 1e-6
            next_t = self.next_hit[self.step_axis]

        if next_t > ts[1]:
            self.dt = ts[1] - self.t
        else:
            self.dt = next_t - self.t
        self.dt = max(self.dt, 0.0)

    #this one adds onto the t , finds what t is at the next intesection , how far you need to go
    @ti.func
    def next_intersection(self: ti.template(), ts):
        self.t = self.next_hit[self.step_axis]
        self.next_hit[self.step_axis] += self.step_size * self.delta[self.step_axis]
        self.curr_coord[self.step_axis] += self.step_size * self.step[self.step_axis]
        self.compute_axis_and_dt(ts)
        return self.t < ts[1]

@ti.func
def HddaState_init():
    s = HddaState()
    s.t = 0.0
    s.dt = 0.0
    s.step_size = 0
    s.step_axis = 0
    s.curr_coord = ivec3(0)
    s.step = ivec3(0)
    s.next_hit = vec3(0.0)
    s.delta = vec3(0.0)
    return s

RayHit = ti.types.struct(hit=ti.i32, t_range=vec2)
@ti.func
def intersect(bbox, origin, dir) -> RayHit:
    '''Check intersection with bounding box - returns Bool and contact distances
    Takes bounding box, ray origin and direction '''
    check = ti.i32(1)

    tmin = ti.f32(0.0)
    tmax = ti.f32(1.0e24)
    bbox_shrink = ti.f32(1e-4)
    ndim = 3

    for dim in range(ndim):
        if check == 0:
            continue
        inv_dir = 1.0 / dir[dim]
        #shrink bbox a little to avoid rays not being detected on the surface
        lower = 0.0 + bbox_shrink
        upper = bbox[dim] - bbox_shrink
        #entry and exit contact distances a and b
        a = (lower - origin[dim])*inv_dir
        b = (upper - origin[dim])*inv_dir
        #need to swap max and min incase of ray from other direction
        if a > b:
            a, b = b, a
        #closest entry and exit - comparing to previous dimension
        if a > tmin:
            tmin = a
        if b < tmax:
            tmax = b
        #total ray miss
        if tmin > tmax:
            check = ti.i32(0)

    return RayHit(hit=check, t_range=vec2(tmin,tmax))

def get_msaa_samples(sample_count: int):
    # https://vulkan.lunarg.com/doc/view/1.4.313.1/mac/antora/spec/latest/chapters/primsrast.html#primsrast-multisampling
    if sample_count not in [1, 2, 4, 8, 16]:
        raise ValueError("Unknown MSAA support requested")
    if sample_count == 1:
        samples = np.array([
            (0.5, 0.5)
        ])
    elif sample_count == 2:
        samples = np.array([
            (0.75, 0.75),
            (0.25, 0.25),
        ])
    elif sample_count == 4:
        samples = np.array([
            (0.375, 0.125),
            (0.875, 0.375),
            (0.125, 0.625),
            (0.625, 0.875),
        ])
    elif sample_count == 8:
        samples = np.array([
            (0.5625, 0.3125),
            (0.4375, 0.6875),
            (0.8125, 0.5625),
            (0.3125, 0.1875),
            (0.1875, 0.8125),
            (0.0625, 0.4375),
            (0.6875, 0.9375),
            (0.9375, 0.0625),
        ])
    elif sample_count == 16:
        samples = np.array([
            (0.5625, 0.5625),
            (0.4375, 0.3125),
            (0.3125, 0.625),
            (0.75, 0.4375),
            (0.1875, 0.375),
            (0.625, 0.8125),
            (0.8125, 0.6875),
            (0.6875, 0.1875),
            (0.375, 0.875),
            (0.5, 0.0625),
            (0.25, 0.125),
            (0.125, 0.75),
            (0.0, 0.5),
            (0.9375, 0.25),
            (0.875, 0.9375),
            (0.0625, 0.0),
        ])
    return samples.astype(np.float32)

SparseHit = ti.types.struct(sample=vec3, first_hit=vec3)
@ti.data_oriented
class TraceDex3d:
    def __init__(self, ds, model_cache: ModelSetup, camera: OrthoCamera, param_name="temperature", transform_param=None, rt_trace=False):
        if ds.program != "dexrt (3d)":
            print("Program tag does not appear to be \"dexrt (3d)\", are you sure this is the right file?")
        if ds.output_format != "sparse":
            raise ValueError("Expected a sparse file")
        self.block_size = int(ds.block_size)        #morton blocksize
        self.block_stride = self.block_size**3
        self.log2_block_size = int(np.log2(self.block_size))
        self.num_active_tiles = int(ds.num_active_tiles.shape[0])   #number of morton tiles

        self.param_name = param_name
        self.rt_trace = rt_trace

        self.voigt = Voigt()
        self.cam = camera

        if USE_MSAA:
            self.msaa_samples = ti.field(vec2, shape=(camera.supersample,))
            samples = get_msaa_samples(camera.supersample)
            self.msaa_samples.from_numpy(samples.astype(PY_FP))
        else:
            self.msaa_samples = ti.field(vec2, shape=(1,))


        self.num_x = int(ds.num_x)
        self.num_y = int(ds.num_y)
        self.num_z = int(ds.num_z)
        self.aabb = vec3(self.num_x, self.num_y, self.num_z)
        self.voxel_scale = float(ds.voxel_scale)
        self.offset_x = float(ds.offset_x)
        self.offset_y = float(ds.offset_y)
        self.offset_z = float(ds.offset_z)

        self.morton_tiles = ti.field(dtype=ti.u32, shape=(self.num_active_tiles,))
        self.param = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.morton_tiles.from_numpy(ds.morton_tiles.values.astype(PY_FP))

        self.boundary_lookup = model_cache.boundary_lookup
        self.boundary_lookup = model_cache.boundary_lookup
        self.mu_max = model_cache.mu_max[0]
        self.mu_min = model_cache.mu_min[0]

        #FOR VIZUALISATION
        param_data = ds[param_name].values
        if transform_param is not None:
            param_data = transform_param(param_data)
        self.param.from_numpy(param_data.astype(PY_FP))

        self.block_map = ti.field(dtype=ti.i32, shape=(
            self.num_z // self.block_size,
            self.num_y // self.block_size,
            self.num_x // self.block_size
        ))
        self.setup_block_map()

        self.model_cache = model_cache

        self.lineidx = model_cache.lineindex
        self.const = model_cache.const
        self.atom =  model_cache.atom
        self.restwav = model_cache.restwav
        #voigt profile container
        self.wofz = ti.field(dtype=tm.vec2, shape=model_cache.wdim)
        self.wofz_a = ti.field(dtype=tm.vec2, shape=model_cache.wdim)
        self.eta = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.chi = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.vproj = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))

    #DIRECTION DEPENDENT
    @ti.kernel
    def get_v_projected(self, ray: vec3, model_cache: ti.template()):
        for i in self.vproj:
            #flipped as data ordered zyx
            vmodel = vec3(model_cache.vx[i], model_cache.vy[i], model_cache.vz[i])
            self.vproj[i] = tm.dot(vmodel, -ray)/tm.length(ray)

    @ti.kernel
    def get_emiss_opac(self, wix: TI_INT, model_cache: ti.template()):
        '''Populate eta_field and chi_field per wavelength index (wix)'''
        lindex = self.lineidx

        for i in self.eta:
            #returns opacity and emissivity - eta&chi - for the model atmosphere
            wdop = model_cache.wdop[i]
            wav_in = model_cache.wav_profile[wix]

            a = self.damping_from_gamma(wav_in, wdop, model_cache.gamma[i])
            v = self.v_for_voigt(wav_in, wdop, self.restwav, self.vproj[i])
            #find absorption profile and write to kernel
            phi = self.voigt.voigtCalc(tm.vec2(v,a))
            self.wofz[wix] = phi
            #phi = psi here because CRD - 1e-3 converts h to kW - 1e-9 converts wavlength to m
            C = (self.const.h * self.const.c * 1e-3) / (4.0 * tm.pi * wav_in * 1e-9)
            #normalize voigt profile
            cnorm = ti.rsqrt(tm.pi) / wdop
            #from Uitenbroek 2001
            Vij = C * self.atom.line_Bij[lindex] * phi[0] * cnorm
            Vji = C * self.atom.line_Bji[lindex] * phi[0] * cnorm
            Uji = C * self.atom.line_Aji[lindex] * phi[0] * cnorm

            upper = model_cache.atom.line_j[lindex]
            lower = model_cache.atom.line_i[lindex]
            
            self.chi[i] = model_cache.pops[lower,i]*Vij - model_cache.pops[upper,i]*Vji
            self.eta[i] = model_cache.pops[upper,i]*Uji

    @ti.func
    def damping_from_gamma(self, wav: TI_FP, dopplerwidth: TI_FP, gammatot: TI_FP):
        '''take gamma and turn it into damping parameter a in the voigt profile'''
        #1e-9 to convert c to nm
        b = ti.pow(wav, 2.0) * 1e-9
        d = (4.0 * tm.pi * self.const.c * dopplerwidth)
        a = b / d
        return a * gammatot

    @ti.func
    def v_for_voigt(self, wav: TI_FP, dopplerwidth: TI_FP, restwav: TI_FP, modelvel: TI_FP):
        '''return v for voigt profile in units per doppler width'''
        #convert wavelength from nm to a scaled parameter v
        return ((wav - restwav) + (modelvel * restwav)/self.const.c) / dopplerwidth

    def setup_block_map(self):
        self.block_map.fill(-1)
        self.setup_block_map_kernel()

    @ti.kernel
    def setup_block_map_kernel(self):
        for tile_idx in self.morton_tiles:
            code = self.morton_tiles[tile_idx]
            block_coord = decode_morton_3(code)
            self.block_map[block_coord.z, block_coord.y, block_coord.x] = tile_idx   #record which tile each coordinate maps to

    @staticmethod
    @ti.func
    def transfer_function(param) -> vec3:
        return TRANSFER_BASE_COLOR * ti.exp(-(TRANSFER_CENTRE_VAL - param)**2 / TRANSFER_CENTRE_WIDTH)

    @ti.func
    def naive_sparse_trace(self, o: vec3, d: vec3, ts: vec2, back: TI_FP = 0.0, step_size: ti.f32 = 0.2) -> vec3:
        t = ts[0]

        result = vec3(0.0)
        tau_running = 0.0
        s = 0.0

        while t < ts[1]:
            sample_pos = o + t * d
            vox = ti.cast(sample_pos, ti.i32)                  #integer voxel position
            block_coord = vox >> ivec3(self.log2_block_size)   #'divide by blocksize' bitwise right shift?
            block_idx = -1
            if (
                ti.cast(block_coord.z, ti.u32) < self.block_map.shape[0] and
                ti.cast(block_coord.y, ti.u32) < self.block_map.shape[1] and
                ti.cast(block_coord.x, ti.u32) < self.block_map.shape[2]
            ):
                block_idx = self.block_map[block_coord.z, block_coord.y, block_coord.x]
            if block_idx != -1:
                inner_coord = vox - block_coord * self.block_size   #coord within block
                ks = block_idx * self.block_stride + ((inner_coord.z * self.block_size + inner_coord.y) * self.block_size + inner_coord.x)
                param = self.param[ks]

                if self.rt_trace == True:
                    tscaled = self.voxel_scale * step_size
                    chi = self.chi[ks]
                    eta = self.eta[ks]

                    if chi > 0:
                        s = eta / chi
                    else:
                        s = 0.0
                    tau = chi * tscaled
                    result += tm.exp(-1.0*tau_running) * s * (1 - tm.exp(-1.0*tau))
                    tau_running += tau
                else:
                    result += self.transfer_function(param) * self.voxel_scale * step_size
            t += step_size
        result += back * tm.exp(-1.0*tau_running)
        return result
    
    @ti.func
    def naive_sparse_trace_mod(self, o: vec3, d: vec3, ts: vec2, back: TI_FP = 0.0, step_size: ti.f32 = 0.2) -> SparseHit:
        t = ts[0]

        result = vec3(0.0)
        hit0 = vec3(0.0)
        tau_running = 0.0
        s = 0.0

        while t < ts[1]:
            sample_pos = o + t * d
            vox = ti.cast(sample_pos, ti.i32)                  #integer voxel position
            block_coord = vox >> ivec3(self.log2_block_size)
            block_idx = -1
            if (
                ti.cast(block_coord.z, ti.u32) < self.block_map.shape[0] and
                ti.cast(block_coord.y, ti.u32) < self.block_map.shape[1] and
                ti.cast(block_coord.x, ti.u32) < self.block_map.shape[2]
            ):
                block_idx = self.block_map[block_coord.z, block_coord.y, block_coord.x]
            if block_idx != -1:
                if tau_running == 0.0:  #store the first hit location option
                    hit0 = sample_pos
                inner_coord = vox - block_coord * self.block_size   #coord within block
                ks = block_idx * self.block_stride + ((inner_coord.z * self.block_size + inner_coord.y) * self.block_size + inner_coord.x)
                param = self.param[ks]

                if self.rt_trace == True:
                    tscaled = self.voxel_scale * step_size
                    chi = self.chi[ks]
                    eta = self.eta[ks]

                    if chi > 0:
                        s = eta / chi
                    else:
                        s = 0.0
                    tau = chi * tscaled
                    result += tm.exp(-1.0*tau_running) * s * (1 - tm.exp(-1.0*tau))
                    tau_running += tau
                else:
                    result += self.transfer_function(param) * self.voxel_scale * step_size
            t += step_size
        result += back * tm.exp(-1.0*tau_running)
        return SparseHit(sample=result, first_hit=hit0)

    @ti.func
    def has_data(self, coord: ivec3) -> ti.i32:
        block_coord = coord >> ivec3(self.log2_block_size)
        block_idx = -1
        if (
            ti.cast(block_coord.z, ti.u32) < self.block_map.shape[0] and
            ti.cast(block_coord.y, ti.u32) < self.block_map.shape[1] and
            ti.cast(block_coord.x, ti.u32) < self.block_map.shape[2]
        ):
            block_idx = self.block_map[block_coord.z, block_coord.y, block_coord.x]
        return block_idx != -1

    @ti.func
    def step_through_grid(self, s: ti.template(), o, d, ts, inv_d):
        result = 0
        while not result and s.next_intersection(ts):
            has_data = self.has_data(s.curr_coord)

            if has_data and s.step_size == 1:
                result = 1
                break   #all good go do physics

            if not has_data and s.step_size != self.block_size:
                s.t += 0.01  #nudge the ray
            s.step_size = 1 if has_data else self.block_size
            curr_pos = o + s.t * d
            new_coord = ti.cast(curr_pos, ti.i32)
            s.curr_coord = new_coord & (~(s.step_size - 1))

            for ax in range(3):
                if s.step[ax] == 0:
                    continue

                s.next_hit[ax] = s.t + (s.curr_coord[ax] - curr_pos[ax]) * inv_d[ax]
                if s.step[ax] > 0:
                    s.next_hit[ax] += s.step_size * inv_d[ax]
            s.compute_axis_and_dt(ts)
            result = has_data

        return result

    @ti.func
    def two_level_hdda_trace(self, o: vec3, d: vec3, ts: vec2) -> vec3:
        s = HddaState_init()
        s.t = ts[0]

        start_pos = o + s.t * d
        s.curr_coord = ti.cast(start_pos, ti.i32)
        s.step_size = 1
        if not self.has_data(s.curr_coord):
            s.step_size = self.block_size
        s.curr_coord &= (~(s.step_size - 1))

        inv_d = 1.0 / d
        for ax in range(3):
            if d[ax] == 0.0:
                s.step[ax] = 0
                s.next_hit[ax] = 1e24
            elif (inv_d[ax] > 0.0):
                s.step[ax] = 1
                s.next_hit[ax] = s.t + (s.curr_coord[ax] + s.step_size - start_pos[ax]) * inv_d[ax]
                s.delta[ax] = inv_d[ax]
            else:
                s.step[ax] = -1
                s.next_hit[ax] = s.t + (s.curr_coord[ax] - start_pos[ax]) * inv_d[ax]
                s.delta[ax] = -inv_d[ax]
        s.compute_axis_and_dt(ts)
        #now this is the tracing loop
        result = vec3(0.0)
        while True: #this is one loop...
            if self.has_data(s.curr_coord):
                block_coord = s.curr_coord >> ivec3(self.log2_block_size)
                block_idx = self.block_map[block_coord.z, block_coord.y, block_coord.x]
                inner_coord = s.curr_coord - block_coord * self.block_size
                ks = block_idx * self.block_stride + ((inner_coord.z * self.block_size + inner_coord.y) * self.block_size + inner_coord.x)
                param = self.param[ks]
                # Replace with RTE
                result += self.transfer_function(param) * self.voxel_scale * s.dt
            if not self.step_through_grid(s, o, d, ts, inv_d):  #this is another loop...
                break
        return result


    @ti.kernel
    def trace(self, buffer:ti.template(), supersample: int, corner: vec3, right: vec3, up: vec3, forward: vec3, step_size: ti.f32, waveindex:TI_FP, outchannel:TI_INT):
        num_aa_samples = self.cam.supersample if USE_MSAA else supersample**2

        for I in ti.grouped(buffer):
        #for u, v in buffer << now I.x I.y
            radiance = vec3(0.0)
            ray_start_uvw = vec3(0.0,0.0,0.0)

            ti.loop_config(serialize=True)
            for sub_ray in range(num_aa_samples):
                u_ray_offset, v_ray_offset = 0.5, 0.5
                if USE_MSAA:
                    u_ray_offset = self.msaa_samples[sub_ray][0]
                    v_ray_offset = self.msaa_samples[sub_ray][1]
                else:
                    u_ray = sub_ray // supersample
                    v_ray = sub_ray - u_ray * supersample
                    u_ray_offset = (u_ray + 0.5) / ti.f32(supersample)
                    v_ray_offset = (v_ray + 0.5) / ti.f32(supersample)

                ray_start_uvw = corner + right * (I.x + u_ray_offset) + up * (I.y + v_ray_offset)
                
                ray_hit = intersect(self.aabb, origin=ray_start_uvw, dir=forward)
                sample_color = vec3(0.0)
                #test for boundary check
                isurface = 0.0
                mu = self.surfacehit(ray_start_uvw, forward)
                if mu > -1.0 and forward.z < 0:
                    wavebin = self.model_cache.wav_to_lookup_index[int(waveindex)]
                    isurface = self.surface_lookup(mu, wavebin)

                if ray_hit.hit:
                    if USE_HDDA:
                        sample_color = self.two_level_hdda_trace(
                            o=ray_start_uvw,
                            d=forward,
                            ts=ray_hit.t_range,
                        )
                    else:
                        sample_color = self.naive_sparse_trace(
                            o=ray_start_uvw,
                            d=forward,
                            ts=ray_hit.t_range,
                            back=isurface,
                            step_size=step_size,
                        )
                else:
                    radiance += isurface #has to be outside of the model box hit check
                radiance += sample_color
            #buffer has RGB channels
            if outchannel > -1:
                buffer[I][outchannel] = radiance[outchannel] / num_aa_samples
            else:
                buffer[I] = radiance / num_aa_samples


    @ti.kernel
    def trace_slit(self, buffer:ti.template(), slit:ti.template(), supersample: int, corner: vec3, right: vec3, up: vec3, forward: vec3, step_size: ti.f32, waveindex: vec3, storefirst: TI_INT):
        num_aa_samples = self.cam.supersample if USE_MSAA else supersample**2
        sdim = slit.slit_field.shape[0]
        for i in slit.slit_field:
            radiance = vec3(0.0)
            ray_start_uvw = vec3(0.0,0.0,0.0)
            dhit = 1e10
            ti.loop_config(serialize=True)  #supersample loop
            for sub_ray in range(num_aa_samples):
                u_ray_offset, v_ray_offset = 0.5, 0.5
                if USE_MSAA:
                    u_ray_offset = self.msaa_samples[sub_ray][0]
                    v_ray_offset = self.msaa_samples[sub_ray][1]
                else:
                    u_ray = sub_ray // supersample
                    v_ray = sub_ray - u_ray * supersample
                    u_ray_offset = (u_ray + 0.5) / ti.f32(supersample)
                    v_ray_offset = (v_ray + 0.5) / ti.f32(supersample)

                spos = slit.slit_field[i]
                ray_start_uvw = corner + right * (spos[0] + u_ray_offset) + up * (spos[1] + v_ray_offset)
                
                ray_hit = intersect(self.aabb, origin=ray_start_uvw, dir=forward)
                sample_color = vec3(0.0)

                isurface = 0.0
                mu = self.surfacehit(ray_start_uvw, forward)
                if mu > -1.0 and forward.z < 0:
                    wavebin = self.model_cache.wav_to_lookup_index[int(waveindex[1])]
                    isurface = self.surface_lookup(mu, wavebin)

                if ray_hit.hit:
                    if USE_HDDA:
                        sample_color = self.two_level_hdda_trace(
                            o=ray_start_uvw,
                            d=forward,
                            ts=ray_hit.t_range,
                        )
                    else:
                        sample_struct = self.naive_sparse_trace_mod(
                            o=ray_start_uvw,
                            d=forward,
                            ts=ray_hit.t_range,
                            back=isurface,
                            step_size=step_size,
                        )
                        sample_color = sample_struct.sample
                        #for first contact
                        if storefirst == 1:
                            dtest = tm.length(ray_start_uvw - sample_struct.first_hit)
                            if dtest < dhit:
                                slit.contact[i] = sample_struct.first_hit
                                dtest = dhit
                else:
                    radiance += isurface
                radiance += sample_color
            #fill buffer the same as wavestep and slit pixel dimensions
            sx = int(waveindex[1])
            buffer[sx, i] = radiance / num_aa_samples

    def trace_for_camera(self, cam: OrthoCamera, slit: Slit, waveindex: vec4, step_size=0.2, mode=0):
        if not hasattr(self, 'fb') or self.fb.shape != (self.cam.x_size, self.cam.y_size):
            self.fb = ti.field(vec3, shape=(self.cam.x_size, self.cam.y_size))    #frame buffer
            self.wb = ti.field(vec3, shape=(WAVESTEPS, SLIT_RES))       #spectrum buffer

        #THIS IS WHERE TO DO CAMERA DEPENDENT CALC FOR ETA AND CHI
        self.get_v_projected(self.cam.forward_ray, self.model_cache)
        #range of three wavelengths indexes === taken from WINGDIST
        #going to keep this RED CENTRE BLUE because of RGB order
        colorsample = ivec3(int(CENWAV+waveindex[3]), int(CENWAV), int(CENWAV-waveindex[3]))
        #for the image
        if mode == 0 or mode == 2:
            self.get_emiss_opac(colorsample[int(waveindex[2])], self.model_cache)
            self.trace(
                supersample=self.cam.supersample,
                buffer=self.fb,
                corner=self.cam.corner,
                right=self.cam.right_ray,
                up=self.cam.up_ray,
                forward=self.cam.forward_ray,
                step_size=step_size,
                waveindex=colorsample[int(waveindex[2])],
                outchannel=int(waveindex[2]),
            )
        if mode == 1 or mode == 2:
            #for the slit window
            self.get_emiss_opac(int(waveindex[1]), self.model_cache)
            self.trace_slit(
                supersample=self.cam.supersample,
                slit=slit,
                buffer=self.wb,
                corner=self.cam.corner,
                right=self.cam.right_ray,
                up=self.cam.up_ray,
                forward=self.cam.forward_ray,
                step_size=step_size,
                waveindex=waveindex,
                storefirst=0
            )
        if mode == 3:
            #for the slit window
            self.get_emiss_opac(int(waveindex[1]), self.model_cache)
            self.trace_slit(
                supersample=self.cam.supersample,
                slit=slit,
                buffer=self.wb,
                corner=self.cam.corner,
                right=self.cam.right_ray,
                up=self.cam.up_ray,
                forward=self.cam.forward_ray,
                step_size=step_size,
                waveindex=waveindex,
                storefirst=1
            )

    @ti.func
    def surfacehit(self, origin:vec3, raydir:vec3):
        '''checks hit with a solar sphere and returns mu angle if hit, -1 if no hit'''
        mu = -1.0
        hit = 0
        #ray start in world coords
        p = vec3((origin[0] + 0.5) * self.voxel_scale + self.offset_z,
                 (origin[1] + 0.5) * self.voxel_scale + self.offset_y,
                 (origin[2] + 0.5) * self.voxel_scale + self.offset_x
                 )
        P_cen = vec3(0.0,0.0,0.0)
        #in units of solar radius
        Rsolar = ti.f32(695700000.0)
        #add one radius to the z so it's a distance above the center
        rayorigin = p/Rsolar
        rayorigin.z += 1.0
        cq = (P_cen - rayorigin)
        #solve the quadratic eqn (rsun = 1.0)
        a = raydir @ raydir
        b = -2.0 * (raydir @ cq)
        c = (cq @ cq) - 1.0
        delta = (b*b) - (4.0 * a * c)

        if delta <= 0:
            hit = 0
        else:
            hit = 1
            #find the hit distance and mu
            t_hit = ((-1.0 * b) + delta)/(2.0 * a)
            P_hit = rayorigin + t_hit * raydir
            #ray and surface normal (at hit) mu_prime
            rayN = tm.normalize(raydir)
            surfaceN = tm.normalize(P_hit - P_cen)
            mu = surfaceN @ rayN
            #mu between ray hit location and disc centre
            Rhit = tm.normalize(rayorigin - P_hit)
            Rcen = tm.normalize(rayorigin - P_cen)
            mu2 = Rhit @ Rcen
        return mu
    
    @ti.func
    def surface_lookup(self, mu:TI_FP, w_frac:TI_FP):
        '''take mu from surfacehit and lookup surface background intensity (w_frac) at given wavelength bin
        Takes GlobalVariables'''
        i_interp = 0.0

        wav_base = self.boundary_lookup.shape[0]
        mu_base = self.boundary_lookup.shape[1]
        #fractional index on the mu array
        mu_frac = (mu / self.mu_max) * (mu_base-1)
        
        mx = ti.int32(mu_frac)
        mweight = mu_frac - mx

        wx = ti.int32(w_frac)
        wweight = w_frac - wx

        #get the fractional intensity between the lambda points
        if mu_frac < (mu_base-1) and w_frac < (wav_base-1):
            #print("normal")
            i1 = self.boundary_lookup[wx, mx]
            i2 = self.boundary_lookup[wx+1, mx]
            ilow = tm.mix(i1, i2, wweight)

            i3 = self.boundary_lookup[wx, mx+1]   
            i4 = self.boundary_lookup[wx+1, mx+1]
            ihigh = tm.mix(i3,i4,wweight)

            i_interp = tm.mix(ilow, ihigh, mweight)
        else:
            if mu_frac >= (mu_base-1) and w_frac < (wav_base-1):
                i_interp = tm.mix(self.boundary_lookup[wx,mx], self.boundary_lookup[wx+1, mx], wweight)
            
            if mu_frac < (mu_base-1) and w_frac >= (wav_base-1):
                i_interp = tm.mix(self.boundary_lookup[wx,mx], self.boundary_lookup[wx, mx+1], wweight)

            if mu_frac >= (mu_base-1) and w_frac >= (wav_base-1):
                i_interp = self.boundary_lookup[wx,mx]
            
        return i_interp

    @ti.kernel
    def rotation_to_screen(self, slit:ti.template(), dtheta:TI_FP, pix:TI_INT) -> TI_FP:
        '''move slit with the sticky pixel'''
        ds = 0.0
        point = slit.contact[pix]
        if tm.length(point) > 0.0:
            modcen = self.aabb // 2
            pr = tm.length(point-modcen)
            ds = pr * ti.sin(dtheta) * (WINDOW_SIZE[0] /self.num_y)
            dscreen = (WINDOW_SIZE[0] /self.num_y)*ds
        return ds