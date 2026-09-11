import taichi as ti
import taichi.math as tm
import numpy as np
from taichi.math import vec2, vec3, vec4, ivec2, ivec3, ivec4

TI_ARCH = ti.gpu
TI_FP = ti.f32
TI_INT = ti.i32
PY_FP = np.float32

@ti.func
def hfit(value: TI_FP, omin: TI_FP, omax: TI_FP, nmin: TI_FP, nmax: TI_FP):
    '''Houdini style fit function - remap a value from one range to another
    value - input
    omin/omax - original range
    nmin/nmax - new range'''
    norm = tm.clamp((value - omin) / (omax - omin), 0.0, 1.0)
    return (1-norm)*nmin + norm*nmax

@ti.func
def tonemap(x: vec3) -> vec3:
    """Uncharted2 tonemap"""
    A = 0.15
    B = 0.5
    C = 0.1
    D = 0.2
    E = 0.02
    F = 0.3

    return ((x * (A*x + C*B) + D*E) / (x * (A*x + B) + D*F)) - E/F

def tonemap_and_blit(trace_buffer, bias, white_point, gamma, blur, blur_buffer, blur_kernel, gui_buffer, VIEWER_AABB):
    if blur > 1:
        compute_blur_kernel(blur, blur_kernel)
        blur_pass_horizontal(trace_buffer, blur_buffer, blur_kernel)
        blur_pass_vertical(blur_buffer, trace_buffer, blur_kernel)
    tonemap_buffer(trace_buffer, bias, white_point, gamma)
    #if trace/cam buffer is the same size as window buffer copy, or interpolate if not
    if gui_buffer.shape == trace_buffer.shape:
        blit(trace_buffer)
    else:
        bilinear_offset(trace_buffer, gui_buffer, VIEWER_AABB)  #interpolation onto the gui viewer box

@ti.kernel
def tonemap_buffer(trace_buffer: ti.template(), bias: ti.f32, white_scale: ti.f32, gamma:ti.f32):
    for i, j in trace_buffer:
        sample = (tonemap(bias * trace_buffer[i, j]) * white_scale)**(1.0 / gamma)
        trace_buffer[i, j] = sample

@ti.kernel
def blit(trace_buffer: ti.template(), gui_buffer:ti.template()):
    for u, v in trace_buffer:
        gui_buffer[u, v] = trace_buffer[u, v]

@ti.kernel
def bilinear_copy(trace_buffer: ti.template()):
    # NOTE(cmo): Only uses height ratio
    ratio = trace_buffer.shape[1] / gui_buffer.shape[1]
    for u, v in gui_buffer:
        uprime = u * ratio
        vprime = v * ratio
        if uprime >= trace_buffer.shape[0] or vprime >= trace_buffer.shape[1]:
            gui_buffer[u, v] = vec3(0.0)
            continue
        uv = vec2(uprime, vprime)
        corner = ti.cast(ti.math.floor(uv), ti.i32)
        bilin_ratio = ti.math.fract(uv)
        weights = vec4(
            (1.0 - bilin_ratio.x) * (1.0 - bilin_ratio.y),
            (1.0 - bilin_ratio.x) * bilin_ratio.y,
            bilin_ratio.x * (1.0 - bilin_ratio.y),
            bilin_ratio.x * bilin_ratio.y,
        )
        result = vec3(0.0)
        for p in range(4):
            py = p // 2
            px = p - py * 2
            result += weights[p] * trace_buffer[
                ti.math.min(corner[0] + px, trace_buffer.shape[0]-1),
                ti.math.min(corner[1] + py, trace_buffer.shape[1]-1),
            ]
        gui_buffer[u, v] = result

@ti.kernel
def bilinear_offset(trace_buffer: ti.template(), gui_buffer: ti.template(), corners: ti.template()):
    # NOTE(cmo): Only uses height ratio
    #ratio = trace_buffer.shape[1] / gui_buffer.shape[1]
    # DG version with offset
    #for u, v in gui_buffer:
    subx = corners[2]-corners[0]
    suby = corners[3]-corners[1]
    ratio = trace_buffer.shape[1] / suby
    for u, v in ti.ndrange(subx, suby):
        uprime = u * ratio
        vprime = v * ratio
        if uprime >= trace_buffer.shape[0] or vprime >= trace_buffer.shape[1]:
            gui_buffer[u, v] = vec3(0.0)
            continue
        uv = vec2(uprime, vprime)
        corner = ti.cast(ti.math.floor(uv), ti.i32)
        bilin_ratio = ti.math.fract(uv)
        weights = vec4(
            (1.0 - bilin_ratio.x) * (1.0 - bilin_ratio.y),
            (1.0 - bilin_ratio.x) * bilin_ratio.y,
            bilin_ratio.x * (1.0 - bilin_ratio.y),
            bilin_ratio.x * bilin_ratio.y,
        )
        result = vec3(0.0)
        for p in range(4):
            py = p // 2
            px = p - py * 2
            result += weights[p] * trace_buffer[
                ti.math.min(corner[0] + px, trace_buffer.shape[0]-1),
                ti.math.min(corner[1] + py, trace_buffer.shape[1]-1),
            ]
        gui_buffer[u+corners[0], v+corners[1]] = result

@ti.kernel
def compute_blur_kernel(fwhm: ti.f32, blur_kernel:ti.template()):
    sigma = fwhm / 2.35482
    mu = blur_kernel.shape[0] // 2
    for i in blur_kernel:
        blur_kernel[i] = 1.0 / (sigma * ti.sqrt(2 * ti.math.pi)) * ti.exp(-0.5 * (i - mu)**2 / sigma**2)

@ti.kernel
def blur_pass_horizontal(buffer: ti.template(), out_buffer: ti.template(), blur_kernel:ti.template()):
    offset = blur_kernel.shape[0] // 2
    for u, v in buffer:
        out_buffer[u, v] = 0.0
        for uu in range(blur_kernel.shape[0]):
            u_idx = u - offset + uu
            if u_idx < 0 or u_idx >= buffer.shape[0]:
                continue
            out_buffer[u, v] += blur_kernel[uu] * buffer[u_idx, v]

@ti.kernel
def blur_pass_vertical(buffer: ti.template(), out_buffer: ti.template(), blur_kernel:ti.template()):
    offset = blur_kernel.shape[0] // 2
    for u, v in buffer:
        out_buffer[u, v] = 0.0
        for vv in range(blur_kernel.shape[0]):
            v_idx = v - offset + vv
            if v_idx < 0 or v_idx >= buffer.shape[1]:
                continue
            out_buffer[u, v] += blur_kernel[vv] * buffer[u, v_idx]

@ti.kernel
def win2cam(x: TI_FP, y: TI_FP, VIEWER_SIZE:ti.template(), CAM_SIZE:ti.template()) -> vec2:
    xout = hfit(x, 0, VIEWER_SIZE[0], 0, CAM_SIZE[0])
    yout = hfit(y, 0, VIEWER_SIZE[1], 0, CAM_SIZE[1])
    return vec2(xout,yout)

@ti.kernel
def cam2win(x: TI_FP, y: TI_FP, VIEWER_SIZE:ti.template(), CAM_SIZE:ti.template()) -> vec2:
    xout = hfit(x, 0, CAM_SIZE[0], 0, VIEWER_SIZE[0])
    yout = hfit(y, 0, CAM_SIZE[1], 0, VIEWER_SIZE[1])
    return vec2(xout,yout)

@ti.kernel
def slit_to_buffer(trace_buffer: ti.template(), slit_buffer: ti.template(), slit_aabb: vec4):
    for u, v in trace_buffer:
        if u > slit_aabb[0] and u < slit_aabb[2]:
            if v > slit_aabb[1] and v < slit_aabb[3]:
                trace_buffer[u,v] = slit_buffer[u,v]

@ti.kernel
def spectrum_to_frame_buffer(buffer_in:ti.template(), buffer_out:ti.template(), aabb:vec4, bias:ti.f32, white_scale:ti.f32, gamma:ti.f32):
    '''Rescale spec plot window from natural resolution to output buffer - do scaling from data to colourtable'''
    #plot location corners on the out buffer
    lx = aabb[0]
    ly = aabb[1]
    ux = aabb[2]
    uy = aabb[3]

    xdim_in = buffer_in.shape[0]
    ydim_in = buffer_in.shape[1]
    #sample spectrum buffer and place it on output buffer
    #sample from incoming buffer at pixel interpolated from the outgoing
    for ii,jj in buffer_out:
        if ii >= lx and ii < ux:
            if jj >= ly and jj < uy:
                iin_f = hfit(ii-lx, 0, ux-lx, 0, xdim_in)
                jjn_f = hfit(jj-ly, 0, uy-ly, 0, ydim_in)
                
                iin = ti.cast(iin_f,dtype=ti.int32)
                jjn = ti.cast(jjn_f,dtype=ti.int32)
                
                samplepix = buffer_in[iin,jjn]
                #tone map has to happen in here
                scaled = (tonemap(bias * samplepix) * white_scale)**(1.0 / gamma)
                buffer_out[ii,jj] = scaled


@ti.kernel
def draw_slit(slit:ti.template(), slitdir:tm.vec2, buffer:ti.template(), window_aabb:vec4):
    '''Plot the intensity along the slit position - Run create slit first - draws on buffer field'''
    midindex = tm.floor(slit.slit_field.shape[0]/2.0)
    ndir = tm.normalize(slitdir)
    for i in slit.slit_field:
        getslit = slit.slit_field[i]
        pviz = getslit + ndir
        #slit position to pixel number
        xpix = tm.floor(window_aabb[0]+pviz.x,dtype=TI_INT)
        ypix = tm.floor(window_aabb[1]+pviz.y,dtype=TI_INT)

        bufferx = buffer.shape[0]
        buffery = buffer.shape[1]

        if xpix < bufferx-1 and xpix > 1:
            if ypix < buffery-5 and ypix > 5:
                buffer[xpix, ypix-1] = vec3(0.0,0.5,1.0)

@ti.kernel
def draw_slit_gui(slit:ti.template(), slitdir:tm.vec2, buffer:ti.template(), viewer_aabb:vec4, CAM_SIZE:ti.template(), VIEWER_SIZE:ti.template()):
    '''Plot the intensity along the slit position - Run create slit first - draws on buffer field'''
    midindex = tm.floor(slit.slit_field.shape[0]/2.0)
    ndir = tm.normalize(slitdir)
    for i in slit.slit_field:
        getslit = slit.slit_field[i]

        xin = hfit(getslit.x, 0, CAM_SIZE[0], 0, VIEWER_SIZE[0])
        yin = hfit(getslit.y, 0, CAM_SIZE[1], 0, VIEWER_SIZE[1])
        pos = vec2(xin,yin)

        pviz = pos + ndir
        #slit position on gui buffer
        xpix = tm.floor(viewer_aabb[0]+pviz.x,dtype=TI_INT)
        ypix = tm.floor(viewer_aabb[1]+pviz.y,dtype=TI_INT)

        bufferx = buffer.shape[0]
        buffery = buffer.shape[1]

        if xpix < bufferx-1 and xpix > 1:
            if ypix < buffery-5 and ypix > 5:
                for a in range(3):
                    buffer[xpix, ypix-1+a] = vec3(0.0,0.5,1.0)

@ti.kernel
def draw_wing_markers(wavecenter: TI_INT, WAVESTEPS:TI_INT, wingidx: TI_INT, buffer:ti.template(), window_aabb:vec4):
    marks = vec3(wavecenter-wingidx, wavecenter, wavecenter+wingidx) #on slit buffer
    #shift to spectra box coordinates
    left = ti.cast(hfit(wavecenter-wingidx, 0, WAVESTEPS, 0, window_aabb[2]),dtype=TI_INT)
    center = ti.cast(hfit(wavecenter, 0, WAVESTEPS, 0, window_aabb[2]),dtype=TI_INT)
    right = ti.cast(hfit(wavecenter+wingidx, 0, WAVESTEPS, 0, window_aabb[2]),dtype=TI_INT)

    for i,j in buffer:
        dotat = j % 5
        xpix = i+ti.cast(window_aabb[0],dtype=TI_INT)
        ypix = j+ti.cast(window_aabb[1],dtype=TI_INT)
        if xpix == left or xpix == right:
            if ypix < window_aabb[3]-40 and ypix > window_aabb[1]+40:
                if dotat == 0:
                    for b in ti.ndrange(3): #for a cross
                        buffer[xpix-1+b,ypix] = vec3(0.0, 0.7, 0.3)
                        buffer[xpix,ypix-1+b] = vec3(0.0, 0.7, 0.3)
                    
        if xpix == center:
            if ypix < window_aabb[3]-20 and ypix > window_aabb[1]+20:
                if dotat == 0:
                    for b in ti.ndrange(3): #for a cross
                        buffer[xpix-1+b,ypix] = vec3(0.0, 1.0, 0.0)
                        buffer[xpix,ypix-1+b] = vec3(0.0, 1.0, 0.0)

                    

@ti.kernel
def click_markers(loc: vec2, buffer:ti.template(), WINDOW_SIZE:ti.template()):
    xpix = ti.cast(loc.x*WINDOW_SIZE[0],dtype=TI_INT)+1
    ypix = ti.cast(loc.y*WINDOW_SIZE[1],dtype=TI_INT)
    wid = 7
    for b in ti.ndrange(wid): #for a cross
        offset = wid // 2
        buffer[xpix-offset+b,ypix] = vec3(1.0, 1.0, 1.0)
        buffer[xpix,ypix-offset+b] = vec3(1.0, 1.0, 1.0)

@ti.kernel
def clear_buffer(buffer_out:ti.template(), corners:ivec4, overflow:TI_INT):
    '''clear an area on a buffer within an AABB boundary and some overflow'''
    lx = corners[0]
    ly = corners[1]
    ux = corners[2]
    uy = corners[3]
    subu = ux-lx
    subv = uy-ly
    for u,v in ti.ndrange(subu,subv):
        uwin = ti.cast(u+lx, dtype=TI_INT)
        vwin = ti.cast(v+ly, dtype=TI_INT)
        buffer_out[uwin,vwin] = vec3(0.0)
        buffer_out[uwin+overflow,vwin+overflow] = vec3(0.0)
        buffer_out[uwin-overflow,vwin-overflow] = vec3(0.0)


@ti.kernel
def draw_spectra_segment(buffer_in:ti.template(), buffer_out:ti.template(), pos: vec2, corners:ivec4, imin:TI_FP, imax:TI_FP, pstart:TI_INT) -> vec4:
    ''' draw a single line segment of the spectrum plot, returns start and end coord'''
    lx = corners[0]
    ly = corners[1]
    ux = corners[2]
    uy = corners[3]
    subu = ux-lx
    subv = uy-ly

    ldim = buffer_in.shape[0] #WAVESTEPS
    ydim = buffer_in.shape[1] #SLIT_RES
    xdim = buffer_out.shape[0]
    ydim = buffer_out.shape[1]

    samplepair = vec2(0.0)
    upair = vec2(0.0)
    vpair = vec2(0.0)

    samplepair[0] = buffer_in[pstart, int(pos.y)][1]
    samplepair[1] = buffer_in[pstart+1, int(pos.y)][1]

    upair[0] = ti.cast((lx + hfit(pstart, 0, ldim, 0, subu)), dtype=TI_INT)
    upair[1] = ti.cast((lx + hfit(pstart+1, 0, ldim, 0, subu)), dtype=TI_INT)
    
    vpair[0] = ti.cast(ly + hfit(samplepair[0], imin, imax, 0.0, subv), dtype=TI_INT)
    vpair[1] = ti.cast(ly + hfit(samplepair[1], imin, imax, 0.0, subv), dtype=TI_INT)
    
    return vec4(upair[0]/xdim, vpair[0]/ydim, upair[1]/xdim, vpair[1]/ydim)
            
@ti.kernel
def draw_spectra_plot(buffer_in:ti.template(), buffer_out:ti.template(), spec:ti.template(), pos: vec2, corners:ivec4, imin:TI_FP, imax:TI_FP, ipasses:TI_INT):
    ''' take a cam buffer index and plot the spectrum to the GUI'''
    lx = corners[0]
    ly = corners[1]
    ux = corners[2]
    uy = corners[3]
    subu = ux-lx
    subv = uy-ly

    ldim = buffer_in.shape[0] #WAVESTEPS
    ydim = buffer_in.shape[1] #SLIT_RES
    xdim = buffer_out.shape[0]
    ydim = buffer_out.shape[1]
    interp = 1

    samplepair = vec2(0.0)
    upair = vec2(0.0)
    vpair = vec2(0.0)

    for l in ti.ndrange(ldim):
        samplepix = buffer_in[l, int(pos.y)][1]
        #samplepix.y  #sample is a vec3 but they're all the same so pick any
        uwindow = hfit(l, 0, ldim, 0, subu)
        u = ti.cast(lx + uwindow, dtype=TI_INT)
        vwindow = hfit(samplepix, imin, imax, 0.0, subv)
        v = ti.cast(ly + vwindow, dtype=TI_INT)

        for b in ti.ndrange(3): #for a cross
            buffer_out[u-1+b,v] = vec3(0.0, 1.0, 0.0)
            buffer_out[u,v-1+b] = vec3(0.0, 1.0, 0.0)

        if l < ldim:
            #get current and the ahead point too
            samplepair[0] = buffer_in[l, int(pos.y)][1]
            samplepair[1] = buffer_in[l+1, int(pos.y)][1]
            upair[0] = ti.cast((lx + hfit(l, 0, ldim, 0, subu)), dtype=TI_INT)
            upair[1] = ti.cast((lx + hfit(l+1, 0, ldim, 0, subu)), dtype=TI_INT)
            vpair[0] = ti.cast(ly + hfit(samplepair[0], imin, imax, 0.0, subv), dtype=TI_INT)
            vpair[1] = ti.cast(ly + hfit(samplepair[1], imin, imax, 0.0, subv), dtype=TI_INT) 

            spec[l] = vec4(upair[0]/xdim, vpair[0]/ydim, upair[1]/xdim, vpair[1]/ydim)

    if interp == 1:
        for p in ti.ndrange(ipasses):
            for l in ti.ndrange(ldim-1):
                samplepair = vec2(0.0)
                upair = vec2(0.0)
                vpair = vec2(0.0)

                samplepair[0] = buffer_in[l, int(pos.y)][1]
                samplepair[1] = buffer_in[l+1, int(pos.y)][1]

                upair[0] = ti.cast((lx + hfit(l, 0, ldim, 0, subu)), dtype=TI_INT)
                upair[1] = ti.cast((lx + hfit(l+1, 0, ldim, 0, subu)), dtype=TI_INT)

                vpair[0] = ti.cast(ly + hfit(samplepair[0], imin, imax, 0.0, subv), dtype=TI_INT)
                vpair[1] = ti.cast(ly + hfit(samplepair[1], imin, imax, 0.0, subv), dtype=TI_INT) 
                
                u = ti.cast(((upair[0]+upair[1])/2.0),dtype=TI_INT)
                v = ti.cast(((vpair[0]+vpair[1])/2.0),dtype=TI_INT)
                
                for b in ti.ndrange(3): #for a cross
                    buffer_out[u-1+b,v] = vec3(1.0, 0.0, 0.0)
                    buffer_out[u,v-1+b] = vec3(1.0, 0.0, 0.0)