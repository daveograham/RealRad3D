import argparse
import numpy as np
from pathlib import Path
import taichi as ti
import taichi.math as tm
from taichi.math import vec2, vec3, vec4, ivec2, ivec3, ivec4
import xarray as xr
from ruamel.yaml import YAML
yaml = YAML()

TI_ARCH = ti.gpu
TI_FP = ti.f32
TI_INT = ti.i32
PY_FP = np.float32

from load_atom import aload
from modelBuild import Constants
from modelBuild import TaiAtom
from modelBuild import ModelSetup
from modelBuild import Slit

#import rays
from rays import TraceDex3d
from camera import OrthoCamera
import viz

from config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace 3D volume from sparse Dex atmosphere"
        )
    parser.add_argument(
        '--path',
        metavar='PATH',
        type=Path,
        help="The path to the dex .yaml config file",
    )
    args = parser.parse_args()

    with open(args.path, 'r') as f:
        dexcfg = yaml.load(f)
    MODEL_ROOT = args.path.parent
    POPS_PATH = MODEL_ROOT / dexcfg["output_path"]
    MODEL_PATH = MODEL_ROOT / dexcfg["atmos_path"]
    ATOM_PATH = TRACER_ROOT + dexcfg["atoms"][ELEMENT]["path"]

    ds = xr.open_dataset(MODEL_PATH)
    ps = xr.open_dataset(POPS_PATH)

    ti.init(arch=TI_ARCH, default_fp=TI_FP)

    cam = OrthoCamera(supersample=4, x_size=int(CAM_ASPECT * CAM_HEIGHT), y_size=CAM_HEIGHT)
    atompy = aload(ATOM_PATH)#, check64=True)
    atomfield = TaiAtom(atompy)

    trslit = Slit(SLIT_RES, CAM_HEIGHT,(CAM_WIDTH/2, CAM_HEIGHT//2))

    #WAVELENGTH ARRAY
    LINEIDX = 0
    restwav = atompy["line_lambda0"][LINEIDX]
    print('TRACING FOR ',restwav)

    model_cache = ModelSetup(ds, ps, Constants(), atomfield, LINEIDX, ATOM)
    
    #CREATE A TRACEDEX3D OBJECT WITH A CUSTOM TRANSFER FUNCTION
    @ti.data_oriented
    class TraceDex3dDoubleTransfer(TraceDex3d):

        @staticmethod
        @ti.func
        def transfer_function(param):
            scale = 0.005
            return (
                scale * vec3(0.0, 0.0, 1.0) * ti.exp(-(param - 3.8)**2 / 0.1**2)
                + scale * vec3(0.0, 1.0, 0.0) * ti.exp(-(param - 4.5)**2 / 0.1**2)
                + scale * vec3(1.0, 0.0, 0.0) * ti.exp(-(param - 5)**2 / 0.1**2)
            )
        #REDEFINE THE TRANSFER FUNCTION (TURNS TEMP/DENS INTO RGB VALUES)

    #START THE TRACE CLASS
    tracer = TraceDex3dDoubleTransfer(
        ds=ds,
        model_cache=model_cache,
        camera=cam,
        param_name="temperature",
        transform_param=lambda x: np.log10(x),
        rt_trace = True
    )

    gui = ti.GUI("TraceTime", WINDOW_SIZE)
    gui_buffer = ti.field(vec3, shape=WINDOW_SIZE)
    spectrum_buffer = ti.field(vec4, shape=WAVESTEPS)
    specsegments = spectrum_buffer.to_numpy()

    spectrum_start = np.zeros((WAVESTEPS,2))
    spectrum_end = np.zeros((WAVESTEPS,2))

    theta_slider = gui.slider("Theta", 0.0, 360.0, step=1)
    phi_slider = gui.slider("Phi", 0.0, 180.0, step=1)
    phi_slider.value = 90.0
    bias = gui.slider("inv bias", 1.0, 10.0)
    white_point = gui.slider("white point", 0.5, 10.0)
    white_point.value = 1.0
    gamma = gui.slider("gamma", 1.5, 3.5)
    gamma.value = 2.2
    #SLIDERS
    blur_width = gui.slider("PSF FWHM", 0.0, BLUR_KERNEL_SIZE // 2)
    cam_dist = 1e3
    blur_buffer = ti.field(vec3, shape=(cam.x_size, cam.y_size))
    blur_kernel = ti.field(ti.f32, shape=(BLUR_KERNEL_SIZE))
    wing_slider = gui.slider("Wing Sample", 1, (WAVESTEPS//2)-1)
    wing_slider.value = WAVESTEPS/4

    #DEFAULT SLIT/POINTER POSITION
    slit_position = WINDOW_SIZE[0]//2
    default_pos = vec2(CAM_WIDTH//2, CAM_HEIGHT//2)
    pos = default_pos

    frame = 0
    iwave = CENWAV
    theta_in = theta_slider.value
    phi_in = phi_slider.value
    sticky_pixel = 0
    scancomplete = 0
    clicked = 0
    noslitdraw = 0
    free_slit = 0
    force_scan = 0
    lookat_modcen = 1
    deadzone = (SPEC_WIDTH + VIEWER_WIDTH)/WINDOW_SIZE[0]
    target = vec3(0.0)
    current_marker = vec2(0.0)
    xcen = (VIEWER_WIDTH//2 + SPEC_WIDTH)/WINDOW_SIZE[0]
    ycen = (VIEWER_HEIGHT//2 + VERTICAL_PAD)/WINDOW_SIZE[1]

    while gui.running:
        events = list(gui.get_events())
        #change in theta from last frame
        target_centre = tracer.aabb // 2
        theta = np.deg2rad(theta_slider.value)
        phi = np.deg2rad(phi_slider.value)
        dviewer = (theta_in - theta) + (phi_in - phi)
        theta_in = theta
        phi_in = phi

        phi = np.deg2rad(phi_slider.value)
        r = vec3(np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))

        if frame == 0:
            #LOOKAT MODEL CENTRE
            trslit.set_pos(vec2(1.0,0.0), vec2(CAM_WIDTH//2, CAM_HEIGHT//2))
            target = target_centre
            cam.set_centre(tracer.aabb // 2 + r * cam_dist).look_at(target)

        if clicked == 2:
            #SET CAMERA POSITION WITH RMB LOOK AT LOCATION
            trslit.set_pos(vec2(1.0,0.0), vec2(CAM_WIDTH//2, CAM_HEIGHT//2))
            if sticky_pixel != 0:
                target = trslit.contact[sticky_pixel]
                cam.set_centre(tracer.aabb // 2 + r * cam_dist).look_at(target)
            force_scan = 1

        if clicked == 0:
            #DEFAULT MODE - STAY ON TARGET
            cam.set_centre(tracer.aabb // 2 + r * cam_dist).look_at(target)
            
        dframe = frame % 2
        #ALTERNATELY FILL SPECTRAL WINDOW
        iframe = frame % WAVESTEPS
        alt = ((iframe % 2)*2)-1
        iwave += iframe*alt
        irgb = frame % 3

        WINGDIST = wing_slider.value
        #RUN THE FULL WINDOW TRACER WITH SLIT TRACE FOR THE CAMERA SETUP ABOVE
        wavepointer = vec4(0, iwave, irgb, WINGDIST) #first value isn't currently used
        #reset the scan to centre when complete
        if iwave == 0:
            iwave = CENWAV
            scancomplete = 1

        #RUN THE TRACE
        if dviewer == 0:
            tracer.trace_for_camera(
                cam=cam,
                slit=trslit,
                waveindex = wavepointer,
                step_size=1,
                mode=2,
            ) #both spec and image
        if force_scan == 1:
            tracer.trace_for_camera(
                cam=cam,
                slit=trslit,
                waveindex = wavepointer,
                step_size=1,
                mode=2,
            ) #both spec and image
            force_scan = 0
        else:
            tracer.trace_for_camera(
                cam=cam,
                slit=trslit,
                waveindex = wavepointer,
                step_size=1,
                mode=0,
            ) # just image

        #GET VIEWER POSITION
        if gui.is_pressed(ti.GUI.LMB):
            fx, fy = gui.get_cursor_pos()
            vx = fx*WINDOW_SIZE[0] - SPEC_WIDTH 
            vy = fy*WINDOW_SIZE[1] - VERTICAL_PAD
            pos = viz.win2cam(vx,vy,VIEWER_SIZE,CAM_SIZE)
        elif gui.is_pressed(ti.GUI.RMB):
            fx, fy = gui.get_cursor_pos()
            vx = fx*WINDOW_SIZE[0] - SPEC_WIDTH
            vy = fy*WINDOW_SIZE[1] - VERTICAL_PAD
            pos = viz.win2cam(vx,vy,VIEWER_SIZE,CAM_SIZE)

        elif gui.is_pressed('d', ti.GUI.LEFT):
            theta_slider.value += 1
            clicked = 0
        elif gui.is_pressed('d', ti.GUI.RIGHT):
            theta_slider.value -= 1
            clicked = 0
        elif gui.is_pressed('d', ti.GUI.UP):
            phi_slider.value += 1
            clicked = 0
        elif gui.is_pressed('d', ti.GUI.DOWN):
            phi_slider.value -= 1
            clicked = 0

        for e in events:
            #MOUSE RELEASES
            if e.type == ti.GUI.RELEASE:
                if e.key == 'LMB':
                    #new slit position without tracking
                    if fx < deadzone:
                        trslit.set_pos(vec2(1.0,0.0), vec2(pos.x, CAM_HEIGHT//2))
                        clicked = 1

                if e.key == 'RMB':
                    #TRACE AT CLICK LOCATION TO GET CONTACT
                    if fx < deadzone:
                        trslit.set_pos(vec2(1.0,0.0), vec2(pos.x, CAM_HEIGHT//2))
                        tracer.trace_for_camera(
                            cam=cam,
                            slit=trslit,
                            waveindex = wavepointer,
                            step_size=1,
                            mode=3,
                        )
                        click = int(pos.y)
                        pt = trslit.contact[click]
                        if vec3(1.0,1.0,1.0) @ pt > 0:
                            sticky_pixel = click
                        else:
                            sticky_pixel = 0
                        noslitdraw = 1
                        clicked = 2
                        pos = default_pos

                #ADDITIONAL KEYBAORD PRESSES
                if e.key == ti.GUI.ESCAPE:
                    gui.close()
                    print('Goodbye!')
                if e.key == 's':
                    print(pos.y)
                    print('Saved spectral profile')
                if e.key == 'u':
                    sticky_pixel = 0
                    print('Unlock slit')
                if e.key == 'f':
                    sticky_pixel = 0
                    target = target_centre
                    clicked = 0
                    print('Re-centre')
                    pos = default_pos
                    trslit.set_pos(vec2(1.0,0.0), default_pos)
                    current_marker = vec2(xcen,ycen)
                if e.key == 'h':
                    sticky_pixel = 0
                    target = target_centre
                    clicked = 0
                    theta_slider.value = 0
                    phi_slider.value = 90
                    pos = default_pos
                    trslit.set_pos(vec2(1.0,0.0), default_pos)
                    current_marker = vec2(xcen,ycen)
                    print('Home view')

        #TONE MAP THE TRACE FRAME BUFFER
        if irgb == 2:
            viz.clear_buffer(gui_buffer, VIEWER_AABB, 2)
            viz.tonemap_and_blit(tracer.fb, 1.0 / bias.value, white_point.value, gamma.value, blur_width.value, blur_buffer, blur_kernel, gui_buffer, VIEWER_AABB)
        if dviewer == 0:
            viz.spectrum_to_frame_buffer(tracer.wb, gui_buffer, SPECTRA_AABB, 1.0 / bias.value, white_point.value, gamma.value)
            viz.draw_wing_markers(CENWAV, WAVESTEPS, int(WINGDIST), gui_buffer, SPECTRA_AABB)
            #plot window
            viz.clear_buffer(gui_buffer, PLOT_AABB, 2)
            viz.draw_spectra_plot(tracer.wb, gui_buffer, spectrum_buffer, pos, PLOT_AABB, 0.0, 8.0, 0)
        #draw slit onto the main window / gui buffer
        if noslitdraw == 0:
            viz.draw_slit_gui(trslit, trslit.dir, gui_buffer, VIEWER_AABB, CAM_SIZE, VIEWER_SIZE)
        else:
            #hold drawing the slit when RMB released until the next frame
            noslitdraw = 0

        if clicked == 1:
            current_marker = vec2(fx,fy)
        if clicked == 2 or frame == 0:
            current_marker = vec2(xcen,ycen)
        if current_marker.x < deadzone:
            viz.click_markers(current_marker, gui_buffer, WINDOW_SIZE)

        gui.set_image(gui_buffer)

        seg = viz.draw_spectra_segment(tracer.wb, gui_buffer, pos, PLOT_AABB, 0.0, 8.0, int(iframe))
        specsegments = spectrum_buffer.to_numpy()
        gui.lines(specsegments[:,0:2], specsegments[:,2::], radius=1, color=0xFFFFFF)

        gui.show()
        frame += 1