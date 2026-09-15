import taichi as ti
import taichi.math as tm
import numpy as np
from taichi.math import vec2, vec3, vec4, ivec2, ivec3, ivec4

TI_ARCH = ti.gpu
TI_FP = ti.f32
TI_INT = ti.i32
PY_FP = np.float32

class OrthoCamera:
    def __init__(self, x_size=512, y_size=512, supersample=1):
        self.x_size = x_size
        self.y_size = y_size
        self.supersample = supersample
        self.centre = vec3(0.5 * x_size, 0.5 * y_size, 0.0)
        self.right_ray = vec3(1.0, 0.0, 0.0)
        self.up_ray = vec3(0.0, 0.0, 1.0)
        self.forward_ray = vec3(0.0, 1.0, 0.0)

    @property
    def corner(self):
        return self.centre - 0.5 * self.x_size * self.right_ray - 0.5 * self.y_size * self.up_ray

    def set_centre(self, centre: vec3):
        self.centre = centre
        return self

    def look_at(self, target):
        forward = target - self.centre
        forward /= forward.norm()
        default_up = vec3(0.0, 0.0, 1.0)
        #if (forward - default_up).norm() < 1e-4:
        if (abs(forward @ default_up) > 0.9999):
            self.right_ray = vec3(0.0, -np.sign(forward[2]), 0.0)
        else:
            self.right_ray = default_up.cross(-forward)
            self.right_ray /= self.right_ray.norm()
        self.up_ray = (-forward).cross(self.right_ray)
        self.up_ray /= self.up_ray.norm()
        self.forward_ray = forward
        return self