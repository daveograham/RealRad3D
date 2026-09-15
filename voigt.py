import taichi as ti
import taichi.math as tm
import numpy as np

@ti.data_oriented
class Voigt:
    def __init__(self):
        wei_coeff24 = np.array([-1.5137461654527820e-10, 4.9048215867870488e-09,
            1.3310461806370372e-09,  -3.0082822811202271e-08,
            -1.9122258522976932e-08, 1.8738343486619108e-07,
            2.5682641346701115e-07,  -1.0856475790698251e-06,
            -3.0388931839840047e-06, 4.1394617248575527e-06,
            3.0471066083243790e-05,  2.4331415462641969e-05,
            -2.0748431511424456e-04, -7.8166429956142650e-04,
            -4.9364269012806686e-04, 6.2150063629501763e-03,
            3.3723366855316413e-02,  1.0838723484566792e-01,
            2.6549639598807689e-01,  5.3611395357291292e-01,
            9.2570871385886788e-01,  1.3948196733791203e+00,
            1.8562864992055408e+00,  2.1978589365315417e+00], dtype=np.float32)
        self.wei24 = ti.field(dtype=ti.f32, shape = 24)
        self.wei24.from_numpy(wei_coeff24)

    @ti.func
    def voigtCalc(self, z: tm.vec2):
        #create voigt profile from two vectors of v (position - real) and a (damping - imaginary)
        v = z[0]
        a = z[1]
        wz = tm.vec2(0.0, 0.0)

        N = 24
        l = 2**(-0.25) * N**(0.5)
        rrpi = ti.rsqrt(tm.pi)

        s = ti.abs(v) + a
        
        iz_neg = tm.vec2(a, -v)
        
        if s >= 15.0: #16
            #iz = tm.vec2(a, -v)
            wz = (iz_neg * rrpi) / (0.5 + iz_neg*iz_neg)
        else:
            # L+iz / L-iz  z=v+ia
            rL = tm.cdiv(tm.vec2(1.0, 0.0), tm.vec2((l + a), -v))
            t = tm.cmul(rL, tm.vec2((l - a), v))
            #horners method here on the sum in Weidman 1994 (38)
            sum = self.horner(t)
            wz = tm.cmul(rL, (tm.vec2(rrpi, 0.0) + (tm.cmul(tm.vec2(2.0, 0.0), tm.cmul(rL, sum)))))
        return wz

    @ti.func
    def horner(self, t: tm.vec2):
        #do the horner a_n*Z^n sum in a loop with complex multiplication
        n = 1
        N = 24
        sum = tm.vec2(self.wei24[0],0.0)
        
        while n < N:      
            #print(f"{wei24[n]:e}")
            sum = tm.vec2(self.wei24[n],0.0) + tm.cmul(sum, t)
            n += 1

        return sum