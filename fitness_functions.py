import numpy as np

# Helper functions
def oscillate(x):
    return np.sign(x) * np.exp(np.abs(x) + 0.049 * (np.sin(c1 * np.abs(x)) + np.sin(c2 * np.abs(x))))

def generate_rotation_matrix(dim):
    A = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(A)
    return Q

def generate_asymmetric_matrix(dim, beta):
    B = np.triu(np.random.rand(dim, dim))
    return np.power(B, 1.0 + beta * np.arange(dim) / (dim - 1))

# Constants
c1 = 10
c2 = 7.9

# BBOB 2020 functions
def f1_sphere(x):
    z = x - 0.5
    return np.sum(z**2) - 450

def f2_ellipsoidal(x):
    z = x - 0.5
    return np.sum((10**(6 * np.arange(len(x)) / (len(x)-1))) * z**2) - 450

def f3_rastrigin(x):
    z = x - 0.5
    return 10 * (len(x) - np.sum(np.cos(2 * np.pi * z))) + np.sum(z**2) - 330

def f4_buche_rastrigin(x):
    z = x - 0.5
    z[::2] = np.abs(z[::2])
    t = np.ones_like(x)
    t[z > 0] = 10
    return 10 * (len(x) - np.sum(np.cos(2 * np.pi * z))) + np.sum(t * z**2) - 330

def f5_linear_slope(x):
    z = x.copy()
    z[x > 0] = 0
    s = np.sign(x - 0.5) * 5**np.arange(1, len(x)+1)
    return np.sum(s * z) + 4.6e+2

def f6_attractive_sector(x):
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    w = np.ones_like(z)
    w[z > 0] = 100
    return np.sum(w * z**2)**0.9 - 360

def f7_step_ellipsoidal(x):
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    return 0.1 * max(np.abs(z[0])/1e4, np.sum((10**(2 * np.arange(len(x)) / (len(x)-1))) * z**2)) + oscillate(z[0]) - 360

def f8_rosenbrock(x):
    z = max(1, len(x)**0.5 / 8) * (x - 0.5) + 1
    return np.sum(100 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2) + 390

def f9_rosenbrock_rotated(x):
    z = max(1, len(x)**0.5 / 8) * generate_rotation_matrix(len(x)) @ (x - 0.5) + 1
    return np.sum(100 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2) + 390

def f10_ellipsoidal_rotated(x):
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    return np.sum((10**(6 * np.arange(len(x)) / (len(x)-1))) * z**2) - 450

def f11_discus(x):
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    return 1e6 * z[0]**2 + np.sum(z[1:]**2) - 460

def f12_bent_cigar(x):
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    return z[0]**2 + 1e6 * np.sum(z[1:]**2) - 460

def f13_sharp_ridge(x):
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    return z[0]**2 + 100 * np.sqrt(np.sum(z[1:]**2)) - 450

def f14_different_powers(x):
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    return np.sum(np.abs(z)**(2 + 4 * np.arange(len(x)) / (len(x) - 1))) - 360

def f15_rastrigin_rotated(x):
    z = generate_rotation_matrix(len(x)) @ (10 * (x - 0.5) / np.sqrt(len(x)))
    return 10 * (len(x) - np.sum(np.cos(2 * np.pi * z))) + np.sum(z**2) - 330

def f16_weierstrass(x):
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    return np.sum([np.sum(0.5**k * np.cos(2 * np.pi * 3**k * (zi + 0.5))) for k in range(12) for zi in z]) - 90

def f17_schaffers_f7(x):
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    s = np.sqrt(z[:-1]**2 + z[1:]**2)
    return (np.sum(np.sqrt(s) + np.sqrt(s) * np.sin(50 * s**0.2)**2) / (len(x) - 1))**2 - 300

def f18_schaffers_f7_ill_conditioned(x):
    z = generate_asymmetric_matrix(len(x), 0.5) @ (x - 0.5)
    s = np.sqrt(z[:-1]**2 + z[1:]**2)
    return (np.sum(np.sqrt(s) + np.sqrt(s) * np.sin(50 * s**0.2)**2) / (len(x) - 1))**2 - 300

def f19_composite_griewank_rosenbrock(x):
    z = max(1, np.sqrt(len(x)) / 8) * (x - 0.5) + 1
    return np.sum(100 * (z[1:] - z[:-1]**2)**2 / 4000 - np.cos(100 * (z[1:] - z[:-1]**2)) + 1) + 10 - 360

def f20_schwefel(x):
    z = 2 * generate_asymmetric_matrix(len(x), 0.5) @ (x - 0.5)
    return 418.9828872724339 - np.sum(z * np.sin(np.sqrt(np.abs(z)))) + 100

def f21_gallagher_gaussian_101me(x):
    # This is a simplified version. The full version requires pre-computed data.
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    return np.sum(z**2) - 360

def f22_gallagher_gaussian_21hi(x):
    # This is a simplified version. The full version requires pre-computed data.
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    return np.sum(z**2) - 360

def f23_katsuura(x):
    z = generate_rotation_matrix(len(x)) @ (x - 0.5)
    return -10 / len(x)**2 * np.prod(1 + np.arange(1, len(x)+1) * np.sum([np.abs(2**j * z - np.round(2**j * z)) / 2**j for j in range(1, 33)])) - 360

def f24_lunacek_bi_rastrigin(x):
    mu0 = 2.5
    s = 1 - 1 / (2 * np.sqrt(len(x) + 20) - 8.2)
    mu1 = -np.sqrt((mu0**2 - 1) / s)
    z = generate_rotation_matrix(len(x)) @ (2 * np.sign(x - 0.5) * x - mu0)
    return min(np.sum((x - mu0)**2), len(x) + s * np.sum((x - mu1)**2)) + 10 * (len(x) - np.sum(np.cos(2 * np.pi * z))) - 360

def get_fitness_function(name):
    functions = {
        'f1_sphere': f1_sphere,
        'f2_ellipsoidal': f2_ellipsoidal,
        'f3_rastrigin': f3_rastrigin,
        'f4_buche_rastrigin': f4_buche_rastrigin,
        'f5_linear_slope': f5_linear_slope,
        'f6_attractive_sector': f6_attractive_sector,
        'f7_step_ellipsoidal': f7_step_ellipsoidal,
        'f8_rosenbrock': f8_rosenbrock,
        'f9_rosenbrock_rotated': f9_rosenbrock_rotated,
        'f10_ellipsoidal_rotated': f10_ellipsoidal_rotated,
        'f11_discus': f11_discus,
        'f12_bent_cigar': f12_bent_cigar,
        'f13_sharp_ridge': f13_sharp_ridge,
        'f14_different_powers': f14_different_powers,
        'f15_rastrigin_rotated': f15_rastrigin_rotated,
        'f16_weierstrass': f16_weierstrass,
        'f17_schaffers_f7': f17_schaffers_f7,
        'f18_schaffers_f7_ill_conditioned': f18_schaffers_f7_ill_conditioned,
        'f19_composite_griewank_rosenbrock': f19_composite_griewank_rosenbrock,
        'f20_schwefel': f20_schwefel,
        'f21_gallagher_gaussian_101me': f21_gallagher_gaussian_101me,
        'f22_gallagher_gaussian_21hi': f22_gallagher_gaussian_21hi,
        'f23_katsuura': f23_katsuura,
        'f24_lunacek_bi_rastrigin': f24_lunacek_bi_rastrigin
    }
    return functions.get(name.lower(), f1_sphere)  # Default to sphere if function not found  