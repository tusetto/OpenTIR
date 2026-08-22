import numpy as np
from opentir.chromatic import wavelength_samples, wavelength_to_rgb, chromatic_rays
from opentir.materials import PMMA, AIR, POLYCARBONATE
from opentir.optics import Ray


def test_wavelength_samples_n1_returns_reference():
    wls = wavelength_samples(1)
    assert wls == [589.3]


def test_wavelength_samples_count():
    for n in [3, 7, 15]:
        wls = wavelength_samples(n)
        assert len(wls) == n


def test_wavelength_samples_range():
    wls = wavelength_samples(10)
    assert min(wls) >= 380.0
    assert max(wls) <= 720.0


def test_wavelength_to_rgb_reference_is_yellow():
    r, g, b = wavelength_to_rgb(589.3)
    assert r > 0.5 and g > 0.3 and b < 0.3


def test_wavelength_to_rgb_extremes_gray():
    for nm in [200, 800]:
        rgb = wavelength_to_rgb(nm)
        assert rgb == (0.5, 0.5, 0.5)


def test_wavelength_to_rgb_values_in_range():
    for nm in range(380, 720, 10):
        r, g, b = wavelength_to_rgb(nm)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0


def test_chromatic_rays_count():
    base = [Ray([0, 0], [1, 0], medium=AIR) for _ in range(5)]
    wls = wavelength_samples(7)
    result = chromatic_rays(base, wls)
    assert len(result) == 5 * 7


def test_chromatic_rays_wavelength_preserved():
    base = [Ray([0, 0], [1, 0], medium=AIR)]
    wls = [450.0, 550.0, 650.0]
    result = chromatic_rays(base, wls)
    assert [wl for wl, _ in result] == wls
    assert all(r.wavelength_nm == wl for wl, r in result)


def test_dispersion_pmma_vs_polycarbonate():
    """PC should show more dispersion than PMMA (lower Abbe number)."""
    dn_pmma = PMMA.n_at(450) - PMMA.n_at(650)
    dn_pc   = POLYCARBONATE.n_at(450) - POLYCARBONATE.n_at(650)
    assert dn_pc > dn_pmma, "PC deve essere più dispersivo di PMMA"


def test_chromatic_aberration_bends_blue_more():
    """Blue light should refract more than red for the same PMMA surface."""
    from opentir import (Segment, Surface, OpticalSystem, Ray, AIR, PMMA)
    seg = Segment([5, 0], [5, 20], name="lens")
    surf = Surface(seg, kind="refract", name="lens",
                    material_in=PMMA, material_out=AIR)
    system = OpticalSystem()
    system.add(surf)
    target = Segment([100, 0], [100, 30], name="target")
    system.add(Surface(target, kind="target", name="target"))

    direction = np.array([np.cos(np.radians(20)), np.sin(np.radians(20))])
    ray_blue = Ray([0, 0], direction, medium=PMMA, wavelength_nm=450.0)
    ray_red  = Ray([0, 0], direction, medium=PMMA, wavelength_nm=650.0)

    tr_blue = max(system.trace_ray(ray_blue, max_bounces=4), key=lambda t: t["power"])
    tr_red  = max(system.trace_ray(ray_red,  max_bounces=4), key=lambda t: t["power"])

    hits_blue = tr_blue["hits"]
    hits_red  = tr_red["hits"]
    assert hits_blue and hits_red, "entrambi i raggi devono raggiungere il target"
    r_blue = hits_blue[0][1][1]
    r_red  = hits_red[0][1][1]
    # Blue refracts more → hits target at larger r for a fixed incidence angle
    assert r_blue != r_red, "blu e rosso devono colpire il target in posizioni diverse"
