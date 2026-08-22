import numpy as np
from opentir import Segment, Arc, Ray, reflect


def test_segment_intersection():
    seg = Segment([10, -5], [10, 5])
    ray = Ray(origin=[0, 0], direction=[1, 0])
    hit = seg.intersect(ray)
    assert hit is not None
    t, point, normal = hit
    assert np.isclose(t, 10.0)
    assert np.allclose(point, [10, 0])


def test_segment_no_intersection_behind():
    seg = Segment([-10, -5], [-10, 5])
    ray = Ray(origin=[0, 0], direction=[1, 0])
    assert seg.intersect(ray) is None


def test_arc_intersection():
    arc = Arc(center=[0, 0], radius=5, theta1=-np.pi, theta2=np.pi)
    ray = Ray(origin=[-10, 0], direction=[1, 0])
    hit = arc.intersect(ray)
    assert hit is not None
    t, point, normal = hit
    assert np.allclose(point, [-5, 0], atol=1e-6)


def test_reflect_normal_incidence():
    d = np.array([1.0, 0.0])
    n = np.array([-1.0, 0.0])
    r = reflect(d, n)
    assert np.allclose(r, [-1.0, 0.0])


def test_reflect_45deg_mirror():
    d = np.array([1.0, 0.0])
    n = np.array([-1.0, 1.0]) / np.sqrt(2)
    r = reflect(d, n)
    assert np.allclose(r, [0.0, 1.0], atol=1e-6)
