"""
opentir.geometry
~~~~~~~~~~~~~~~~~
Primitive entities used to describe the 2D cross-section of an
axisymmetric optical system.

Convention
----------
- z : coordinate along the optical axis
- r : radial coordinate (>= 0). The full 3D optic is obtained, in a
  later release, by revolving this profile around the z axis.

All coordinates are stored as numpy arrays [z, r].
"""

import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return v / n


class Segment:
    """A straight optical surface element between two points."""

    def __init__(self, p1, p2, name="segment"):
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.name = name

    def intersect(self, ray, t_min=1e-9):
        """
        Intersect a Ray with this segment.
        Returns (t, point, normal) or None if there is no valid hit.
        """
        d = self.p2 - self.p1
        A = np.array([[ray.direction[0], -d[0]],
                      [ray.direction[1], -d[1]]])
        b = self.p1 - ray.origin
        det = np.linalg.det(A)
        if abs(det) < 1e-14:
            return None  # parallel
        t, s = np.linalg.solve(A, b)
        if t <= t_min or s < 0 or s > 1:
            return None
        point = ray.origin + t * ray.direction
        normal = _unit(np.array([-d[1], d[0]]))
        return t, point, normal

    def sample_points(self, n=2):
        return np.array([self.p1 + (self.p2 - self.p1) * k / (n - 1) for k in range(n)])


class Arc:
    """A circular arc optical surface element."""

    def __init__(self, center, radius, theta1, theta2, name="arc"):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.theta1 = float(theta1)  # radians
        self.theta2 = float(theta2)
        self.name = name

    def intersect(self, ray, t_min=1e-9):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        candidates = sorted([(-b - sq) / (2 * a), (-b + sq) / (2 * a)])
        for t in candidates:
            if t <= t_min:
                continue
            point = ray.origin + t * ray.direction
            theta = np.arctan2(point[1] - self.center[1], point[0] - self.center[0])
            if self._theta_in_range(theta):
                normal = _unit(point - self.center)
                return t, point, normal
        return None

    def _theta_in_range(self, theta):
        two_pi = 2 * np.pi
        theta = theta % two_pi
        t1 = self.theta1 % two_pi
        t2 = self.theta2 % two_pi
        if t1 <= t2:
            return t1 <= theta <= t2
        return theta >= t1 or theta <= t2

    def sample_points(self, n=32):
        thetas = np.linspace(self.theta1, self.theta2, n)
        return np.array([self.center + self.radius * np.array([np.cos(t), np.sin(t)]) for t in thetas])


class Profile:
    """A composite profile made of an ordered list of Segment/Arc elements."""

    def __init__(self, elements=None, name="profile"):
        self.elements = elements or []
        self.name = name

    def add(self, element):
        self.elements.append(element)
        return self

    def sample_points(self, n_per_element=32):
        pts = []
        for el in self.elements:
            pts.append(el.sample_points(n_per_element))
        return np.vstack(pts) if pts else np.empty((0, 2))
