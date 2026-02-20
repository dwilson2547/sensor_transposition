"""
transform.py

Utility class wrapping a 4×4 homogeneous transformation matrix with
convenience methods for applying transforms to points and point clouds.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class Transform:
    """Wraps a 4×4 homogeneous transformation matrix.

    Supports composition (``@``) and application to 3-D points.

    Args:
        matrix: A 4×4 array-like.  Defaults to the identity transform.
    """

    def __init__(self, matrix: np.ndarray | None = None) -> None:
        if matrix is None:
            self._matrix = np.eye(4, dtype=float)
        else:
            self._matrix = np.asarray(matrix, dtype=float)
            if self._matrix.shape != (4, 4):
                raise ValueError(f"Transform matrix must be 4×4, got {self._matrix.shape}.")

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls) -> "Transform":
        """Return the identity transform."""
        return cls(np.eye(4, dtype=float))

    @classmethod
    def from_translation(cls, translation: Sequence[float]) -> "Transform":
        """Build a pure-translation transform."""
        T = np.eye(4, dtype=float)
        T[:3, 3] = translation
        return cls(T)

    @classmethod
    def from_rotation_matrix(cls, rotation: np.ndarray, translation: Sequence[float] | None = None) -> "Transform":
        """Build a transform from a 3×3 rotation matrix and optional translation."""
        T = np.eye(4, dtype=float)
        T[:3, :3] = rotation
        if translation is not None:
            T[:3, 3] = translation
        return cls(T)

    @classmethod
    def from_quaternion(cls, quaternion: Sequence[float], translation: Sequence[float] | None = None) -> "Transform":
        """Build a transform from a quaternion [w, x, y, z] and optional translation."""
        from sensor_transposition.sensor_collection import _quaternion_to_rotation_matrix
        R = _quaternion_to_rotation_matrix(list(quaternion))
        return cls.from_rotation_matrix(R, translation)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """The underlying 4×4 numpy array."""
        return self._matrix

    @property
    def rotation(self) -> np.ndarray:
        """The 3×3 rotation sub-matrix."""
        return self._matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """The 3-element translation vector."""
        return self._matrix[:3, 3]

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def inverse(self) -> "Transform":
        """Return the inverse of this transform."""
        return Transform(np.linalg.inv(self._matrix))

    def __matmul__(self, other: "Transform") -> "Transform":
        """Compose two transforms: ``self @ other``."""
        if isinstance(other, Transform):
            return Transform(self._matrix @ other._matrix)
        return NotImplemented

    # ------------------------------------------------------------------
    # Applying to points
    # ------------------------------------------------------------------

    def apply_to_point(self, point: Sequence[float]) -> np.ndarray:
        """Transform a single 3-D point.

        Args:
            point: A (3,) or (4,) array-like.  If (3,), a homogeneous
                   coordinate of 1 is appended automatically.

        Returns:
            A (3,) numpy array with the transformed coordinates.
        """
        p = np.asarray(point, dtype=float)
        if p.shape == (3,):
            p = np.append(p, 1.0)
        if p.shape != (4,):
            raise ValueError(f"Point must be length 3 or 4, got {p.shape}.")
        return (self._matrix @ p)[:3]

    def apply_to_points(self, points: np.ndarray) -> np.ndarray:
        """Transform an (N, 3) or (N, 4) array of points.

        Args:
            points: Shape ``(N, 3)`` or ``(N, 4)``.  If ``(N, 3)``, homogeneous
                    coordinates of 1 are appended automatically.

        Returns:
            Shape ``(N, 3)`` numpy array of transformed points.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            return self.apply_to_point(pts)
        if pts.shape[1] == 3:
            ones = np.ones((pts.shape[0], 1), dtype=float)
            pts = np.hstack([pts, ones])
        if pts.shape[1] != 4:
            raise ValueError(f"Points array must have 3 or 4 columns, got {pts.shape[1]}.")
        return (self._matrix @ pts.T).T[:, :3]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return np.allclose(self._matrix, other._matrix)

    def __repr__(self) -> str:
        return f"Transform(\n{self._matrix}\n)"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def sensor_to_sensor(
    source_to_ego: np.ndarray,
    target_to_ego: np.ndarray,
) -> np.ndarray:
    """Compute the transform from *source* to *target* sensor frame.

    Args:
        source_to_ego: 4×4 homogeneous matrix T_source→ego.
        target_to_ego: 4×4 homogeneous matrix T_target→ego.

    Returns:
        4×4 homogeneous matrix T_source→target.

        Derivation::

            T_source_to_target = inv(T_target_to_ego) @ T_source_to_ego
    """
    return np.linalg.inv(target_to_ego) @ source_to_ego
