"""
pose_graph.py

Pose graph data structure and Gauss-Newton optimisation back-end.

A *pose graph* is a sparse graph where:

* Each **node** holds the absolute 6-DOF pose (position + orientation) of a
  keyframe in the world/map frame.
* Each **edge** encodes a relative-pose constraint between two nodes, together
  with a 6×6 information matrix (inverse covariance) that weights the
  constraint.

The optimiser finds the set of node poses that best satisfies all constraints
simultaneously.  This is the standard *back-end* for graph-SLAM systems and is
what software packages such as **g2o**, **GTSAM**, and **Ceres** implement for
production systems.  The implementation here uses pure NumPy and SciPy (already
required by ``sensor_transposition``), so no additional dependencies are needed.

Edge types
----------
* **Odometry edges** — consecutive keyframe constraints from LiDAR ICP (see
  :func:`sensor_transposition.lidar.scan_matching.icp_align`) or visual
  odometry; typically have high information (low uncertainty).
* **Loop-closure edges** — long-range constraints detected by
  :class:`sensor_transposition.loop_closure.ScanContextDatabase` and verified
  geometrically by :func:`~sensor_transposition.lidar.scan_matching.icp_align`;
  correct accumulated drift by introducing cross-trajectory constraints.

Algorithm
---------
The optimiser minimises the *total cost*:

.. code-block:: text

    F(x) = ½ Σ  eᵢⱼ(x)ᵀ Ωᵢⱼ eᵢⱼ(x)

where the sum runs over all edges, ``eᵢⱼ`` is the 6-D pose error for the
edge, and ``Ωᵢⱼ`` is its 6×6 information matrix.

The 6-D error for edge (i → j) with measured relative transform
``T̂ᵢⱼ`` is::

    eᵢⱼ = [tₑ ; φₑ]

where::

    Tₑ = T̂ᵢⱼ⁻¹ · Tᵢ⁻¹ · Tⱼ        (error transform)
    tₑ = Tₑ[:3, 3]                  (translation error)
    φₑ = log(Tₑ[:3, :3])            (SO(3) log → rotation-vector error)

Gauss-Newton iterations are used to minimise ``F``.  Jacobians are computed
via central finite differences (step 10⁻⁷).  The first node is kept fixed to
remove the gauge freedom of the global reference frame.

Reference
---------
Kümmerle, R., Grisetti, G., Strasdat, H., Konolige, K., & Burgard, W. (2011).
"g2o: A general framework for graph optimization." *IEEE ICRA 2011*, 3607–3613.

Typical use-case
----------------
::

    from sensor_transposition.pose_graph import PoseGraph, optimize_pose_graph
    import numpy as np

    graph = PoseGraph()

    # Add keyframe poses (translation, quaternion [w, x, y, z]).
    for i, (t, q) in enumerate(zip(translations, quaternions)):
        graph.add_node(i, translation=t, quaternion=q)

    # Add odometry edges from LiDAR scan matching.
    for i in range(len(translations) - 1):
        result = icp_align(clouds[i], clouds[i + 1])
        graph.add_edge(
            from_id=i, to_id=i + 1,
            transform=result.transform,
            information=np.eye(6) * 100.0,
        )

    # Add a loop-closure edge.
    graph.add_edge(
        from_id=loop_from, to_id=loop_to,
        transform=loop_transform,
        information=np.eye(6) * 50.0,
    )

    opt = optimize_pose_graph(graph)
    if opt.success:
        for node_id, pose in opt.optimized_poses.items():
            print(node_id, pose["translation"], pose["quaternion"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Internal SE(3) / SO(3) helpers
# ---------------------------------------------------------------------------


def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix for the cross-product with *v*."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map: rotation vector → 3×3 rotation matrix (Rodrigues).

    For near-zero angles the first-order approximation ``I + [φ]×`` is used.
    """
    angle = float(np.linalg.norm(phi))
    if angle < 1e-10:
        return np.eye(3) + _skew(phi)
    K = _skew(phi / angle)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map: 3×3 rotation matrix → rotation vector.

    Uses the axis-angle formula.  For near-identity rotations the
    first-order approximation is used to avoid division by zero.
    """
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-10:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return (angle / (2.0 * np.sin(angle))) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix from unit quaternion ``[w, x, y, z]``."""
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z),  2.0 * (x * y - w * z),        2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z),          1.0 - 2.0 * (x * x + z * z),  2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y),          2.0 * (y * z + w * x),         1.0 - 2.0 * (x * x + y * y)],
    ])


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` from a 3×3 rotation matrix.

    Uses Shepperd's numerical method to select the most stable formula.
    """
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 1e-15 else np.array([1.0, 0.0, 0.0, 0.0])


def _params_to_transform(params: np.ndarray) -> np.ndarray:
    """Build a 4×4 SE(3) matrix from a 6-D pose vector ``[t; rotvec]``."""
    T = np.eye(4)
    T[:3, 3] = params[:3]
    T[:3, :3] = _exp_so3(params[3:6])
    return T


def _transform_to_params(T: np.ndarray) -> np.ndarray:
    """Extract a 6-D pose vector ``[t; rotvec]`` from a 4×4 SE(3) matrix."""
    params = np.zeros(6)
    params[:3] = T[:3, 3]
    params[3:6] = _log_so3(T[:3, :3])
    return params


def _edge_error(
    x: np.ndarray,
    idx_i: int,
    idx_j: int,
    T_meas: np.ndarray,
) -> np.ndarray:
    """Compute the 6-D error for a pose-graph edge.

    Args:
        x: State vector of length ``6 * n_nodes`` holding all pose parameters.
        idx_i: Index of the *from* node in *x*.
        idx_j: Index of the *to* node in *x*.
        T_meas: 4×4 SE(3) measured relative transform from node *i* to *j*.

    Returns:
        6-D error vector ``[t_err (3); φ_err (3)]``:

        * ``t_err = T_err[:3, 3]``      – translation error
        * ``φ_err = log(T_err[:3, :3])`` – rotation error (rotation vector)

        where ``T_err = T_meas⁻¹ · Tᵢ⁻¹ · Tⱼ``.
    """
    T_i = _params_to_transform(x[6 * idx_i: 6 * idx_i + 6])
    T_j = _params_to_transform(x[6 * idx_j: 6 * idx_j + 6])
    T_pred = np.linalg.inv(T_i) @ T_j
    T_err = np.linalg.inv(T_meas) @ T_pred
    return np.concatenate([T_err[:3, 3], _log_so3(T_err[:3, :3])])


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class PoseGraphNode:
    """A node in the pose graph representing one keyframe pose.

    Attributes:
        node_id: Unique integer identifier for this node.
        translation: ``[x, y, z]`` position in the world/map frame (metres).
        quaternion: Unit quaternion ``[w, x, y, z]`` encoding the orientation
            of the ego frame in the world/map frame.
    """

    node_id: int
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quaternion: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])

    @property
    def transform(self) -> np.ndarray:
        """4×4 homogeneous SE(3) matrix: ego frame → world/map frame."""
        T = np.eye(4)
        T[:3, 3] = self.translation
        T[:3, :3] = _quat_to_rotmat(np.asarray(self.quaternion, dtype=float))
        return T


@dataclass
class PoseGraphEdge:
    """A relative-pose constraint (edge) in the pose graph.

    Attributes:
        from_id: ``node_id`` of the *from* node.
        to_id: ``node_id`` of the *to* node.
        transform: 4×4 SE(3) matrix encoding the measured relative pose of
            the *to* frame expressed in the *from* frame (i.e. ``T_from→to``).
        information: 6×6 information matrix (inverse covariance) weighting
            this constraint.  The ordering is ``[t (3); φ (3)]`` – translation
            first, then rotation.  Defaults to the 6×6 identity.
    """

    from_id: int
    to_id: int
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    information: np.ndarray = field(default_factory=lambda: np.eye(6))


# ---------------------------------------------------------------------------
# PoseGraph
# ---------------------------------------------------------------------------


class PoseGraph:
    """A pose graph for SLAM back-end optimisation.

    Nodes represent keyframe poses and edges encode relative-pose constraints
    between them (e.g. from LiDAR odometry or loop-closure detection).

    Example::

        import numpy as np
        from sensor_transposition.pose_graph import PoseGraph, optimize_pose_graph

        graph = PoseGraph()
        graph.add_node(0, translation=[0.0, 0.0, 0.0])
        graph.add_node(1, translation=[1.0, 0.0, 0.0])
        graph.add_edge(
            from_id=0, to_id=1,
            transform=np.array([[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]], float),
            information=np.eye(6) * 100.0,
        )
        result = optimize_pose_graph(graph)
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, PoseGraphNode] = {}
        self._edges: List[PoseGraphEdge] = []

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: int,
        *,
        translation: Sequence[float] | None = None,
        quaternion: Sequence[float] | None = None,
        transform: np.ndarray | None = None,
    ) -> PoseGraphNode:
        """Add a keyframe node to the pose graph.

        The pose can be provided either as separate *translation* /
        *quaternion* components, or as a single 4×4 *transform* matrix.
        If *transform* is given it takes precedence.

        Args:
            node_id: Unique integer identifier.  Must not already exist.
            translation: ``[x, y, z]`` position in the world/map frame.
                Defaults to the origin.
            quaternion: Unit quaternion ``[w, x, y, z]``.  Defaults to
                the identity orientation.
            transform: 4×4 SE(3) homogeneous transform (ego → world).
                If supplied, *translation* and *quaternion* are ignored.

        Returns:
            The newly created :class:`PoseGraphNode`.

        Raises:
            ValueError: If *node_id* already exists in the graph.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} already exists in the pose graph.")

        if transform is not None:
            T = np.asarray(transform, dtype=float)
            if T.shape != (4, 4):
                raise ValueError(f"transform must be a 4×4 matrix, got {T.shape}.")
            t = T[:3, 3].tolist()
            q = _rotmat_to_quat(T[:3, :3]).tolist()
        else:
            t = list(translation) if translation is not None else [0.0, 0.0, 0.0]
            q = list(quaternion) if quaternion is not None else [1.0, 0.0, 0.0, 0.0]
            if len(t) != 3:
                raise ValueError(f"translation must have 3 components, got {len(t)}.")
            if len(q) != 4:
                raise ValueError(f"quaternion must have 4 components, got {len(q)}.")

        node = PoseGraphNode(node_id=node_id, translation=t, quaternion=q)
        self._nodes[node_id] = node
        return node

    def get_node(self, node_id: int) -> PoseGraphNode:
        """Retrieve a node by its identifier.

        Raises:
            KeyError: If *node_id* is not in the graph.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} not found in the pose graph.")
        return self._nodes[node_id]

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        *,
        transform: np.ndarray,
        information: np.ndarray | None = None,
    ) -> PoseGraphEdge:
        """Add a relative-pose constraint (edge) to the pose graph.

        Args:
            from_id: ``node_id`` of the source node.
            to_id: ``node_id`` of the target node.
            transform: 4×4 SE(3) matrix: measured relative pose of the
                *target* frame expressed in the *source* frame.
            information: 6×6 information matrix ``Ω`` weighting this
                constraint.  Column/row ordering is
                ``[tx, ty, tz, rx, ry, rz]``.  Defaults to ``np.eye(6)``.

        Returns:
            The newly created :class:`PoseGraphEdge`.

        Raises:
            ValueError: If either node does not exist, or if *transform* /
                *information* have unexpected shapes.
        """
        if from_id not in self._nodes:
            raise ValueError(f"from_id={from_id} does not exist in the graph.")
        if to_id not in self._nodes:
            raise ValueError(f"to_id={to_id} does not exist in the graph.")

        T = np.asarray(transform, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"transform must be 4×4, got {T.shape}.")

        if information is None:
            Omega = np.eye(6)
        else:
            Omega = np.asarray(information, dtype=float)
            if Omega.shape != (6, 6):
                raise ValueError(f"information must be 6×6, got {Omega.shape}.")

        edge = PoseGraphEdge(from_id=from_id, to_id=to_id, transform=T, information=Omega)
        self._edges.append(edge)
        return edge

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> Dict[int, PoseGraphNode]:
        """Dictionary mapping node IDs to :class:`PoseGraphNode` objects."""
        return dict(self._nodes)

    @property
    def edges(self) -> List[PoseGraphEdge]:
        """List of all :class:`PoseGraphEdge` objects in the graph."""
        return list(self._edges)

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)


# ---------------------------------------------------------------------------
# Optimisation result
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Result returned by :func:`optimize_pose_graph`.

    Attributes:
        optimized_poses: Mapping from ``node_id`` to a dict with keys:

            * ``"translation"`` – optimised ``[x, y, z]`` (list of float).
            * ``"quaternion"``  – optimised ``[w, x, y, z]`` (list of float).
            * ``"transform"``   – optimised 4×4 SE(3) numpy array.

        final_cost: Total weighted squared error at convergence
            ``½ Σ eᵢⱼᵀ Ωᵢⱼ eᵢⱼ``.
        iterations: Number of Gauss-Newton iterations performed.
        success: ``True`` if the solver converged (step norm below
            *tolerance*) or if the graph has at most one node.
    """

    optimized_poses: Dict[int, dict]
    final_cost: float
    iterations: int
    success: bool


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


def optimize_pose_graph(
    graph: PoseGraph,
    *,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    damping: float = 1e-6,
) -> OptimizationResult:
    """Optimise a pose graph using Gauss-Newton iteration.

    The optimiser minimises the total weighted squared error over all edges:

    .. code-block:: text

        F(x) = ½ Σ  eᵢⱼ(x)ᵀ Ωᵢⱼ eᵢⱼ(x)

    using a Gauss-Newton strategy with numerical Jacobians (central
    differences, step 10⁻⁷).  The *first* node (by insertion order) is
    kept fixed throughout the optimisation to remove the gauge freedom of
    the global reference frame.  All other node poses are updated to
    minimise the cost.

    Args:
        graph: The :class:`PoseGraph` to optimise.  Must have at least one
            node.  If it has fewer than two nodes or no edges the result is
            returned immediately with ``success=True``.
        max_iterations: Maximum number of Gauss-Newton iterations.
            Default ``20``.
        tolerance: Convergence threshold on the Euclidean norm of the
            update step ``‖Δx‖``.  Default ``1e-6``.
        damping: Levenberg-Marquardt–style diagonal damping added to the
            Hessian blocks of the free nodes for numerical robustness.
            Default ``1e-6``.

    Returns:
        :class:`OptimizationResult` containing the optimised poses, final
        cost, iteration count, and convergence flag.

    Raises:
        ValueError: If *max_iterations* or *tolerance* are not positive.

    Example::

        from sensor_transposition.pose_graph import PoseGraph, optimize_pose_graph
        import numpy as np

        graph = PoseGraph()
        graph.add_node(0, translation=[0.0, 0.0, 0.0])
        graph.add_node(1, translation=[0.9, 0.0, 0.0])  # noisy initial pose
        graph.add_edge(
            from_id=0, to_id=1,
            transform=np.array([
                [1, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=float),
            information=np.eye(6) * 1000.0,
        )
        result = optimize_pose_graph(graph)
        # result.optimized_poses[1]["translation"] ≈ [1.0, 0.0, 0.0]
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}.")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}.")

    node_ids = list(graph._nodes.keys())
    n = len(node_ids)

    if n == 0:
        return OptimizationResult(
            optimized_poses={},
            final_cost=0.0,
            iterations=0,
            success=True,
        )

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Build initial state vector: 6*n floats, each node as [tx, ty, tz, rx, ry, rz].
    x = np.zeros(6 * n)
    for i, nid in enumerate(node_ids):
        T = graph._nodes[nid].transform
        x[6 * i: 6 * i + 6] = _transform_to_params(T)

    # With 0 or 1 free nodes (n <= 1) or no edges, return immediately.
    if n <= 1 or len(graph._edges) == 0:
        return _build_result(x, node_ids, graph, id_to_idx, 0, success=True)

    # Number of *free* nodes (all except the first, which is fixed).
    n_free = n - 1
    dim_free = 6 * n_free

    eps = 1e-7  # finite-difference step for Jacobians

    converged = False
    iteration = 0
    for iteration in range(max_iterations):
        H_free = np.zeros((dim_free, dim_free))
        b_free = np.zeros(dim_free)

        for edge in graph._edges:
            i = id_to_idx[edge.from_id]
            j = id_to_idx[edge.to_id]

            e0 = _edge_error(x, i, j, edge.transform)   # (6,)
            Omega = edge.information                      # (6, 6)

            # --- Numerical Jacobians (central differences) ---
            # J_i[:, k] = ∂e / ∂x[6i+k]  (free nodes only: index > 0)
            Ji = np.zeros((6, 6))
            for k in range(6):
                xp = x.copy(); xp[6 * i + k] += eps
                xm = x.copy(); xm[6 * i + k] -= eps
                Ji[:, k] = (
                    _edge_error(xp, i, j, edge.transform)
                    - _edge_error(xm, i, j, edge.transform)
                ) / (2.0 * eps)

            Jj = np.zeros((6, 6))
            for k in range(6):
                xp = x.copy(); xp[6 * j + k] += eps
                xm = x.copy(); xm[6 * j + k] -= eps
                Jj[:, k] = (
                    _edge_error(xp, i, j, edge.transform)
                    - _edge_error(xm, i, j, edge.transform)
                ) / (2.0 * eps)

            JiT_O = Ji.T @ Omega   # (6, 6)
            JjT_O = Jj.T @ Omega   # (6, 6)

            # Accumulate into the *free* (reduced) Hessian and gradient.
            # Node 0 is fixed; free node index in H_free is (node_idx - 1).
            if i > 0:
                fi = (i - 1) * 6
                H_free[fi: fi + 6, fi: fi + 6] += JiT_O @ Ji
                b_free[fi: fi + 6] += JiT_O @ e0
            if j > 0:
                fj = (j - 1) * 6
                H_free[fj: fj + 6, fj: fj + 6] += JjT_O @ Jj
                b_free[fj: fj + 6] += JjT_O @ e0
            if i > 0 and j > 0:
                fi = (i - 1) * 6
                fj = (j - 1) * 6
                H_free[fi: fi + 6, fj: fj + 6] += JiT_O @ Jj
                H_free[fj: fj + 6, fi: fi + 6] += JjT_O @ Ji

        # Levenberg-Marquardt damping for numerical robustness.
        H_free += damping * np.eye(dim_free)

        # Solve H_free · Δx_free = −b_free.
        try:
            dx_free = np.linalg.solve(H_free, -b_free)
        except np.linalg.LinAlgError:
            dx_free, _, _, _ = np.linalg.lstsq(H_free, -b_free, rcond=None)

        # Apply update to all free nodes.
        for k in range(n_free):
            node_x_offset = (k + 1) * 6          # offset in full state x
            free_x_offset = k * 6                 # offset in dx_free

            dt = dx_free[free_x_offset: free_x_offset + 3]
            dr = dx_free[free_x_offset + 3: free_x_offset + 6]

            x[node_x_offset: node_x_offset + 3] += dt
            R_old = _exp_so3(x[node_x_offset + 3: node_x_offset + 6])
            dR = _exp_so3(dr)
            x[node_x_offset + 3: node_x_offset + 6] = _log_so3(R_old @ dR)

        step_norm = float(np.linalg.norm(dx_free))
        if step_norm < tolerance:
            converged = True
            break

    return _build_result(x, node_ids, graph, id_to_idx, iteration + 1, success=converged)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _total_cost(
    x: np.ndarray,
    graph: PoseGraph,
    id_to_idx: Dict[int, int],
) -> float:
    """Compute the total weighted squared error for the current state *x*."""
    cost = 0.0
    for edge in graph._edges:
        i = id_to_idx[edge.from_id]
        j = id_to_idx[edge.to_id]
        e = _edge_error(x, i, j, edge.transform)
        cost += float(e @ edge.information @ e)
    return 0.5 * cost


def _build_result(
    x: np.ndarray,
    node_ids: List[int],
    graph: PoseGraph,
    id_to_idx: Dict[int, int],
    iterations: int,
    *,
    success: bool,
) -> OptimizationResult:
    """Build an :class:`OptimizationResult` from the final state vector."""
    optimized_poses: Dict[int, dict] = {}
    for i, nid in enumerate(node_ids):
        params = x[6 * i: 6 * i + 6]
        T = _params_to_transform(params)
        optimized_poses[nid] = {
            "translation": T[:3, 3].tolist(),
            "quaternion": _rotmat_to_quat(T[:3, :3]).tolist(),
            "transform": T,
        }
    final_cost = _total_cost(x, graph, id_to_idx)
    return OptimizationResult(
        optimized_poses=optimized_poses,
        final_cost=final_cost,
        iterations=iterations,
        success=success,
    )
