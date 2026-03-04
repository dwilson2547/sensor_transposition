"""
sliding_window.py

Fixed-lag sliding-window smoother for online SLAM.

A *sliding-window smoother* (also called a *fixed-lag smoother*) keeps only
the most recent ``window_size`` keyframe nodes in the active optimisation
window, rather than growing the pose graph unboundedly as in full batch
optimisation.  This bounds the per-step cost to O(window_size³) regardless of
trajectory length.

When the active window overflows:

1. The **oldest node** is *marginalised* — its most recent optimised pose is
   recorded as a world-frame reference.
2. Any edges from the departing node to nodes that **remain active** are
   converted into *prior factors* on those neighbours: unary constraints
   encoding the world-frame position implied by the marginalised node.
3. Prior factors are represented as edges from a fixed **world anchor** node
   (node ID ``-1`` at the world origin) to the constrained active nodes.  The
   anchor is added as the first node in the pose graph and is therefore held
   fixed by :func:`~sensor_transposition.pose_graph.optimize_pose_graph`.
4. Future optimisation steps use these priors to prevent the active window
   from drifting away from the marginalised portion of the trajectory.

This is a practical approximation of the theoretically exact Schur-complement
marginalisation: it accumulates information from departed nodes through the
chain of odometry edges rather than as a dense cross-node factor.  In the
linear-Gaussian case the result is equivalent when each node has at most one
successor in the active window.

Algorithm
---------
At each step the smoother:

* Builds a :class:`~sensor_transposition.pose_graph.PoseGraph` from the active
  nodes, the odometry / loop-closure edges between them, and the accumulated
  prior edges from marginalisations.
* Calls :func:`~sensor_transposition.pose_graph.optimize_pose_graph` on this
  reduced graph.
* Updates the stored node poses with the optimised values, so that the next
  marginalisation step uses the best available estimate.

Typical use-case
----------------
::

    from sensor_transposition.sliding_window import SlidingWindowSmoother
    import numpy as np

    smoother = SlidingWindowSmoother(window_size=5)

    for i, (trans, rel_tf) in enumerate(keyframe_stream):
        smoother.add_node(i, translation=trans)
        if i > 0:
            smoother.add_edge(i - 1, i, transform=rel_tf, information=np.eye(6) * 200.0)
        result = smoother.optimize()
        if result.success:
            print(f"Node {i}: {result.optimized_poses[i]['translation']}")

Reference
---------
Sibley, G., Matthies, L., & Sukhatme, G. (2010).
"Sliding window filter with application to planetary landing."
*Journal of Field Robotics*, 27(5), 587–608.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from sensor_transposition.pose_graph import (
    OptimizationResult,
    PoseGraph,
    _quat_to_rotmat,
    _rotmat_to_quat,
    optimize_pose_graph,
)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_ANCHOR_ID: int = -1
"""Reserved node ID for the fixed world-frame anchor node.

This node is inserted at the identity pose and held fixed by the optimiser
whenever prior factors are present.  It must not be used as a regular
keyframe ID.
"""


def _pose_dict(translation: list, quaternion: list) -> dict:
    """Build the standard pose dict ``{"translation", "quaternion", "transform"}``.

    Returns copies of the input lists to ensure independence from the caller.
    """
    T = np.eye(4)
    T[:3, 3] = np.asarray(translation, dtype=float)
    T[:3, :3] = _quat_to_rotmat(np.asarray(quaternion, dtype=float))
    return {
        "translation": list(translation),
        "quaternion": list(quaternion),
        "transform": T,
    }


# ---------------------------------------------------------------------------
# SlidingWindowSmoother
# ---------------------------------------------------------------------------


class SlidingWindowSmoother:
    """Online fixed-lag SLAM smoother with world-frame prior factors.

    Maintains at most ``window_size`` active (optimisable) keyframe nodes at
    any time.  When a new node causes the window to overflow, the oldest
    active node is *marginalised*:

    * Its last optimised world-frame pose is recorded.
    * Every edge from the departing node to a node that **remains active** is
      converted into a *prior edge* from the fixed world anchor to that
      neighbour, encoding where the neighbour should be in world coordinates
      according to the departing node.
    * The departing node is then removed from the active window.

    All prior edges accumulated from previous marginalisations are included in
    each optimisation step so that the active window is always anchored to the
    history of the trajectory.

    Args:
        window_size: Maximum number of active (optimisable) nodes.  Must be
            at least 1.  Default ``10``.
        max_iterations: Maximum Gauss-Newton iterations per optimisation step.
            Default ``20``.
        tolerance: Convergence threshold on the step norm ``‖Δx‖``.
            Default ``1e-6``.
        damping: Levenberg-Marquardt diagonal damping for numerical
            robustness.  Default ``1e-6``.

    Raises:
        ValueError: If ``window_size < 1``.

    Example::

        import numpy as np
        from sensor_transposition.sliding_window import SlidingWindowSmoother

        smoother = SlidingWindowSmoother(window_size=5)

        # Build a straight-line trajectory: node i is at (i, 0, 0).
        T_step = np.eye(4)
        T_step[0, 3] = 1.0  # 1 m step in x

        for i in range(20):
            smoother.add_node(i, translation=[float(i), 0.0, 0.0])
            if i > 0:
                smoother.add_edge(i - 1, i, transform=T_step)
            result = smoother.optimize()
            if result.success:
                pos = result.optimized_poses[i]["translation"]
                print(f"node {i:2d}  x={pos[0]:.3f}")
    """

    def __init__(
        self,
        window_size: int = 10,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
        damping: float = 1e-6,
    ) -> None:
        if window_size < 1:
            raise ValueError(
                f"window_size must be >= 1, got {window_size}."
            )
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {max_iterations}."
            )
        if tolerance <= 0.0:
            raise ValueError(
                f"tolerance must be > 0, got {tolerance}."
            )

        self._window_size: int = window_size
        self._max_iterations: int = max_iterations
        self._tolerance: float = tolerance
        self._damping: float = damping

        # Active nodes: ordered by insertion time.
        self._window: deque = deque()

        # Stored pose data for active nodes: node_id -> {"translation": list,
        # "quaternion": list}.  Updated after every optimize() call.
        self._node_data: Dict[int, dict] = {}

        # All edges ever added (including those involving already-marginalised
        # nodes; stale edges are pruned during marginalisation).
        # Each entry: {"from_id": int, "to_id": int,
        #               "transform": ndarray, "information": ndarray}
        self._edges: List[dict] = []

        # World-frame prior factors generated by marginalisation.
        # Maps active node_id → list of (T_prior: ndarray(4,4), Omega: ndarray(6,6)).
        self._priors: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

        # Historical record of each evicted node's last optimised world-frame
        # pose.  Grows monotonically as nodes are marginalised, enabling full
        # trajectory recovery for arbitrarily long runs.
        # node_id -> {"translation": list[float], "quaternion": list[float]}
        self._marginalised_poses: Dict[int, dict] = {}

        self._latest_result: Optional[OptimizationResult] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        """Maximum number of simultaneously active nodes."""
        return self._window_size

    @property
    def active_node_ids(self) -> List[int]:
        """Ordered list of node IDs currently in the active window."""
        return list(self._window)

    @property
    def latest_result(self) -> Optional[OptimizationResult]:
        """The :class:`~sensor_transposition.pose_graph.OptimizationResult`
        from the most recent :meth:`optimize` call, or ``None`` if
        :meth:`optimize` has not yet been called."""
        return self._latest_result

    @property
    def marginalised_poses(self) -> Dict[int, dict]:
        """World-frame poses of all nodes that have been evicted from the window.

        Maps node ID to a dict with keys:

        * ``"translation"`` – optimised ``[x, y, z]`` at eviction time.
        * ``"quaternion"``  – optimised ``[w, x, y, z]`` at eviction time.
        * ``"transform"``   – 4×4 SE(3) numpy array at eviction time.

        For very long trajectories this preserves the complete historical
        record of poses that have left the active window.  The entries are
        added in eviction order and are never removed.
        """
        out: Dict[int, dict] = {}
        for nid, d in self._marginalised_poses.items():
            out[nid] = _pose_dict(d["translation"], d["quaternion"])
        return out

    def full_trajectory(self) -> Dict[int, dict]:
        """Return the complete trajectory: marginalised nodes and active nodes.

        Merges :attr:`marginalised_poses` (historical) with the most recently
        optimised poses from the active window to produce a mapping from every
        node ID ever added to its best world-frame pose estimate.

        * **Marginalised nodes** use the pose recorded at eviction time.
        * **Active nodes** use the latest value updated by :meth:`optimize`.

        This method enables full trajectory recovery for arbitrarily long runs
        without keeping all nodes in the active optimisation window.

        Returns:
            Dict mapping ``node_id`` to
            ``{"translation": list, "quaternion": list, "transform": ndarray}``.
        """
        trajectory = self.marginalised_poses  # returns a fresh copy
        for nid, d in self._node_data.items():
            trajectory[nid] = _pose_dict(d["translation"], d["quaternion"])
        return trajectory

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: int,
        *,
        translation: Optional[List[float]] = None,
        quaternion: Optional[List[float]] = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        """Add a new keyframe to the active window.

        The pose can be provided as separate *translation* / *quaternion*
        components or as a single 4×4 *transform* matrix.  If *transform* is
        given it takes precedence.  All values default to the identity pose.

        If adding the node would exceed ``window_size``, the oldest active
        node is automatically marginalised before the new node is inserted.

        Args:
            node_id: Unique integer identifier.  Must not be
                :data:`_ANCHOR_ID` (``-1``) and must not already be active.
            translation: ``[x, y, z]`` position in the world frame (metres).
                Defaults to the origin.
            quaternion: Unit quaternion ``[w, x, y, z]``.  Defaults to the
                identity orientation.
            transform: 4×4 SE(3) homogeneous transform (ego → world).

        Raises:
            ValueError: If ``node_id`` is reserved or already active.
        """
        if node_id == _ANCHOR_ID:
            raise ValueError(
                f"node_id {_ANCHOR_ID!r} is reserved for the internal "
                "world anchor and cannot be used as a keyframe ID."
            )
        if node_id in self._node_data:
            raise ValueError(
                f"Node {node_id} is already in the active window."
            )

        if transform is not None:
            T = np.asarray(transform, dtype=float)
            if T.shape != (4, 4):
                raise ValueError(
                    f"transform must be a 4×4 matrix, got {T.shape}."
                )
            t = T[:3, 3].tolist()
            q = _rotmat_to_quat(T[:3, :3]).tolist()
        else:
            t = list(translation) if translation is not None else [0.0, 0.0, 0.0]
            q = list(quaternion) if quaternion is not None else [1.0, 0.0, 0.0, 0.0]

        self._node_data[node_id] = {"translation": t, "quaternion": q}
        self._window.append(node_id)

        # Marginalise the oldest node if the window has overflowed.
        if len(self._window) > self._window_size:
            self._marginalise_oldest()

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        *,
        transform: np.ndarray,
        information: Optional[np.ndarray] = None,
    ) -> None:
        """Add a relative-pose constraint between two nodes.

        The edge does not need to connect two currently active nodes; it will
        be silently ignored in :meth:`optimize` if either endpoint has already
        been marginalised.

        Args:
            from_id: Source node ID.
            to_id: Target node ID.
            transform: 4×4 SE(3) matrix: measured relative pose of the target
                frame expressed in the source frame.
            information: 6×6 information matrix ``Ω``.  Defaults to
                ``np.eye(6)``.

        Raises:
            ValueError: If *transform* is not 4×4 or *information* is not 6×6.
        """
        T = np.asarray(transform, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(
                f"transform must be a 4×4 matrix, got {T.shape}."
            )
        if information is None:
            Omega = np.eye(6)
        else:
            Omega = np.asarray(information, dtype=float)
            if Omega.shape != (6, 6):
                raise ValueError(
                    f"information must be a 6×6 matrix, got {Omega.shape}."
                )
        self._edges.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "transform": T,
                "information": Omega,
            }
        )

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> OptimizationResult:
        """Optimise the active window and return the result.

        Builds a :class:`~sensor_transposition.pose_graph.PoseGraph` from:

        * The active nodes in the current window.
        * All odometry / loop-closure edges whose **both** endpoints are
          currently active.
        * Prior edges (from the fixed world anchor to active nodes) derived
          from previous marginalisation steps.

        If any prior factors are present, the world anchor node (ID ``-1``)
        is inserted as the **first** node in the graph — ensuring it is held
        fixed by :func:`~sensor_transposition.pose_graph.optimize_pose_graph`.
        Otherwise, the first active node is fixed as usual.

        After optimisation the stored poses for active nodes are updated with
        the optimised values, improving the quality of subsequent
        marginalisation steps.

        Args:
            callback: Optional callable invoked at the end of every
                Gauss-Newton iteration with ``(iteration: int, cost: float)``.
                Forwarded directly to
                :func:`~sensor_transposition.pose_graph.optimize_pose_graph`.

        Returns:
            :class:`~sensor_transposition.pose_graph.OptimizationResult`
            containing optimised poses **only for the active nodes** (the
            anchor node is excluded).

        Raises:
            ValueError: Propagated from
                :func:`~sensor_transposition.pose_graph.optimize_pose_graph`
                if optimisation parameters are invalid.
        """
        if not self._window:
            return OptimizationResult(
                optimized_poses={},
                final_cost=0.0,
                iterations=0,
                success=True,
            )

        active_list = list(self._window)
        active_set = set(active_list)

        # Determine whether any prior edges connect to active nodes.
        has_priors = any(nid in active_set for nid in self._priors)

        g = PoseGraph()

        # If prior factors exist, add the anchor as the very first node so
        # that it is fixed throughout the optimisation.
        if has_priors:
            g.add_node(_ANCHOR_ID, translation=[0.0, 0.0, 0.0])

        # Add all active nodes.
        for nid in active_list:
            d = self._node_data[nid]
            g.add_node(nid, translation=d["translation"], quaternion=d["quaternion"])

        # Add odometry / loop-closure edges between active nodes.
        for edge in self._edges:
            fid = edge["from_id"]
            tid = edge["to_id"]
            if fid in active_set and tid in active_set:
                g.add_edge(
                    fid,
                    tid,
                    transform=edge["transform"],
                    information=edge["information"],
                )

        # Add prior edges from the world anchor to active nodes.
        if has_priors:
            for nid, prior_list in self._priors.items():
                if nid in active_set:
                    for T_prior, Omega in prior_list:
                        g.add_edge(
                            _ANCHOR_ID,
                            nid,
                            transform=T_prior,
                            information=Omega,
                        )

        full_result = optimize_pose_graph(
            g,
            max_iterations=self._max_iterations,
            tolerance=self._tolerance,
            damping=self._damping,
            callback=callback,
        )

        # Update stored poses with the optimised values so that subsequent
        # marginalisation steps use the best available estimate.
        for nid in active_list:
            if nid in full_result.optimized_poses:
                pose = full_result.optimized_poses[nid]
                self._node_data[nid]["translation"] = pose["translation"]
                self._node_data[nid]["quaternion"] = pose["quaternion"]

        # Build the filtered result (active nodes only, no anchor).
        filtered_poses = {
            nid: full_result.optimized_poses[nid]
            for nid in active_list
            if nid in full_result.optimized_poses
        }

        self._latest_result = OptimizationResult(
            optimized_poses=filtered_poses,
            final_cost=full_result.final_cost,
            iterations=full_result.iterations,
            success=full_result.success,
        )
        return self._latest_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _marginalise_oldest(self) -> None:
        """Remove the oldest node from the window and create prior factors.

        For every edge that connects the departing node to a node that
        **remains active**, a world-frame prior is computed and stored in
        ``self._priors``.  The prior encodes the world-frame pose of the
        active neighbour as inferred by composing the departing node's
        optimised world-frame pose with the measured relative transform.

        Edges and node data for the departing node are then pruned.
        """
        old_id = self._window.popleft()
        active_set = set(self._window)

        # Get the best available world-frame pose of the departing node.
        T_old = self._get_node_transform(old_id)

        # Convert edges from/to the departing node into priors on neighbours.
        for edge in self._edges:
            fid = edge["from_id"]
            tid = edge["to_id"]
            T_meas = edge["transform"]
            Omega = edge["information"]

            if fid == old_id and tid in active_set:
                # Edge: old_id → tid  (T_meas = T_{old→tid})
                # World-frame pose of tid: T_tid_world = T_old * T_meas
                T_prior = T_old @ T_meas
                self._priors.setdefault(tid, []).append(
                    (T_prior.copy(), Omega.copy())
                )
            elif tid == old_id and fid in active_set:
                # Edge: fid → old_id  (T_meas = T_{fid→old})
                # World-frame pose of fid: T_fid_world = T_old * inv(T_meas)
                T_prior = T_old @ np.linalg.inv(T_meas)
                self._priors.setdefault(fid, []).append(
                    (T_prior.copy(), Omega.copy())
                )

        # Drop any priors previously accumulated on the departing node —
        # they have already been incorporated into its optimised pose, which
        # is now reflected in the priors we just created for its neighbours.
        self._priors.pop(old_id, None)

        # Prune edges that involve the departing node.
        self._edges = [
            e
            for e in self._edges
            if e["from_id"] != old_id and e["to_id"] != old_id
        ]

        # Record the final optimised pose for long-trajectory access before
        # removing the node from the active data store.  Store copies of the
        # lists to ensure the stored snapshot is independent of any future
        # mutations to the (now-deleted) node data.
        d = self._node_data[old_id]
        self._marginalised_poses[old_id] = {
            "translation": list(d["translation"]),
            "quaternion": list(d["quaternion"]),
        }

        # Remove the departing node's data.
        del self._node_data[old_id]

    def _get_node_transform(self, node_id: int) -> np.ndarray:
        """Return the 4×4 world-frame transform for *node_id*.

        Uses the stored pose data (which is updated with optimised values
        after every :meth:`optimize` call).
        """
        d = self._node_data[node_id]
        T = np.eye(4)
        T[:3, 3] = np.asarray(d["translation"], dtype=float)
        T[:3, :3] = _quat_to_rotmat(np.asarray(d["quaternion"], dtype=float))
        return T
