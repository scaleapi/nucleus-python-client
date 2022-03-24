import numpy as np
import transforms3d as t3d
from pyquaternion import Quaternion


class Transform:
    """Transform object represent a rigid transformation matrix (rotation and translation).

    Transform is a 4x4 matrix, although it could be instance using (16,1), (3,4), (3,3) or (3,1) matrixes.
    **Note**: not all the methods from Transform will work using scaled/small matrixes

    .. highlight:: python
    .. code-block:: python

      [
        [ r00, r01, r02, t0],
        [ r10, r11, r12, t1],
        [ r20, r21, r22, t2],
        [ 0, 0, 0, 1]
      ]

    """

    def __init__(self, value=None):
        self.matrix = np.eye(4)

        if isinstance(value, Transform):
            self.matrix = np.array(value)

        elif isinstance(value, Quaternion):
            self.rotation = value.rotation_matrix

        elif value is not None:
            value = np.array(value)
            if value.shape == (4, 4):
                self.matrix = value

            if value.shape == (16,):
                self.matrix = np.array(value).reshape((4, 4))

            elif value.shape == (3, 4):
                self.matrix[:3, :4] = value

            elif value.shape == (3, 3):
                self.rotation = value

            elif value.shape == (3,):
                self.translation = value

    @staticmethod
    def from_Rt(R, t):
        """Create a transform based on a rotation and a translation components.

        :param R: Rotation matrix or quaternion.
        :type R: Quaternion, list
        :param t: Translation component
        :type t: list
        :returns: Transform created based on the components
        :rtype: Transform

        """
        if isinstance(R, Quaternion):
            R = R.rotation_matrix
        return Transform(np.block([[R, np.mat(t).T], [np.zeros(3), 1]]))

    @staticmethod
    def from_euler(angles, axes="sxyz", degrees=False):
        """Create a transform from euler angles

        :param angles: Values of the rotation per axis
        :type angles: list
        :param axes: Order of the axis (default ``sxyz``)
        :type axes: str
        :param degrees: Use degrees or radians values (default ``False`` = radians)
        :type degrees: boolean
        :returns: Transform created from euler angles
        :rtype: Transform

        """
        if degrees:
            angles = np.deg2rad(angles)
        return Transform(t3d.euler.euler2mat(*angles, axes=axes))

    @staticmethod
    def from_transformed_points(A, B):
        """Create a transform from two points

        :param A: Point A (x,y,z)
        :type A: list
        :param B: Point B (x,y,z)
        :type B: list
        :returns: Transform created from the angles
        :rtype: Transform

        """
        assert A.shape == B.shape

        mean_A = np.mean(A, axis=0)
        mean_B = np.mean(B, axis=0)
        centroid_A = A - mean_A
        centroid_B = B - mean_B

        C = centroid_A.T @ centroid_B
        V, S, W = np.linalg.svd(C)

        if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]

        R = V @ W
        t = mean_B - mean_A @ R

        return Transform.from_Rt(R.T, t)

    @staticmethod
    def random():
        """Create a transform from random rotation and translation

        :returns: Transform created based on the angles
        :rtype: Transform

        """
        return Transform.from_Rt(Quaternion.random(), np.random.rand(3))

    @property
    def rotation(self):
        """Transform rotation

        :getter: Return transform's rotation
        :setter: Set transform rotation, could use a 3x3 matrix or a Quaternion
        :type: 3x3 matrix
        """
        return self.matrix[:3, :3]

    @property
    def quaternion(self):
        """Transform rotation as quaternion

        :getter: Return transform's rotation as quaternion
        :type: Quaternion
        """
        return Quaternion(t3d.quaternions.mat2quat(self.matrix[:3, :3]))

    @rotation.setter
    def rotation(self, rotation):
        if isinstance(rotation, Quaternion):
            rotation = rotation.rotation_matrix
        self.matrix[:3, :3] = rotation

    @property
    def position(self):
        """Transform position/translation

        :getter: Return transform's position
        :setter: Set transform's position list(3x1)
        :type: list
        """
        return self.matrix[:3, 3].flatten()

    @position.setter
    def position(self, position):
        self.matrix[:3, 3] = np.array(position).reshape(3)

    @property
    def translation(self):
        """Transform position/translation

        :getter: Return transform's position
        :setter: Set transform's position list(3x1)
        :type: list
        """
        return self.matrix[:3, 3].flatten()

    @translation.setter
    def translation(self, translation):
        self.matrix[:3, 3] = np.array(translation).reshape(3)

    @property
    def euler_angles(self, axes="sxyz"):
        """Transform rotation in euler angles

        :getter: Return transform's rotaiton in euler angles
        :type: list
        """
        return t3d.euler.mat2euler(self.matrix, axes=axes)

    @property
    def euler_degrees(self, axes="sxyz"):
        """Transform rotation in euler degrees

        :getter: Return transform's rotaiton in euler degrees
        :type: list
        """
        return np.rad2deg(t3d.euler.mat2euler(self.matrix, axes=axes))

    @property
    def T(self):
        """Transpose of the transform

        :returns: Transpose of the transform
        :rtype: Transform

        """
        try:
            return Transform(self.matrix.T)
        except ValueError:
            print("Can not transpose the Transform matrix")

    @property
    def inverse(self):
        """Inverse of the transform

        :returns: Inverse of the transform
        :rtype: Transform

        """
        try:
            return Transform.from_Rt(
                self.rotation.T, np.dot(-self.rotation.T, self.translation)
            )
        except ValueError:
            print("Can not inverse the Transform matrix")

    def apply(self, points):
        """Apply transform to a list of points

        :param points: List of points (N,3) or (N,4)
        :returns: List of points witht the transform applied
        :rtype: list

        """
        points_4d = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
        transformed_4d = points_4d.dot(self.matrix.T)
        return np.hstack([transformed_4d[:, :3], points[:, 3:]])

    def interpolate(self, other, factor):
        """Interpotation of the transform

        :param other: Transform to interpolate with
        :type other: Transform
        :param factor: Factor of interpolation
        :type factor: float between 0 and 1
        :returns: Transform resulted from the interpolation
        :rtype: Transform

        """
        assert 0 <= factor <= 1.0
        other = Transform(other)
        return self.from_Rt(
            Quaternion.slerp(self.quaternion, other.quaternion, factor),
            self.position + factor * (other.position - self.position),
        )

    def __array__(self):
        return self.matrix

    def __getitem__(self, values):
        return self.matrix.__getitem__(values)

    def __add__(self, other):
        return Transform(other) @ self

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            return self.apply(other)
        return Transform(self.matrix @ Transform(other).matrix)

    def __eq__(self, other):
        return np.allclose(self.matrix, other.matrix)

    def __repr__(self):
        return "R=%s t=%s" % (
            np.array_str(self.euler_degrees, precision=3, suppress_small=True),
            np.array_str(self.position, precision=3, suppress_small=True),
        )
