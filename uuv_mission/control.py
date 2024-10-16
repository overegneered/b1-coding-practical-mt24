import numpy as np
import numpy.typing as npt

class Controller:
    """
    Discrete PD controller for UUV.

    Parameters
    ----------
    proportional_gain : float
        The proportional gain of the PD controller.
    derivative_gain : float
        The derivative gain of the PD controller.
    reference : NDArray[np.float64]
        The discrete reference track to follow, with the same reference datum as the UUV
        depth measurement.
    """
    def __init__(
            self,
            proportional_gain: float,
            derivative_gain: float,
            reference: npt.NDArray[np.float64],
    ) -> None:
        self.proportional_gain = proportional_gain
        self.derivative_gain = derivative_gain
        self.depth_reference = reference

        self.x_position: float = 0.0
        self.previous_error: float = 0.0

    def step(self, current_depth: float, current_velocity: float = 1.0) -> float:
        """Step the controller, returning the new control input.

        Parameters
        ----------
        current_depth : float
            The measured depth of the UUV.
        current_velocity : float
            The measured velocity of the UUV. Defaults to one unit per step if not
            supplied.

        Returns
        -------
        The new control input.

        Notes
        -----
        Assumes the UUV is travelling horizontally (in the :math:`x`-direction) at one
        unit per step.
        """
        closest_x_position = int(np.round(self.x_position))
        error: float = self.depth_reference[closest_x_position] - current_depth
        self.x_position += current_velocity

        proportional = self.proportional_gain * error
        derivative = self.derivative_gain * (error - self.previous_error)
        self.previous_error = error

        return proportional + derivative

    def reset(self) -> None:
        self.x_position = 0
        self.previous_error = 0.0
