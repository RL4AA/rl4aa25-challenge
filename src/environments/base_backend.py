from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class TransverseTuningBaseBackend(ABC):
    """Abstract class for a backend imlementation of the ARES Experimental Area."""

    @abstractmethod
    def is_beam_on_screen(self) -> bool:
        """
        Return `True` when the beam is on the screen and `False` when it isn't.

        Override with backend-specific imlementation. Must be implemented!
        """
        pass

    def setup(self) -> None:
        """
        Prepare the accelerator for use with the environment. Should mostly be used for
        setting up simulations.

        Override with backend-specific imlementation. Optional.
        """
        pass

    @abstractmethod
    def get_magnets(self) -> np.ndarray:
        """
        Return the magnet values as a NumPy array in order as the magnets appear in the
        accelerator.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    @abstractmethod
    def set_magnets(self, values: Union[np.ndarray, list]) -> None:
        """
        Set the magnets to the given values.

        The argument `magnets` will be passed as a NumPy array in the order the magnets
        appear in the accelerator.

        When applicable, this method should block until the magnet values are acutally
        set!

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Code that should set the accelerator up for a new episode. Run when the `reset`
        is called.

        Mostly meant for simulations to switch to a new incoming beam / misalignments or
        simular things.

        Override with backend-specific imlementation. Optional.
        """
        pass

    def update(self) -> None:
        """
        Update accelerator metrics for later use. Use this to run the simulation or
        cache the beam image.

        Override with backend-specific imlementation. Optional.
        """
        pass

    @abstractmethod
    def get_beam_parameters(self) -> np.ndarray:
        """
        Get the beam parameters measured on the diagnostic screen as NumPy array grouped
        by dimension (e.g. mu_x, sigma_x, mu_y, sigma_y).

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def get_incoming_parameters(self) -> np.ndarray:
        """
        Get all physical beam parameters of the incoming beam as NumPy array in order
        energy, mu_x, mu_xp, mu_y, mu_yp, sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s,
        sigma_p.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_misalignments(self) -> np.ndarray:
        """
        Get misalignments of the quadrupoles and the diagnostic screen as NumPy array in
        order AREAMQZM1.misalignment.x, AREAMQZM1.misalignment.y,
        AREAMQZM2.misalignment.x, AREAMQZM2.misalignment.y, AREAMQZM3.misalignment.x,
        AREAMQZM3.misalignment.y, AREABSCR1.misalignment.x, AREABSCR1.misalignment.y.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_screen_image(self) -> np.ndarray:
        """
        Retreive the beam image as a 2-dimensional NumPy array.

        Note that if reading the beam image is expensive, it is best to cache the image
        in the `update_accelerator` method and the read the cached variable here.

        Ideally, the pixel values should look somewhat similar to the 12-bit values from
        the real screen camera.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    @abstractmethod
    def get_binning(self) -> np.ndarray:
        """
        Return binning currently set on the screen camera as NumPy array [x, y].

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    @abstractmethod
    def get_screen_resolution(self) -> np.ndarray:
        """
        Return (binned) resolution of the screen camera as NumPy array [x, y].

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    @abstractmethod
    def get_pixel_size(self) -> np.ndarray:
        """
        Return the (binned) size of the area on the diagnostic screen covered by one
        pixel as NumPy array [x, y].

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def get_info(self) -> dict:
        """
        Return a dictionary of aditional info from the accelerator backend, e.g.
        incoming beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}
