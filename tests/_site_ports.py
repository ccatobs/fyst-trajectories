"""FYST site variants that differ only in the Nasmyth port."""

import dataclasses

from fyst_trajectories.site import (
    get_fyst_site,
)


def _site_with_port(port):
    """Return the FYST site with only its Nasmyth port changed."""
    return dataclasses.replace(get_fyst_site(), nasmyth_port=port)
