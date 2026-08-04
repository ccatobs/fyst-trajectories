Overhead Models
===============

Data model for observation scheduling: the
:class:`~fyst_trajectories.overhead.ObservingPatch` inputs to
:func:`~fyst_trajectories.overhead.generate_timeline`, the
:class:`~fyst_trajectories.overhead.TimelineBlock` entries it returns,
and the overhead/calibration parameters that shape them.

ObservingPatch
--------------

.. autoclass:: fyst_trajectories.overhead.ObservingPatch
   :members:
   :undoc-members:

BlockType
---------

.. autoclass:: fyst_trajectories.overhead.BlockType
   :members:
   :undoc-members:

CalibrationType
----------------

.. autoclass:: fyst_trajectories.overhead.CalibrationType
   :members:
   :undoc-members:

CalibrationSpec
---------------

.. autoclass:: fyst_trajectories.overhead.CalibrationSpec
   :members:

TimelineBlock
-------------

.. autoclass:: fyst_trajectories.overhead.TimelineBlock
   :members:
   :undoc-members:

OverheadModel
-------------

.. autoclass:: fyst_trajectories.overhead.OverheadModel
   :members:

CalibrationPolicy
-----------------

.. autoclass:: fyst_trajectories.overhead.CalibrationPolicy
   :members:

CalibrationState
-----------------

.. autoclass:: fyst_trajectories.overhead.CalibrationState
   :members:

ObservingTimeline
-----------------

.. autoclass:: fyst_trajectories.overhead.ObservingTimeline
   :members:

Typed Metadata Schemas
----------------------

Block metadata and patch ``scan_params`` follow scan-type-specific
schemas.

ScienceBlockMetadata
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fyst_trajectories.overhead.ScienceBlockMetadata
   :members:

CalibrationBlockMetadata
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fyst_trajectories.overhead.CalibrationBlockMetadata
   :members:

CEScanParams
~~~~~~~~~~~~

.. autoclass:: fyst_trajectories.overhead.CEScanParams
   :members:

PongScanParams
~~~~~~~~~~~~~~

.. autoclass:: fyst_trajectories.overhead.PongScanParams
   :members:

DaisyScanParams
~~~~~~~~~~~~~~~

.. autoclass:: fyst_trajectories.overhead.DaisyScanParams
   :members:

SourceCESScanParams
~~~~~~~~~~~~~~~~~~~

.. autoclass:: fyst_trajectories.overhead.SourceCESScanParams
   :members:

ScanParamsDict
~~~~~~~~~~~~~~

.. autodata:: fyst_trajectories.overhead.ScanParamsDict
   :annotation: = CEScanParams | PongScanParams | DaisyScanParams

   Umbrella union alias for the ``scan_params`` mapping carried on
   :class:`ObservingPatch` and :class:`ScienceBlockMetadata`. The
   concrete schema depends on the patch's ``scan_type``. All keys
   are optional, so any subset (including ``{}``) is valid.
   Typos and scan-type/parameter mismatches can be caught with
   :func:`validate_scan_params`, whose accepted set is one wider than
   this union: it also validates ``"source_ces"`` params recorded on
   planet-calibration blocks, which this union deliberately omits. See
   :ref:`scan-type-vocabularies`.

EmptyBlockMetadata
~~~~~~~~~~~~~~~~~~

.. autoclass:: fyst_trajectories.overhead.EmptyBlockMetadata
   :members:

TimelineBlockMetadata
~~~~~~~~~~~~~~~~~~~~~

.. autodata:: fyst_trajectories.overhead.TimelineBlockMetadata
   :annotation: = ScienceBlockMetadata | CalibrationBlockMetadata | EmptyBlockMetadata

   Exhaustive union of metadata shapes a :class:`TimelineBlock` may carry, one
   variant per :class:`BlockType` member.

Validators
----------

.. autofunction:: fyst_trajectories.overhead.validate_scan_params
