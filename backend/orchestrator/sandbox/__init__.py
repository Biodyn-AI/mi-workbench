"""Confinement helpers for :mod:`backend.orchestrator.code_executor`.

Standard library only.  Modules:

* :mod:`.limits`       - POSIX rlimit planning and the exec launcher
* :mod:`.seatbelt`     - macOS ``sandbox-exec`` profile/command construction
  (template: ``seatbelt_profile.sb``)
* :mod:`.docker`       - ``docker run`` command construction and checks
* :mod:`.interpreter`  - interpreter resolution and its read paths
* :mod:`.capture`      - bounded output capture and safe file capture
"""
