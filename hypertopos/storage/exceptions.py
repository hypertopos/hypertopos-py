from __future__ import annotations


class GDSError(Exception):
    """Base error for the entire hypertopos / GDS framework."""
    pass


class GDSStorageError(GDSError):
    """Storage-layer errors (files, I/O)."""
    pass


class GDSMissingFileError(GDSStorageError):
    """An expected data file was not found."""
    pass


class GDSCorruptedFileError(GDSStorageError):
    """A data file exists but its content is invalid or unreadable."""
    pass


class GDSVersionError(GDSError):
    """Version mismatch or requested version not found."""
    pass
