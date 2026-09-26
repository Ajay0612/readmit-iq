"""Sanitized domain errors: no input values, identifiers, paths or traceback text."""


class ServingError(Exception):
    def __init__(self, code, message, status=503, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or []


class ModelIntegrityError(ServingError):
    pass
