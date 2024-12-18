class SamariaError(Exception):
    """Base exception for samaria_contrast."""

    pass


class ModelError(SamariaError):
    """Raised when there's an error with model operations."""

    pass


class TokenizerError(SamariaError):
    """Raised when there's an error with tokenization."""

    pass


class ConfigError(SamariaError):
    """Raised when there's an error with configuration."""

    pass
