class FGAError(Exception):
    """Base error for FileGraph Agents."""


class ToolError(FGAError):
    """Tool invocation failed."""


class PermissionDenied(ToolError):
    """An actor attempted an operation outside its authority."""
