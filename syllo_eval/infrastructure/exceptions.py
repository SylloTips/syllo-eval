"""
Custom exceptions for the infrastructure layer.
"""

from typing import Any, Optional


class InfrastructureError(Exception):
  """Base exception for infrastructure-related errors."""

  pass


class PersistenceError(InfrastructureError):
  """Base exception for persistence-related errors."""

  pass


class ConnectionError(PersistenceError):
  """Raised when database connection fails."""

  def __init__(self, message: str, original_error: Optional[Exception] = None):
    super().__init__(message)
    self.original_error = original_error


class NotFoundError(PersistenceError):
  """Raised when a requested entity is not found."""

  def __init__(self, entity_type: str, identifier: Any):
    self.entity_type = entity_type
    self.identifier = identifier
    super().__init__(f'{entity_type} with identifier {identifier} not found')


class DuplicateError(PersistenceError):
  """Raised when attempting to create a duplicate entity."""

  def __init__(self, entity_type: str, constraint: str, details: Optional[str] = None):
    self.entity_type = entity_type
    self.constraint = constraint
    message = f"Duplicate {entity_type}: constraint '{constraint}' violated"
    if details:
      message += f' ({details})'
    super().__init__(message)


class ValidationError(PersistenceError):
  """Raised when entity validation fails."""

  def __init__(self, entity_type: str, field: str, message: str):
    self.entity_type = entity_type
    self.field = field
    super().__init__(f'Validation error for {entity_type}.{field}: {message}')


class TransactionError(PersistenceError):
  """Raised when a transaction fails."""

  def __init__(self, message: str, original_error: Optional[Exception] = None):
    super().__init__(message)
    self.original_error = original_error


class IntegrityError(PersistenceError):
  """Raised when referential integrity is violated."""

  def __init__(
    self,
    entity_type: str,
    foreign_key: str,
    referenced_entity: str,
    identifier: Any,
  ):
    self.entity_type = entity_type
    self.foreign_key = foreign_key
    self.referenced_entity = referenced_entity
    self.identifier = identifier
    super().__init__(
      f'Integrity error in {entity_type}: '
      f'{foreign_key} references non-existent {referenced_entity} '
      f'with id {identifier}'
    )


class QueryError(PersistenceError):
  """Raised when a database query fails."""

  def __init__(self, query: str, error: Exception):
    self.query = query
    self.original_error = error
    super().__init__(f'Query failed: {error}')


class ConfigurationError(InfrastructureError):
  """Raised when infrastructure configuration is invalid."""

  def __init__(self, component: str, field: str, message: str):
    self.component = component
    self.field = field
    super().__init__(f'Configuration error for {component}.{field}: {message}')


class ExternalServiceError(InfrastructureError):
  """Raised when an external infrastructure service operation fails."""

  def __init__(self, service: str, operation: str, original_error: Optional[Exception] = None):
    self.service = service
    self.operation = operation
    self.original_error = original_error
    message = f'External service error in {service}.{operation}'
    if original_error is not None:
      message += f': {original_error}'
    super().__init__(message)


class DataMappingError(InfrastructureError):
  """Raised when external data cannot be mapped to expected in-memory format."""

  def __init__(self, source: str, message: str, original_error: Optional[Exception] = None):
    self.source = source
    self.original_error = original_error
    full_message = f'Data mapping error from {source}: {message}'
    if original_error is not None:
      full_message += f' ({original_error})'
    super().__init__(full_message)
