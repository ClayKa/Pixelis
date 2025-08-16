# core.utils

## Classes

### class `ColoredFormatter`

```python
ColoredFormatter(fmt=None, datefmt=None, style='%', validate=True, *, defaults=None)
```

Colored formatter for console output.

#### Methods

##### `__init__(self, fmt=None, datefmt=None, style='%', validate=True, *, defaults=None)`

Initialize the formatter with specified format strings.

Initialize the formatter either with the specified format string, or a
default as described above. Allow for specialized date formatting with
the optional datefmt argument. If datefmt is omitted, you get an
ISO8601-like (or RFC 3339-like) format.

Use a style parameter of '%', '{' or '$' to specify that you want to
use one of %-formatting, :meth:`str.format` (``{}``) formatting or
:class:`string.Template` formatting in your format string.

.. versionchanged:: 3.2
   Added the ``style`` parameter.

##### `format(self, record)`

Format the log record with colors.

##### `formatException(self, ei)`

Format and return the specified exception information as a string.

This default implementation just uses
traceback.print_exception()

##### `formatMessage(self, record)`

##### `formatStack(self, stack_info)`

This method is provided as an extension point for specialized
formatting of stack information.

The input data is a string as returned from a call to
:func:`traceback.print_stack`, but with the last trailing newline
removed.

The base implementation just returns the value passed in.

##### `formatTime(self, record, datefmt=None)`

Return the creation time of the specified LogRecord as formatted text.

This method should be called from format() by a formatter which
wants to make use of a formatted time. This method can be overridden
in formatters to provide for any specific requirement, but the
basic behaviour is as follows: if datefmt (a string) is specified,
it is used with time.strftime() to format the creation time of the
record. Otherwise, an ISO8601-like (or RFC 3339-like) format is used.
The resulting string is returned. This function uses a user-configurable
function to convert the creation time to a tuple. By default,
time.localtime() is used; to change this for a particular formatter
instance, set the 'converter' attribute to a function with the same
signature as time.localtime() or time.gmtime(). To change it for all
formatters, for example if you want all logging times to be shown in GMT,
set the 'converter' attribute in the Formatter class.

##### `usesTime(self)`

Check if the format uses the creation time of the record.

---

## Functions

### `add_file_handler(log_file: pathlib.Path) -> None`

Add a file handler to the root logger.

Args:
    log_file: Path to log file

---

### `disable_third_party_logs() -> None`

Disable verbose logging from third-party libraries.

---

### `enable_colored_logging() -> None`

Enable colored logging for console output.

---

### `get_logger(name: str) -> logging.Logger`

Get a logger instance with the given name.

Args:
    name: Logger name (typically __name__)

Returns:
    Logger instance

---

### `log_system_info(logger: Optional[logging.Logger] = None) -> None`

Log system information for debugging.

Args:
    logger: Logger to use (default: root logger)

---

### `set_log_level(level: int) -> None`

Set the global logging level.

Args:
    level: Logging level (e.g., logging.DEBUG)

---

### `setup_logging(level: Optional[int] = None, log_file: Optional[pathlib.Path] = None, format_str: Optional[str] = None) -> None`

Configure global logging settings.

Args:
    level: Logging level (e.g., logging.INFO)
    log_file: Optional file to write logs to
    format_str: Custom format string for log messages

---

