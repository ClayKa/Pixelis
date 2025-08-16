# core.modules

## Classes

### class `ActionEncoder`

```python
ActionEncoder(vocab_size: Optional[int] = None, continuous_dim: Optional[int] = None, output_dim: int = 128, hidden_dim: int = 256, device: str = 'cuda')
```

Encoder for action representations.

Transforms discrete or continuous actions into
dense representations for dynamics modeling.

#### Methods

##### `__call__(self, *args, **kwargs)`

##### `__delattr__(self, name)`

Implement delattr(self, name).

##### `__dir__(self)`

Default dir() implementation.

##### `__getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]`

##### `__getstate__(self)`

Helper for pickle.

##### `__init__(self, vocab_size: Optional[int] = None, continuous_dim: Optional[int] = None, output_dim: int = 128, hidden_dim: int = 256, device: str = 'cuda')`

Initialize action encoder.

Args:
    vocab_size: Size of action vocabulary (for discrete actions)
    continuous_dim: Dimension of continuous actions
    output_dim: Output representation dimension
    hidden_dim: Hidden layer dimension
    device: Device for computation

##### `__repr__(self)`

Return repr(self).

##### `__setattr__(self, name: str, value: Union[torch.Tensor, ForwardRef('Module')]) -> None`

Implement setattr(self, name, value).

##### `__setstate__(self, state)`

##### `add_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Args:
    name (str): name of the child module. The child module can be
        accessed from this module using the given name
    module (Module): child module to be added to the module.

##### `apply(self: ~T, fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T`

Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

Args:
    fn (:class:`Module` -> None): function to be applied to each submodule

Returns:
    Module: self

Example::

    >>> @torch.no_grad()
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )

##### `bfloat16(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``bfloat16`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `buffers(self, recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]`

Return an iterator over module buffers.

Args:
    recurse (bool): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module.

Yields:
    torch.Tensor: module buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for buf in model.buffers():
    >>>     print(type(buf), buf.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `children(self) -> collections.abc.Iterator['Module']`

Return an iterator over immediate children modules.

Yields:
    Module: a child module

##### `compile(self, *args, **kwargs)`

Compile this Module's forward using :func:`torch.compile`.

This Module's `__call__` method is compiled and all arguments are passed as-is
to :func:`torch.compile`.

See :func:`torch.compile` for details on the arguments for this function.

##### `cpu(self: ~T) -> ~T`

Move all model parameters and buffers to the CPU.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `cuda(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

.. note::
    This method modifies the module in-place.

Args:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `double(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``double`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `eval(self: ~T) -> ~T`

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

See :ref:`locally-disable-grad-doc` for a comparison between
`.eval()` and several similar mechanisms that may be confused with it.

Returns:
    Module: self

##### `extra_repr(self) -> str`

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

##### `float(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``float`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `forward(self, action: torch.Tensor) -> torch.Tensor`

Encode action to representation.

Args:
    action: Action tensor (indices for discrete, values for continuous)
    
Returns:
    Encoded action representation

##### `get_buffer(self, target: str) -> 'Tensor'`

Return the buffer given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the buffer
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.Tensor: The buffer referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not a
        buffer

##### `get_extra_state(self) -> Any`

Return any extra state to include in the module's state_dict.

Implement this and a corresponding :func:`set_extra_state` for your module
if you need to store extra state. This function is called when building the
module's `state_dict()`.

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:
    object: Any extra state to store in the module's state_dict

##### `get_parameter(self, target: str) -> 'Parameter'`

Return the parameter given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the Parameter
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Parameter: The Parameter referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not an
        ``nn.Parameter``

##### `get_submodule(self, target: str) -> 'Module'`

Return the submodule given by ``target`` if it exists, otherwise throw an error.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
            )
            (linear): Linear(in_features=100, out_features=200, bias=True)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To check whether or not we have the ``linear`` submodule, we
would call ``get_submodule("net_b.linear")``. To check whether
we have the ``conv`` submodule, we would call
``get_submodule("net_b.net_c.conv")``.

The runtime of ``get_submodule`` is bounded by the degree
of module nesting in ``target``. A query against
``named_modules`` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, ``get_submodule`` should always be
used.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Module: The submodule referenced by ``target``

Raises:
    AttributeError: If at any point along the path resulting from
        the target string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `half(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``half`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `ipu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `load_state_dict(self, state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)`

Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

Args:
    state_dict (dict): a dict containing parameters and
        persistent buffers.
    strict (bool, optional): whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    assign (bool, optional): When set to ``False``, the properties of the tensors
        in the current module are preserved whereas setting it to ``True`` preserves
        properties of the Tensors in the state dict. The only
        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
        for which the value from the module is preserved.
        Default: ``False``

Returns:
    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
        * **missing_keys** is a list of str containing any keys that are expected
            by this module but missing from the provided ``state_dict``.
        * **unexpected_keys** is a list of str containing the keys that are not
            expected by this module but present in the provided ``state_dict``.

Note:
    If a parameter or buffer is registered as ``None`` and its corresponding key
    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    ``RuntimeError``.

##### `modules(self) -> collections.abc.Iterator['Module']`

Return an iterator over all modules in the network.

Yields:
    Module: a module in the network

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
    ...     print(idx, '->', m)

    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)

##### `mtia(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]`

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Args:
    prefix (str): prefix to prepend to all buffer names.
    recurse (bool, optional): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module. Defaults to True.
    remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

Yields:
    (str, torch.Tensor): Tuple containing the name and buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, buf in self.named_buffers():
    >>>     if name in ['running_var']:
    >>>         print(buf.size())

##### `named_children(self) -> collections.abc.Iterator[tuple[str, 'Module']]`

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:
    (str, Module): Tuple containing a name and child module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)

##### `named_modules(self, memo: Optional[set['Module']] = None, prefix: str = '', remove_duplicate: bool = True)`

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Args:
    memo: a memo to store the set of modules already added to the result
    prefix: a prefix that will be added to the name of the module
    remove_duplicate: whether to remove the duplicated module instances in the result
        or not

Yields:
    (str, Module): Tuple of name and module

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
    ...     print(idx, '->', m)

    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

##### `named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]`

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Args:
    prefix (str): prefix to prepend to all parameter names.
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.
    remove_duplicate (bool, optional): whether to remove the duplicated
        parameters in the result. Defaults to True.

Yields:
    (str, Parameter): Tuple containing the name and parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, param in self.named_parameters():
    >>>     if name in ['bias']:
    >>>         print(param.size())

##### `parameters(self, recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]`

Return an iterator over module parameters.

This is typically passed to an optimizer.

Args:
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.

Yields:
    Parameter: module parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for param in model.parameters():
    >>>     print(type(param), param.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `register_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
the behavior of this function will change in future versions.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None`

Add a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting :attr:`persistent` to ``False``. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
:attr:`state_dict`.

Buffers can be accessed as attributes using given names.

Args:
    name (str): name of the buffer. The buffer can be accessed
        from this module using the given name
    tensor (Tensor or None): buffer to be registered. If ``None``, then operations
        that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
        the buffer is **not** included in the module's :attr:`state_dict`.
    persistent (bool): whether the buffer is part of this module's
        :attr:`state_dict`.

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> self.register_buffer('running_mean', torch.zeros(num_features))

##### `register_forward_hook(self, hook: Union[Callable[[~T, tuple[Any, ...], Any], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]]], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.

If ``with_kwargs`` is ``False`` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after :func:`forward` is called. The hook
should have the following signature::

    hook(module, args, output) -> None or modified output

If ``with_kwargs`` is ``True``, the forward hook will be passed the
``kwargs`` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature::

    hook(module, args, kwargs, output) -> None or modified output

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If ``True``, the provided ``hook`` will be fired
        before all existing ``forward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``forward`` hooks registered with
        :func:`register_module_forward_hook` will fire before all hooks
        registered by this method.
        Default: ``False``
    with_kwargs (bool): If ``True``, the ``hook`` will be passed the
        kwargs given to the forward function.
        Default: ``False``
    always_call (bool): If ``True`` the ``hook`` will be run regardless of
        whether an exception is raised while calling the Module.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_forward_pre_hook(self, hook: Union[Callable[[~T, tuple[Any, ...]], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any]], Optional[tuple[Any, dict[str, Any]]]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.


If ``with_kwargs`` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature::

    hook(module, args) -> None or modified input

If ``with_kwargs`` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature::

    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``forward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``forward_pre`` hooks registered with
        :func:`register_module_forward_pre_hook` will fire before all
        hooks registered by this method.
        Default: ``False``
    with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
        given to the forward function.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module
are computed, i.e. the hook will execute if and only if the gradients with
respect to module outputs are computed. The hook should have the following
signature::

    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of :attr:`grad_input` in
subsequent computations. :attr:`grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs or outputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``backward`` hooks registered with
        :func:`register_module_full_backward_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_pre_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature::

    hook(module, grad_output) -> tuple[Tensor] or None

The :attr:`grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of :attr:`grad_output` in
subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``backward_pre`` hooks registered with
        :func:`register_module_full_backward_pre_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_post_hook(self, hook)`

Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, incompatible_keys) -> None

The ``module`` argument is the current module that this hook is registered
on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
is a ``list`` of ``str`` containing the missing keys and
``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling :func:`load_state_dict` with
``strict=True`` are affected by modifications the hook makes to
``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
set of keys will result in an error being thrown when ``strict=True``, and
clearing out both missing and unexpected keys will avoid an error.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_pre_hook(self, hook)`

Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

Arguments:
    hook (Callable): Callable hook that will be invoked before
        loading the state dict.

##### `register_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Alias for :func:`add_module`.

##### `register_parameter(self, name: str, param: Optional[torch.nn.parameter.Parameter]) -> None`

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Args:
    name (str): name of the parameter. The parameter can be accessed
        from this module using the given name
    param (Parameter or None): parameter to be added to the module. If
        ``None``, then operations that run on parameters, such as :attr:`cuda`,
        are ignored. If ``None``, the parameter is **not** included in the
        module's :attr:`state_dict`.

##### `register_state_dict_post_hook(self, hook)`

Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the ``state_dict`` inplace.

##### `register_state_dict_pre_hook(self, hook)`

Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the ``state_dict``
call is made.

##### `requires_grad_(self: ~T, requires_grad: bool = True) -> ~T`

Change if autograd should record operations on parameters in this module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See :ref:`locally-disable-grad-doc` for a comparison between
`.requires_grad_()` and several similar mechanisms that may be confused with it.

Args:
    requires_grad (bool): whether autograd should record operations on
                          parameters in this module. Default: ``True``.

Returns:
    Module: self

##### `set_extra_state(self, state: Any) -> None`

Set extra state contained in the loaded `state_dict`.

This function is called from :func:`load_state_dict` to handle any extra state
found within the `state_dict`. Implement this function and a corresponding
:func:`get_extra_state` for your module if you need to store extra state within its
`state_dict`.

Args:
    state (dict): Extra state from the `state_dict`

##### `set_submodule(self, target: str, module: 'Module', strict: bool = False) -> None`

Set the submodule given by ``target`` if it exists, otherwise throw an error.

.. note::
    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
    the method will only attempt to replace an existing submodule and throw an error if
    the submodule does not exist.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(3, 3, 3)
            )
            (linear): Linear(3, 3)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To override the ``Conv2d`` with a new submodule ``Linear``, you
could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
where ``strict`` could be ``True`` or ``False``

To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

In the above if you set ``strict=True`` and call
``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
will be raised because ``net_b`` does not have a submodule named ``conv``.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)
    module: The module to set the submodule to.
    strict: If ``False``, the method will replace an existing submodule
        or create a new submodule if the parent module exists. If ``True``,
        the method will only attempt to replace an existing submodule and throw an error
        if the submodule doesn't already exist.

Raises:
    ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
    AttributeError: If at any point along the path resulting from
        the ``target`` string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `share_memory(self: ~T) -> ~T`

See :meth:`torch.Tensor.share_memory_`.

##### `state_dict(self, *args, destination=None, prefix='', keep_vars=False)`

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to ``None`` are not included.

.. note::
    The returned object is a shallow copy. It contains references
    to the module's parameters and buffers.

.. warning::
    Currently ``state_dict()`` also accepts positional arguments for
    ``destination``, ``prefix`` and ``keep_vars`` in order. However,
    this is being deprecated and keyword arguments will be enforced in
    future releases.

.. warning::
    Please avoid the use of argument ``destination`` as it is not
    designed for end-users.

Args:
    destination (dict, optional): If provided, the state of module will
        be updated into the dict and the same object is returned.
        Otherwise, an ``OrderedDict`` will be created and returned.
        Default: ``None``.
    prefix (str, optional): a prefix added to parameter and buffer
        names to compose the keys in state_dict. Default: ``''``.
    keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
        returned in the state dict are detached from autograd. If it's
        set to ``True``, detaching will not be performed.
        Default: ``False``.

Returns:
    dict:
        a dictionary containing a whole state of the module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> module.state_dict().keys()
    ['bias', 'weight']

##### `to(self, *args, **kwargs)`

Move and/or cast the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)
   :noindex:

.. function:: to(dtype, non_blocking=False)
   :noindex:

.. function:: to(tensor, non_blocking=False)
   :noindex:

.. function:: to(memory_format=torch.channels_last)
   :noindex:

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point or complex :attr:`dtype`\ s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

Args:
    device (:class:`torch.device`): the desired device of the parameters
        and buffers in this module
    dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
        the parameters and buffers in this module
    tensor (torch.Tensor): Tensor whose dtype and device are the desired
        dtype and device for all parameters and buffers in this module
    memory_format (:class:`torch.memory_format`): the desired memory
        format for 4D parameters and buffers in this module (keyword
        only argument)

Returns:
    Module: self

Examples::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)

    >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.3741+0.j,  0.2382+0.j],
            [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
    >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
    tensor([[0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

##### `to_empty(self: ~T, *, device: Union[int, str, torch.device, NoneType], recurse: bool = True) -> ~T`

Move the parameters and buffers to the specified device without copying storage.

Args:
    device (:class:`torch.device`): The desired device of the parameters
        and buffers in this module.
    recurse (bool): Whether parameters and buffers of submodules should
        be recursively moved to the specified device.

Returns:
    Module: self

##### `train(self: ~T, mode: bool = True) -> ~T`

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

Args:
    mode (bool): whether to set training mode (``True``) or evaluation
                 mode (``False``). Default: ``True``.

Returns:
    Module: self

##### `type(self: ~T, dst_type: Union[torch.dtype, str]) -> ~T`

Casts all parameters and buffers to :attr:`dst_type`.

.. note::
    This method modifies the module in-place.

Args:
    dst_type (type or string): the desired type

Returns:
    Module: self

##### `xpu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `zero_grad(self, set_to_none: bool = True) -> None`

Reset gradients of all model parameters.

See similar function under :class:`torch.optim.Optimizer` for more context.

Args:
    set_to_none (bool): instead of setting to zero, set the grads to None.
        See :meth:`torch.optim.Optimizer.zero_grad` for details.

---

### class `Alert`

```python
Alert(alert_id: str, severity: core.modules.alerter.AlertSeverity, component: str, message: str, details: Dict[str, Any], timestamp: datetime.datetime = <factory>) -> None
```

Represents a system alert.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, alert_id: str, severity: core.modules.alerter.AlertSeverity, component: str, message: str, details: Dict[str, Any], timestamp: datetime.datetime = <factory>) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__repr__(self)`

Return repr(self).

##### `to_dict(self) -> Dict[str, Any]`

Convert alert to dictionary.

---

### class `AlertSeverity`

```python
AlertSeverity(*args, **kwds)
```

Alert severity levels.

#### Methods

##### `__contains__(value)`

Return True if `value` is in `cls`.

`value` is in `cls` if:
1) `value` is a member of `cls`, or
2) `value` is the value of one of the `cls`'s members.

##### `__getitem__(name)`

Return the member matching `name`.

##### `__iter__()`

Return members in definition order.

##### `__len__()`

Return the number of members (no aliases)

---

### class `AlertThreshold`

```python
AlertThreshold(metric_name: str, threshold_value: float, comparison: str, severity: core.modules.alerter.AlertSeverity, cooldown_seconds: int = 300, description: str = '') -> None
```

Configuration for an alert threshold.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, metric_name: str, threshold_value: float, comparison: str, severity: core.modules.alerter.AlertSeverity, cooldown_seconds: int = 300, description: str = '') -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__repr__(self)`

Return repr(self).

##### `check(self, value: float) -> bool`

Check if the value exceeds the threshold.

---

### class `Alerter`

```python
Alerter(config: Dict[str, Any])
```

Central alerting system for the Pixelis framework.

Monitors system health metrics and sends alerts when thresholds are exceeded.
Implements cooldown periods to prevent alert spam.

#### Methods

##### `__init__(self, config: Dict[str, Any])`

Initialize the alerter.

Args:
    config: Configuration dictionary containing alert settings

##### `check_metrics(self, metrics: Dict[str, float], component: str = 'system')`

Check metrics against thresholds and send alerts if needed.

Args:
    metrics: Dictionary of metric values
    component: Component name for alert attribution

##### `clear_history(self)`

Clear alert history (useful for testing).

##### `get_recent_alerts(self, limit: int = 10) -> List[core.modules.alerter.Alert]`

Get recent alerts.

Args:
    limit: Maximum number of alerts to return
    
Returns:
    List of recent alerts

##### `get_statistics(self) -> Dict[str, Any]`

Get alerter statistics.

##### `send_alert(self, severity: core.modules.alerter.AlertSeverity, component: str, message: str, details: Optional[Dict[str, Any]] = None)`

Send an alert through configured channels.

Args:
    severity: Alert severity level
    component: Component that triggered the alert
    message: Alert message
    details: Additional alert details

---

### class `AuditEntry`

```python
AuditEntry(timestamp: str = <factory>, event_type: core.modules.audit.AuditEventType = <AuditEventType.SYSTEM_ERROR: 'system_error'>, actor: str = 'system', action: str = '', resource: str = '', result: core.modules.audit.AuditResult = <AuditResult.SUCCESS: 'success'>, metadata: Dict[str, Any] = <factory>, hash_previous: Optional[str] = None, hash_current: Optional[str] = None) -> None
```

Represents a single audit log entry.

Attributes:
    timestamp: When the event occurred
    event_type: Type of event
    actor: Who/what performed the action
    action: What action was performed
    resource: What resource was affected
    result: Result of the action
    metadata: Additional event-specific data
    hash_previous: Hash of the previous entry (for chain integrity)
    hash_current: Hash of this entry

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, timestamp: str = <factory>, event_type: core.modules.audit.AuditEventType = <AuditEventType.SYSTEM_ERROR: 'system_error'>, actor: str = 'system', action: str = '', resource: str = '', result: core.modules.audit.AuditResult = <AuditResult.SUCCESS: 'success'>, metadata: Dict[str, Any] = <factory>, hash_previous: Optional[str] = None, hash_current: Optional[str] = None) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__repr__(self)`

Return repr(self).

##### `calculate_hash(self, previous_hash: Optional[str] = None) -> str`

Calculate cryptographic hash for this entry.

Args:
    previous_hash: Hash of the previous entry in the chain
    
Returns:
    SHA-256 hash of this entry

##### `from_dict(data: Dict[str, Any]) -> 'AuditEntry'`

Create from dictionary.

##### `to_dict(self) -> Dict[str, Any]`

Convert to dictionary for JSON serialization.

---

### class `AuditEventType`

```python
AuditEventType(*args, **kwds)
```

Types of audit events.

#### Methods

##### `__contains__(value)`

Return True if `value` is in `cls`.

`value` is in `cls` if:
1) `value` is a member of `cls`, or
2) `value` is the value of one of the `cls`'s members.

##### `__getitem__(name)`

Return the member matching `name`.

##### `__iter__()`

Return members in definition order.

##### `__len__()`

Return the number of members (no aliases)

---

### class `AuditLogger`

```python
AuditLogger(audit_dir: str = './audit', max_file_size: int = 100000000, retention_days: int = 365, enable_async: bool = True, enable_encryption: bool = False)
```

Centralized audit logger for the Pixelis system.

Implements:
- Append-only logging with cryptographic hash chain
- Automatic rotation and archiving
- Tamper detection and verification
- Asynchronous logging for performance
- Compliance with security policy requirements

#### Methods

##### `__init__(self, audit_dir: str = './audit', max_file_size: int = 100000000, retention_days: int = 365, enable_async: bool = True, enable_encryption: bool = False)`

Initialize the audit logger.

Args:
    audit_dir: Directory for audit logs
    max_file_size: Maximum size per audit file before rotation
    retention_days: How long to retain audit logs
    enable_async: Use asynchronous logging
    enable_encryption: Encrypt audit logs at rest

##### `cleanup_old_logs(self)`

Remove audit logs older than retention period.

##### `get_statistics(self) -> Dict[str, Any]`

Get audit logger statistics.

##### `log(self, event_type: core.modules.audit.AuditEventType, actor: str, action: str, resource: str, result: core.modules.audit.AuditResult = <AuditResult.SUCCESS: 'success'>, metadata: Optional[Dict[str, Any]] = None) -> bool`

Log an audit event.

Args:
    event_type: Type of event
    actor: Who performed the action
    action: What action was performed
    resource: What resource was affected
    result: Result of the action
    metadata: Additional event data
    
Returns:
    True if successfully logged

##### `search(self, start_date: Optional[datetime.datetime] = None, end_date: Optional[datetime.datetime] = None, event_type: Optional[core.modules.audit.AuditEventType] = None, actor: Optional[str] = None, resource: Optional[str] = None, result: Optional[core.modules.audit.AuditResult] = None, limit: int = 1000) -> List[core.modules.audit.AuditEntry]`

Search audit logs with filters.

Args:
    start_date: Start of date range
    end_date: End of date range
    event_type: Filter by event type
    actor: Filter by actor
    resource: Filter by resource
    result: Filter by result
    limit: Maximum results to return
    
Returns:
    List of matching audit entries

##### `shutdown(self)`

Gracefully shutdown the audit logger.

##### `verify_integrity(self, file_path: Optional[pathlib.Path] = None, verbose: bool = False) -> Dict[str, Any]`

Verify the integrity of an audit file.

Args:
    file_path: Path to audit file (uses current if None)
    verbose: Include detailed verification info
    
Returns:
    Verification results

---

### class `AuditResult`

```python
AuditResult(*args, **kwds)
```

Result of an audited action.

#### Methods

##### `__contains__(value)`

Return True if `value` is in `cls`.

`value` is in `cls` if:
1) `value` is a member of `cls`, or
2) `value` is the value of one of the `cls`'s members.

##### `__getitem__(name)`

Return the member matching `name`.

##### `__iter__()`

Return members in definition order.

##### `__len__()`

Return the number of members (no aliases)

---

### class `BaseOperation`

Abstract base class for all visual operations.

All specific operations must inherit from this class and implement
the run method to ensure a consistent interface.

#### Methods

##### `__init__(self)`

Initialize the base operation.

##### `__repr__(self) -> str`

String representation of the operation.

##### `run(self, **kwargs) -> Any`

Execute the operation.

Args:
    **kwargs: Operation-specific arguments
    
Returns:
    Operation result (format depends on specific operation)

##### `validate_inputs(self, **kwargs) -> bool`

Validate input arguments for the operation.

Override in subclasses to provide specific validation.

Args:
    **kwargs: Operation-specific arguments
    
Returns:
    True if inputs are valid, False otherwise

---

### class `CuriosityDynamicsModel`

```python
CuriosityDynamicsModel(state_dim: int = 768, action_dim: int = 128, encoded_dim: int = 256, config: Optional[Dict[str, Any]] = None)
```

Complete dynamics model for curiosity-driven learning.

Combines forward model, inverse model, and encoders
for computing intrinsic rewards.

#### Methods

##### `__call__(self, *args, **kwargs)`

##### `__delattr__(self, name)`

Implement delattr(self, name).

##### `__dir__(self)`

Default dir() implementation.

##### `__getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]`

##### `__getstate__(self)`

Helper for pickle.

##### `__init__(self, state_dim: int = 768, action_dim: int = 128, encoded_dim: int = 256, config: Optional[Dict[str, Any]] = None)`

Initialize complete curiosity dynamics model.

Args:
    state_dim: Raw state dimension
    action_dim: Raw action dimension
    encoded_dim: Encoded representation dimension
    config: Additional configuration

##### `__repr__(self)`

Return repr(self).

##### `__setattr__(self, name: str, value: Union[torch.Tensor, ForwardRef('Module')]) -> None`

Implement setattr(self, name, value).

##### `__setstate__(self, state)`

##### `add_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Args:
    name (str): name of the child module. The child module can be
        accessed from this module using the given name
    module (Module): child module to be added to the module.

##### `apply(self: ~T, fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T`

Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

Args:
    fn (:class:`Module` -> None): function to be applied to each submodule

Returns:
    Module: self

Example::

    >>> @torch.no_grad()
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )

##### `bfloat16(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``bfloat16`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `buffers(self, recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]`

Return an iterator over module buffers.

Args:
    recurse (bool): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module.

Yields:
    torch.Tensor: module buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for buf in model.buffers():
    >>>     print(type(buf), buf.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `children(self) -> collections.abc.Iterator['Module']`

Return an iterator over immediate children modules.

Yields:
    Module: a child module

##### `compile(self, *args, **kwargs)`

Compile this Module's forward using :func:`torch.compile`.

This Module's `__call__` method is compiled and all arguments are passed as-is
to :func:`torch.compile`.

See :func:`torch.compile` for details on the arguments for this function.

##### `compute_intrinsic_reward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]`

Compute intrinsic curiosity reward.

Args:
    state: Current state
    action: Action taken
    next_state: Next state
    
Returns:
    Tuple of (intrinsic_reward, loss_dict)

##### `cpu(self: ~T) -> ~T`

Move all model parameters and buffers to the CPU.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `cuda(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

.. note::
    This method modifies the module in-place.

Args:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `double(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``double`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `eval(self: ~T) -> ~T`

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

See :ref:`locally-disable-grad-doc` for a comparison between
`.eval()` and several similar mechanisms that may be confused with it.

Returns:
    Module: self

##### `extra_repr(self) -> str`

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

##### `float(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``float`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `forward(self, *input: Any) -> None`

Define the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

##### `get_buffer(self, target: str) -> 'Tensor'`

Return the buffer given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the buffer
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.Tensor: The buffer referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not a
        buffer

##### `get_extra_state(self) -> Any`

Return any extra state to include in the module's state_dict.

Implement this and a corresponding :func:`set_extra_state` for your module
if you need to store extra state. This function is called when building the
module's `state_dict()`.

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:
    object: Any extra state to store in the module's state_dict

##### `get_parameter(self, target: str) -> 'Parameter'`

Return the parameter given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the Parameter
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Parameter: The Parameter referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not an
        ``nn.Parameter``

##### `get_submodule(self, target: str) -> 'Module'`

Return the submodule given by ``target`` if it exists, otherwise throw an error.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
            )
            (linear): Linear(in_features=100, out_features=200, bias=True)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To check whether or not we have the ``linear`` submodule, we
would call ``get_submodule("net_b.linear")``. To check whether
we have the ``conv`` submodule, we would call
``get_submodule("net_b.net_c.conv")``.

The runtime of ``get_submodule`` is bounded by the degree
of module nesting in ``target``. A query against
``named_modules`` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, ``get_submodule`` should always be
used.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Module: The submodule referenced by ``target``

Raises:
    AttributeError: If at any point along the path resulting from
        the target string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `half(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``half`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `ipu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `load_state_dict(self, state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)`

Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

Args:
    state_dict (dict): a dict containing parameters and
        persistent buffers.
    strict (bool, optional): whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    assign (bool, optional): When set to ``False``, the properties of the tensors
        in the current module are preserved whereas setting it to ``True`` preserves
        properties of the Tensors in the state dict. The only
        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
        for which the value from the module is preserved.
        Default: ``False``

Returns:
    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
        * **missing_keys** is a list of str containing any keys that are expected
            by this module but missing from the provided ``state_dict``.
        * **unexpected_keys** is a list of str containing the keys that are not
            expected by this module but present in the provided ``state_dict``.

Note:
    If a parameter or buffer is registered as ``None`` and its corresponding key
    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    ``RuntimeError``.

##### `modules(self) -> collections.abc.Iterator['Module']`

Return an iterator over all modules in the network.

Yields:
    Module: a module in the network

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
    ...     print(idx, '->', m)

    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)

##### `mtia(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]`

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Args:
    prefix (str): prefix to prepend to all buffer names.
    recurse (bool, optional): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module. Defaults to True.
    remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

Yields:
    (str, torch.Tensor): Tuple containing the name and buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, buf in self.named_buffers():
    >>>     if name in ['running_var']:
    >>>         print(buf.size())

##### `named_children(self) -> collections.abc.Iterator[tuple[str, 'Module']]`

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:
    (str, Module): Tuple containing a name and child module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)

##### `named_modules(self, memo: Optional[set['Module']] = None, prefix: str = '', remove_duplicate: bool = True)`

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Args:
    memo: a memo to store the set of modules already added to the result
    prefix: a prefix that will be added to the name of the module
    remove_duplicate: whether to remove the duplicated module instances in the result
        or not

Yields:
    (str, Module): Tuple of name and module

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
    ...     print(idx, '->', m)

    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

##### `named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]`

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Args:
    prefix (str): prefix to prepend to all parameter names.
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.
    remove_duplicate (bool, optional): whether to remove the duplicated
        parameters in the result. Defaults to True.

Yields:
    (str, Parameter): Tuple containing the name and parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, param in self.named_parameters():
    >>>     if name in ['bias']:
    >>>         print(param.size())

##### `parameters(self, recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]`

Return an iterator over module parameters.

This is typically passed to an optimizer.

Args:
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.

Yields:
    Parameter: module parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for param in model.parameters():
    >>>     print(type(param), param.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `register_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
the behavior of this function will change in future versions.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None`

Add a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting :attr:`persistent` to ``False``. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
:attr:`state_dict`.

Buffers can be accessed as attributes using given names.

Args:
    name (str): name of the buffer. The buffer can be accessed
        from this module using the given name
    tensor (Tensor or None): buffer to be registered. If ``None``, then operations
        that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
        the buffer is **not** included in the module's :attr:`state_dict`.
    persistent (bool): whether the buffer is part of this module's
        :attr:`state_dict`.

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> self.register_buffer('running_mean', torch.zeros(num_features))

##### `register_forward_hook(self, hook: Union[Callable[[~T, tuple[Any, ...], Any], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]]], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.

If ``with_kwargs`` is ``False`` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after :func:`forward` is called. The hook
should have the following signature::

    hook(module, args, output) -> None or modified output

If ``with_kwargs`` is ``True``, the forward hook will be passed the
``kwargs`` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature::

    hook(module, args, kwargs, output) -> None or modified output

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If ``True``, the provided ``hook`` will be fired
        before all existing ``forward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``forward`` hooks registered with
        :func:`register_module_forward_hook` will fire before all hooks
        registered by this method.
        Default: ``False``
    with_kwargs (bool): If ``True``, the ``hook`` will be passed the
        kwargs given to the forward function.
        Default: ``False``
    always_call (bool): If ``True`` the ``hook`` will be run regardless of
        whether an exception is raised while calling the Module.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_forward_pre_hook(self, hook: Union[Callable[[~T, tuple[Any, ...]], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any]], Optional[tuple[Any, dict[str, Any]]]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.


If ``with_kwargs`` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature::

    hook(module, args) -> None or modified input

If ``with_kwargs`` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature::

    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``forward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``forward_pre`` hooks registered with
        :func:`register_module_forward_pre_hook` will fire before all
        hooks registered by this method.
        Default: ``False``
    with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
        given to the forward function.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module
are computed, i.e. the hook will execute if and only if the gradients with
respect to module outputs are computed. The hook should have the following
signature::

    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of :attr:`grad_input` in
subsequent computations. :attr:`grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs or outputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``backward`` hooks registered with
        :func:`register_module_full_backward_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_pre_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature::

    hook(module, grad_output) -> tuple[Tensor] or None

The :attr:`grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of :attr:`grad_output` in
subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``backward_pre`` hooks registered with
        :func:`register_module_full_backward_pre_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_post_hook(self, hook)`

Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, incompatible_keys) -> None

The ``module`` argument is the current module that this hook is registered
on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
is a ``list`` of ``str`` containing the missing keys and
``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling :func:`load_state_dict` with
``strict=True`` are affected by modifications the hook makes to
``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
set of keys will result in an error being thrown when ``strict=True``, and
clearing out both missing and unexpected keys will avoid an error.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_pre_hook(self, hook)`

Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

Arguments:
    hook (Callable): Callable hook that will be invoked before
        loading the state dict.

##### `register_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Alias for :func:`add_module`.

##### `register_parameter(self, name: str, param: Optional[torch.nn.parameter.Parameter]) -> None`

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Args:
    name (str): name of the parameter. The parameter can be accessed
        from this module using the given name
    param (Parameter or None): parameter to be added to the module. If
        ``None``, then operations that run on parameters, such as :attr:`cuda`,
        are ignored. If ``None``, the parameter is **not** included in the
        module's :attr:`state_dict`.

##### `register_state_dict_post_hook(self, hook)`

Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the ``state_dict`` inplace.

##### `register_state_dict_pre_hook(self, hook)`

Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the ``state_dict``
call is made.

##### `requires_grad_(self: ~T, requires_grad: bool = True) -> ~T`

Change if autograd should record operations on parameters in this module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See :ref:`locally-disable-grad-doc` for a comparison between
`.requires_grad_()` and several similar mechanisms that may be confused with it.

Args:
    requires_grad (bool): whether autograd should record operations on
                          parameters in this module. Default: ``True``.

Returns:
    Module: self

##### `set_extra_state(self, state: Any) -> None`

Set extra state contained in the loaded `state_dict`.

This function is called from :func:`load_state_dict` to handle any extra state
found within the `state_dict`. Implement this function and a corresponding
:func:`get_extra_state` for your module if you need to store extra state within its
`state_dict`.

Args:
    state (dict): Extra state from the `state_dict`

##### `set_submodule(self, target: str, module: 'Module', strict: bool = False) -> None`

Set the submodule given by ``target`` if it exists, otherwise throw an error.

.. note::
    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
    the method will only attempt to replace an existing submodule and throw an error if
    the submodule does not exist.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(3, 3, 3)
            )
            (linear): Linear(3, 3)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To override the ``Conv2d`` with a new submodule ``Linear``, you
could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
where ``strict`` could be ``True`` or ``False``

To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

In the above if you set ``strict=True`` and call
``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
will be raised because ``net_b`` does not have a submodule named ``conv``.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)
    module: The module to set the submodule to.
    strict: If ``False``, the method will replace an existing submodule
        or create a new submodule if the parent module exists. If ``True``,
        the method will only attempt to replace an existing submodule and throw an error
        if the submodule doesn't already exist.

Raises:
    ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
    AttributeError: If at any point along the path resulting from
        the ``target`` string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `share_memory(self: ~T) -> ~T`

See :meth:`torch.Tensor.share_memory_`.

##### `state_dict(self, *args, destination=None, prefix='', keep_vars=False)`

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to ``None`` are not included.

.. note::
    The returned object is a shallow copy. It contains references
    to the module's parameters and buffers.

.. warning::
    Currently ``state_dict()`` also accepts positional arguments for
    ``destination``, ``prefix`` and ``keep_vars`` in order. However,
    this is being deprecated and keyword arguments will be enforced in
    future releases.

.. warning::
    Please avoid the use of argument ``destination`` as it is not
    designed for end-users.

Args:
    destination (dict, optional): If provided, the state of module will
        be updated into the dict and the same object is returned.
        Otherwise, an ``OrderedDict`` will be created and returned.
        Default: ``None``.
    prefix (str, optional): a prefix added to parameter and buffer
        names to compose the keys in state_dict. Default: ``''``.
    keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
        returned in the state dict are detached from autograd. If it's
        set to ``True``, detaching will not be performed.
        Default: ``False``.

Returns:
    dict:
        a dictionary containing a whole state of the module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> module.state_dict().keys()
    ['bias', 'weight']

##### `to(self, *args, **kwargs)`

Move and/or cast the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)
   :noindex:

.. function:: to(dtype, non_blocking=False)
   :noindex:

.. function:: to(tensor, non_blocking=False)
   :noindex:

.. function:: to(memory_format=torch.channels_last)
   :noindex:

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point or complex :attr:`dtype`\ s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

Args:
    device (:class:`torch.device`): the desired device of the parameters
        and buffers in this module
    dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
        the parameters and buffers in this module
    tensor (torch.Tensor): Tensor whose dtype and device are the desired
        dtype and device for all parameters and buffers in this module
    memory_format (:class:`torch.memory_format`): the desired memory
        format for 4D parameters and buffers in this module (keyword
        only argument)

Returns:
    Module: self

Examples::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)

    >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.3741+0.j,  0.2382+0.j],
            [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
    >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
    tensor([[0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

##### `to_empty(self: ~T, *, device: Union[int, str, torch.device, NoneType], recurse: bool = True) -> ~T`

Move the parameters and buffers to the specified device without copying storage.

Args:
    device (:class:`torch.device`): The desired device of the parameters
        and buffers in this module.
    recurse (bool): Whether parameters and buffers of submodules should
        be recursively moved to the specified device.

Returns:
    Module: self

##### `train(self: ~T, mode: bool = True) -> ~T`

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

Args:
    mode (bool): whether to set training mode (``True``) or evaluation
                 mode (``False``). Default: ``True``.

Returns:
    Module: self

##### `type(self: ~T, dst_type: Union[torch.dtype, str]) -> ~T`

Casts all parameters and buffers to :attr:`dst_type`.

.. note::
    This method modifies the module in-place.

Args:
    dst_type (type or string): the desired type

Returns:
    Module: self

##### `xpu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `zero_grad(self, set_to_none: bool = True) -> None`

Reset gradients of all model parameters.

See similar function under :class:`torch.optim.Optimizer` for more context.

Args:
    set_to_none (bool): instead of setting to zero, set the grads to None.
        See :meth:`torch.optim.Optimizer.zero_grad` for details.

---

### class `CuriosityRewardModule`

```python
CuriosityRewardModule(state_dim: int = 768, action_dim: int = 128, hidden_dim: int = 256, beta: float = 0.2, eta: float = 0.5, device: str = 'cuda')
```

Implements curiosity-driven reward based on prediction error.

Uses a lightweight dynamics model to predict next state and
rewards based on prediction error (encouraging exploration).

#### Methods

##### `__call__(self, *args, **kwargs)`

##### `__delattr__(self, name)`

Implement delattr(self, name).

##### `__dir__(self)`

Default dir() implementation.

##### `__getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]`

##### `__getstate__(self)`

Helper for pickle.

##### `__init__(self, state_dim: int = 768, action_dim: int = 128, hidden_dim: int = 256, beta: float = 0.2, eta: float = 0.5, device: str = 'cuda')`

Initialize curiosity module.

Args:
    state_dim: Dimension of state embeddings
    action_dim: Dimension of action embeddings
    hidden_dim: Hidden layer dimension
    beta: Weight for forward model loss
    eta: Scaling factor for intrinsic reward
    device: Device for computation

##### `__repr__(self)`

Return repr(self).

##### `__setattr__(self, name: str, value: Union[torch.Tensor, ForwardRef('Module')]) -> None`

Implement setattr(self, name, value).

##### `__setstate__(self, state)`

##### `add_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Args:
    name (str): name of the child module. The child module can be
        accessed from this module using the given name
    module (Module): child module to be added to the module.

##### `apply(self: ~T, fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T`

Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

Args:
    fn (:class:`Module` -> None): function to be applied to each submodule

Returns:
    Module: self

Example::

    >>> @torch.no_grad()
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )

##### `bfloat16(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``bfloat16`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `buffers(self, recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]`

Return an iterator over module buffers.

Args:
    recurse (bool): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module.

Yields:
    torch.Tensor: module buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for buf in model.buffers():
    >>>     print(type(buf), buf.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `calculate_trajectory_curiosity(self, trajectory: core.data_structures.Trajectory, state_embeddings: List[torch.Tensor]) -> float`

Calculate curiosity reward for entire trajectory.

Args:
    trajectory: Reasoning trajectory
    state_embeddings: List of state embeddings
    
Returns:
    Average curiosity reward

##### `children(self) -> collections.abc.Iterator['Module']`

Return an iterator over immediate children modules.

Yields:
    Module: a child module

##### `compile(self, *args, **kwargs)`

Compile this Module's forward using :func:`torch.compile`.

This Module's `__call__` method is compiled and all arguments are passed as-is
to :func:`torch.compile`.

See :func:`torch.compile` for details on the arguments for this function.

##### `cpu(self: ~T) -> ~T`

Move all model parameters and buffers to the CPU.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `cuda(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

.. note::
    This method modifies the module in-place.

Args:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `double(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``double`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `eval(self: ~T) -> ~T`

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

See :ref:`locally-disable-grad-doc` for a comparison between
`.eval()` and several similar mechanisms that may be confused with it.

Returns:
    Module: self

##### `extra_repr(self) -> str`

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

##### `float(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``float`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]`

Calculate curiosity reward.

Args:
    state: Current state embedding
    action: Action embedding
    next_state: Next state embedding
    
Returns:
    Tuple of (curiosity_reward, loss_dict)

##### `get_buffer(self, target: str) -> 'Tensor'`

Return the buffer given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the buffer
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.Tensor: The buffer referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not a
        buffer

##### `get_extra_state(self) -> Any`

Return any extra state to include in the module's state_dict.

Implement this and a corresponding :func:`set_extra_state` for your module
if you need to store extra state. This function is called when building the
module's `state_dict()`.

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:
    object: Any extra state to store in the module's state_dict

##### `get_parameter(self, target: str) -> 'Parameter'`

Return the parameter given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the Parameter
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Parameter: The Parameter referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not an
        ``nn.Parameter``

##### `get_submodule(self, target: str) -> 'Module'`

Return the submodule given by ``target`` if it exists, otherwise throw an error.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
            )
            (linear): Linear(in_features=100, out_features=200, bias=True)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To check whether or not we have the ``linear`` submodule, we
would call ``get_submodule("net_b.linear")``. To check whether
we have the ``conv`` submodule, we would call
``get_submodule("net_b.net_c.conv")``.

The runtime of ``get_submodule`` is bounded by the degree
of module nesting in ``target``. A query against
``named_modules`` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, ``get_submodule`` should always be
used.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Module: The submodule referenced by ``target``

Raises:
    AttributeError: If at any point along the path resulting from
        the target string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `half(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``half`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `ipu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `load_state_dict(self, state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)`

Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

Args:
    state_dict (dict): a dict containing parameters and
        persistent buffers.
    strict (bool, optional): whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    assign (bool, optional): When set to ``False``, the properties of the tensors
        in the current module are preserved whereas setting it to ``True`` preserves
        properties of the Tensors in the state dict. The only
        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
        for which the value from the module is preserved.
        Default: ``False``

Returns:
    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
        * **missing_keys** is a list of str containing any keys that are expected
            by this module but missing from the provided ``state_dict``.
        * **unexpected_keys** is a list of str containing the keys that are not
            expected by this module but present in the provided ``state_dict``.

Note:
    If a parameter or buffer is registered as ``None`` and its corresponding key
    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    ``RuntimeError``.

##### `modules(self) -> collections.abc.Iterator['Module']`

Return an iterator over all modules in the network.

Yields:
    Module: a module in the network

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
    ...     print(idx, '->', m)

    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)

##### `mtia(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]`

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Args:
    prefix (str): prefix to prepend to all buffer names.
    recurse (bool, optional): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module. Defaults to True.
    remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

Yields:
    (str, torch.Tensor): Tuple containing the name and buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, buf in self.named_buffers():
    >>>     if name in ['running_var']:
    >>>         print(buf.size())

##### `named_children(self) -> collections.abc.Iterator[tuple[str, 'Module']]`

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:
    (str, Module): Tuple containing a name and child module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)

##### `named_modules(self, memo: Optional[set['Module']] = None, prefix: str = '', remove_duplicate: bool = True)`

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Args:
    memo: a memo to store the set of modules already added to the result
    prefix: a prefix that will be added to the name of the module
    remove_duplicate: whether to remove the duplicated module instances in the result
        or not

Yields:
    (str, Module): Tuple of name and module

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
    ...     print(idx, '->', m)

    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

##### `named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]`

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Args:
    prefix (str): prefix to prepend to all parameter names.
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.
    remove_duplicate (bool, optional): whether to remove the duplicated
        parameters in the result. Defaults to True.

Yields:
    (str, Parameter): Tuple containing the name and parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, param in self.named_parameters():
    >>>     if name in ['bias']:
    >>>         print(param.size())

##### `parameters(self, recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]`

Return an iterator over module parameters.

This is typically passed to an optimizer.

Args:
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.

Yields:
    Parameter: module parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for param in model.parameters():
    >>>     print(type(param), param.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `register_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
the behavior of this function will change in future versions.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None`

Add a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting :attr:`persistent` to ``False``. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
:attr:`state_dict`.

Buffers can be accessed as attributes using given names.

Args:
    name (str): name of the buffer. The buffer can be accessed
        from this module using the given name
    tensor (Tensor or None): buffer to be registered. If ``None``, then operations
        that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
        the buffer is **not** included in the module's :attr:`state_dict`.
    persistent (bool): whether the buffer is part of this module's
        :attr:`state_dict`.

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> self.register_buffer('running_mean', torch.zeros(num_features))

##### `register_forward_hook(self, hook: Union[Callable[[~T, tuple[Any, ...], Any], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]]], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.

If ``with_kwargs`` is ``False`` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after :func:`forward` is called. The hook
should have the following signature::

    hook(module, args, output) -> None or modified output

If ``with_kwargs`` is ``True``, the forward hook will be passed the
``kwargs`` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature::

    hook(module, args, kwargs, output) -> None or modified output

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If ``True``, the provided ``hook`` will be fired
        before all existing ``forward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``forward`` hooks registered with
        :func:`register_module_forward_hook` will fire before all hooks
        registered by this method.
        Default: ``False``
    with_kwargs (bool): If ``True``, the ``hook`` will be passed the
        kwargs given to the forward function.
        Default: ``False``
    always_call (bool): If ``True`` the ``hook`` will be run regardless of
        whether an exception is raised while calling the Module.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_forward_pre_hook(self, hook: Union[Callable[[~T, tuple[Any, ...]], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any]], Optional[tuple[Any, dict[str, Any]]]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.


If ``with_kwargs`` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature::

    hook(module, args) -> None or modified input

If ``with_kwargs`` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature::

    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``forward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``forward_pre`` hooks registered with
        :func:`register_module_forward_pre_hook` will fire before all
        hooks registered by this method.
        Default: ``False``
    with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
        given to the forward function.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module
are computed, i.e. the hook will execute if and only if the gradients with
respect to module outputs are computed. The hook should have the following
signature::

    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of :attr:`grad_input` in
subsequent computations. :attr:`grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs or outputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``backward`` hooks registered with
        :func:`register_module_full_backward_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_pre_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature::

    hook(module, grad_output) -> tuple[Tensor] or None

The :attr:`grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of :attr:`grad_output` in
subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``backward_pre`` hooks registered with
        :func:`register_module_full_backward_pre_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_post_hook(self, hook)`

Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, incompatible_keys) -> None

The ``module`` argument is the current module that this hook is registered
on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
is a ``list`` of ``str`` containing the missing keys and
``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling :func:`load_state_dict` with
``strict=True`` are affected by modifications the hook makes to
``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
set of keys will result in an error being thrown when ``strict=True``, and
clearing out both missing and unexpected keys will avoid an error.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_pre_hook(self, hook)`

Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

Arguments:
    hook (Callable): Callable hook that will be invoked before
        loading the state dict.

##### `register_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Alias for :func:`add_module`.

##### `register_parameter(self, name: str, param: Optional[torch.nn.parameter.Parameter]) -> None`

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Args:
    name (str): name of the parameter. The parameter can be accessed
        from this module using the given name
    param (Parameter or None): parameter to be added to the module. If
        ``None``, then operations that run on parameters, such as :attr:`cuda`,
        are ignored. If ``None``, the parameter is **not** included in the
        module's :attr:`state_dict`.

##### `register_state_dict_post_hook(self, hook)`

Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the ``state_dict`` inplace.

##### `register_state_dict_pre_hook(self, hook)`

Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the ``state_dict``
call is made.

##### `requires_grad_(self: ~T, requires_grad: bool = True) -> ~T`

Change if autograd should record operations on parameters in this module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See :ref:`locally-disable-grad-doc` for a comparison between
`.requires_grad_()` and several similar mechanisms that may be confused with it.

Args:
    requires_grad (bool): whether autograd should record operations on
                          parameters in this module. Default: ``True``.

Returns:
    Module: self

##### `set_extra_state(self, state: Any) -> None`

Set extra state contained in the loaded `state_dict`.

This function is called from :func:`load_state_dict` to handle any extra state
found within the `state_dict`. Implement this function and a corresponding
:func:`get_extra_state` for your module if you need to store extra state within its
`state_dict`.

Args:
    state (dict): Extra state from the `state_dict`

##### `set_submodule(self, target: str, module: 'Module', strict: bool = False) -> None`

Set the submodule given by ``target`` if it exists, otherwise throw an error.

.. note::
    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
    the method will only attempt to replace an existing submodule and throw an error if
    the submodule does not exist.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(3, 3, 3)
            )
            (linear): Linear(3, 3)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To override the ``Conv2d`` with a new submodule ``Linear``, you
could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
where ``strict`` could be ``True`` or ``False``

To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

In the above if you set ``strict=True`` and call
``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
will be raised because ``net_b`` does not have a submodule named ``conv``.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)
    module: The module to set the submodule to.
    strict: If ``False``, the method will replace an existing submodule
        or create a new submodule if the parent module exists. If ``True``,
        the method will only attempt to replace an existing submodule and throw an error
        if the submodule doesn't already exist.

Raises:
    ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
    AttributeError: If at any point along the path resulting from
        the ``target`` string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `share_memory(self: ~T) -> ~T`

See :meth:`torch.Tensor.share_memory_`.

##### `state_dict(self, *args, destination=None, prefix='', keep_vars=False)`

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to ``None`` are not included.

.. note::
    The returned object is a shallow copy. It contains references
    to the module's parameters and buffers.

.. warning::
    Currently ``state_dict()`` also accepts positional arguments for
    ``destination``, ``prefix`` and ``keep_vars`` in order. However,
    this is being deprecated and keyword arguments will be enforced in
    future releases.

.. warning::
    Please avoid the use of argument ``destination`` as it is not
    designed for end-users.

Args:
    destination (dict, optional): If provided, the state of module will
        be updated into the dict and the same object is returned.
        Otherwise, an ``OrderedDict`` will be created and returned.
        Default: ``None``.
    prefix (str, optional): a prefix added to parameter and buffer
        names to compose the keys in state_dict. Default: ``''``.
    keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
        returned in the state dict are detached from autograd. If it's
        set to ``True``, detaching will not be performed.
        Default: ``False``.

Returns:
    dict:
        a dictionary containing a whole state of the module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> module.state_dict().keys()
    ['bias', 'weight']

##### `to(self, *args, **kwargs)`

Move and/or cast the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)
   :noindex:

.. function:: to(dtype, non_blocking=False)
   :noindex:

.. function:: to(tensor, non_blocking=False)
   :noindex:

.. function:: to(memory_format=torch.channels_last)
   :noindex:

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point or complex :attr:`dtype`\ s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

Args:
    device (:class:`torch.device`): the desired device of the parameters
        and buffers in this module
    dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
        the parameters and buffers in this module
    tensor (torch.Tensor): Tensor whose dtype and device are the desired
        dtype and device for all parameters and buffers in this module
    memory_format (:class:`torch.memory_format`): the desired memory
        format for 4D parameters and buffers in this module (keyword
        only argument)

Returns:
    Module: self

Examples::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)

    >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.3741+0.j,  0.2382+0.j],
            [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
    >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
    tensor([[0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

##### `to_empty(self: ~T, *, device: Union[int, str, torch.device, NoneType], recurse: bool = True) -> ~T`

Move the parameters and buffers to the specified device without copying storage.

Args:
    device (:class:`torch.device`): The desired device of the parameters
        and buffers in this module.
    recurse (bool): Whether parameters and buffers of submodules should
        be recursively moved to the specified device.

Returns:
    Module: self

##### `train(self: ~T, mode: bool = True) -> ~T`

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

Args:
    mode (bool): whether to set training mode (``True``) or evaluation
                 mode (``False``). Default: ``True``.

Returns:
    Module: self

##### `type(self: ~T, dst_type: Union[torch.dtype, str]) -> ~T`

Casts all parameters and buffers to :attr:`dst_type`.

.. note::
    This method modifies the module in-place.

Args:
    dst_type (type or string): the desired type

Returns:
    Module: self

##### `xpu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `zero_grad(self, set_to_none: bool = True) -> None`

Reset gradients of all model parameters.

See similar function under :class:`torch.optim.Optimizer` for more context.

Args:
    set_to_none (bool): instead of setting to zero, set the grads to None.
        See :meth:`torch.optim.Optimizer.zero_grad` for details.

---

### class `DataAnonymizer`

```python
DataAnonymizer(config: Optional[core.modules.privacy.PrivacyConfig] = None)
```

Comprehensive data anonymization for the Pixelis framework.

Combines PII redaction, metadata stripping, and other privacy techniques.

#### Methods

##### `__init__(self, config: Optional[core.modules.privacy.PrivacyConfig] = None)`

Initialize the data anonymizer.

Args:
    config: Privacy configuration

##### `anonymize_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]`

Anonymize an experience before storage.

Args:
    experience_data: Raw experience data
    
Returns:
    Anonymized experience data

##### `get_statistics(self) -> Dict[str, Any]`

Get anonymization statistics.

##### `verify_anonymization(self, data: Dict[str, Any]) -> Dict[str, Any]`

Verify that data has been properly anonymized.

Args:
    data: Data to verify
    
Returns:
    Verification report

---

### class `DynamicsModel`

```python
DynamicsModel(state_dim: int = 768, action_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1, use_residual: bool = True, device: str = 'cuda')
```

Lightweight dynamics model for state prediction.

Used by the curiosity module to predict next states and
generate intrinsic rewards based on prediction errors.

#### Methods

##### `__call__(self, *args, **kwargs)`

##### `__delattr__(self, name)`

Implement delattr(self, name).

##### `__dir__(self)`

Default dir() implementation.

##### `__getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]`

##### `__getstate__(self)`

Helper for pickle.

##### `__init__(self, state_dim: int = 768, action_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1, use_residual: bool = True, device: str = 'cuda')`

Initialize dynamics model.

Args:
    state_dim: Dimension of state representations
    action_dim: Dimension of action representations
    hidden_dim: Hidden layer dimension
    num_layers: Number of hidden layers
    dropout: Dropout probability
    use_residual: Whether to use residual connections
    device: Device for computation

##### `__repr__(self)`

Return repr(self).

##### `__setattr__(self, name: str, value: Union[torch.Tensor, ForwardRef('Module')]) -> None`

Implement setattr(self, name, value).

##### `__setstate__(self, state)`

##### `add_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Args:
    name (str): name of the child module. The child module can be
        accessed from this module using the given name
    module (Module): child module to be added to the module.

##### `apply(self: ~T, fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T`

Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

Args:
    fn (:class:`Module` -> None): function to be applied to each submodule

Returns:
    Module: self

Example::

    >>> @torch.no_grad()
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )

##### `bfloat16(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``bfloat16`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `buffers(self, recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]`

Return an iterator over module buffers.

Args:
    recurse (bool): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module.

Yields:
    torch.Tensor: module buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for buf in model.buffers():
    >>>     print(type(buf), buf.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `children(self) -> collections.abc.Iterator['Module']`

Return an iterator over immediate children modules.

Yields:
    Module: a child module

##### `compile(self, *args, **kwargs)`

Compile this Module's forward using :func:`torch.compile`.

This Module's `__call__` method is compiled and all arguments are passed as-is
to :func:`torch.compile`.

See :func:`torch.compile` for details on the arguments for this function.

##### `compute_prediction_error(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reduction: str = 'mean') -> torch.Tensor`

Compute prediction error for curiosity reward.

Args:
    state: Current state
    action: Action taken
    next_state: Actual next state
    reduction: How to reduce error ('mean', 'sum', 'none')
    
Returns:
    Prediction error

##### `count_parameters(self) -> int`

Count trainable parameters.

##### `cpu(self: ~T) -> ~T`

Move all model parameters and buffers to the CPU.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `cuda(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

.. note::
    This method modifies the module in-place.

Args:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `double(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``double`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `eval(self: ~T) -> ~T`

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

See :ref:`locally-disable-grad-doc` for a comparison between
`.eval()` and several similar mechanisms that may be confused with it.

Returns:
    Module: self

##### `extra_repr(self) -> str`

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

##### `float(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``float`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor`

Predict next state given current state and action.

Args:
    state: Current state tensor [batch_size, state_dim]
    action: Action tensor [batch_size, action_dim]
    
Returns:
    Predicted next state [batch_size, state_dim]

##### `get_buffer(self, target: str) -> 'Tensor'`

Return the buffer given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the buffer
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.Tensor: The buffer referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not a
        buffer

##### `get_extra_state(self) -> Any`

Return any extra state to include in the module's state_dict.

Implement this and a corresponding :func:`set_extra_state` for your module
if you need to store extra state. This function is called when building the
module's `state_dict()`.

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:
    object: Any extra state to store in the module's state_dict

##### `get_parameter(self, target: str) -> 'Parameter'`

Return the parameter given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the Parameter
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Parameter: The Parameter referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not an
        ``nn.Parameter``

##### `get_submodule(self, target: str) -> 'Module'`

Return the submodule given by ``target`` if it exists, otherwise throw an error.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
            )
            (linear): Linear(in_features=100, out_features=200, bias=True)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To check whether or not we have the ``linear`` submodule, we
would call ``get_submodule("net_b.linear")``. To check whether
we have the ``conv`` submodule, we would call
``get_submodule("net_b.net_c.conv")``.

The runtime of ``get_submodule`` is bounded by the degree
of module nesting in ``target``. A query against
``named_modules`` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, ``get_submodule`` should always be
used.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Module: The submodule referenced by ``target``

Raises:
    AttributeError: If at any point along the path resulting from
        the target string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `half(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``half`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `ipu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `load_state_dict(self, state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)`

Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

Args:
    state_dict (dict): a dict containing parameters and
        persistent buffers.
    strict (bool, optional): whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    assign (bool, optional): When set to ``False``, the properties of the tensors
        in the current module are preserved whereas setting it to ``True`` preserves
        properties of the Tensors in the state dict. The only
        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
        for which the value from the module is preserved.
        Default: ``False``

Returns:
    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
        * **missing_keys** is a list of str containing any keys that are expected
            by this module but missing from the provided ``state_dict``.
        * **unexpected_keys** is a list of str containing the keys that are not
            expected by this module but present in the provided ``state_dict``.

Note:
    If a parameter or buffer is registered as ``None`` and its corresponding key
    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    ``RuntimeError``.

##### `modules(self) -> collections.abc.Iterator['Module']`

Return an iterator over all modules in the network.

Yields:
    Module: a module in the network

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
    ...     print(idx, '->', m)

    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)

##### `mtia(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]`

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Args:
    prefix (str): prefix to prepend to all buffer names.
    recurse (bool, optional): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module. Defaults to True.
    remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

Yields:
    (str, torch.Tensor): Tuple containing the name and buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, buf in self.named_buffers():
    >>>     if name in ['running_var']:
    >>>         print(buf.size())

##### `named_children(self) -> collections.abc.Iterator[tuple[str, 'Module']]`

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:
    (str, Module): Tuple containing a name and child module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)

##### `named_modules(self, memo: Optional[set['Module']] = None, prefix: str = '', remove_duplicate: bool = True)`

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Args:
    memo: a memo to store the set of modules already added to the result
    prefix: a prefix that will be added to the name of the module
    remove_duplicate: whether to remove the duplicated module instances in the result
        or not

Yields:
    (str, Module): Tuple of name and module

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
    ...     print(idx, '->', m)

    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

##### `named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]`

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Args:
    prefix (str): prefix to prepend to all parameter names.
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.
    remove_duplicate (bool, optional): whether to remove the duplicated
        parameters in the result. Defaults to True.

Yields:
    (str, Parameter): Tuple containing the name and parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, param in self.named_parameters():
    >>>     if name in ['bias']:
    >>>         print(param.size())

##### `parameters(self, recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]`

Return an iterator over module parameters.

This is typically passed to an optimizer.

Args:
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.

Yields:
    Parameter: module parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for param in model.parameters():
    >>>     print(type(param), param.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `predict_trajectory(self, initial_state: torch.Tensor, action_sequence: List[torch.Tensor]) -> List[torch.Tensor]`

Predict a sequence of states given initial state and actions.

Args:
    initial_state: Initial state tensor
    action_sequence: List of action tensors
    
Returns:
    List of predicted states

##### `register_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
the behavior of this function will change in future versions.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None`

Add a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting :attr:`persistent` to ``False``. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
:attr:`state_dict`.

Buffers can be accessed as attributes using given names.

Args:
    name (str): name of the buffer. The buffer can be accessed
        from this module using the given name
    tensor (Tensor or None): buffer to be registered. If ``None``, then operations
        that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
        the buffer is **not** included in the module's :attr:`state_dict`.
    persistent (bool): whether the buffer is part of this module's
        :attr:`state_dict`.

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> self.register_buffer('running_mean', torch.zeros(num_features))

##### `register_forward_hook(self, hook: Union[Callable[[~T, tuple[Any, ...], Any], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]]], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.

If ``with_kwargs`` is ``False`` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after :func:`forward` is called. The hook
should have the following signature::

    hook(module, args, output) -> None or modified output

If ``with_kwargs`` is ``True``, the forward hook will be passed the
``kwargs`` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature::

    hook(module, args, kwargs, output) -> None or modified output

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If ``True``, the provided ``hook`` will be fired
        before all existing ``forward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``forward`` hooks registered with
        :func:`register_module_forward_hook` will fire before all hooks
        registered by this method.
        Default: ``False``
    with_kwargs (bool): If ``True``, the ``hook`` will be passed the
        kwargs given to the forward function.
        Default: ``False``
    always_call (bool): If ``True`` the ``hook`` will be run regardless of
        whether an exception is raised while calling the Module.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_forward_pre_hook(self, hook: Union[Callable[[~T, tuple[Any, ...]], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any]], Optional[tuple[Any, dict[str, Any]]]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.


If ``with_kwargs`` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature::

    hook(module, args) -> None or modified input

If ``with_kwargs`` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature::

    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``forward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``forward_pre`` hooks registered with
        :func:`register_module_forward_pre_hook` will fire before all
        hooks registered by this method.
        Default: ``False``
    with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
        given to the forward function.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module
are computed, i.e. the hook will execute if and only if the gradients with
respect to module outputs are computed. The hook should have the following
signature::

    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of :attr:`grad_input` in
subsequent computations. :attr:`grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs or outputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``backward`` hooks registered with
        :func:`register_module_full_backward_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_pre_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature::

    hook(module, grad_output) -> tuple[Tensor] or None

The :attr:`grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of :attr:`grad_output` in
subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``backward_pre`` hooks registered with
        :func:`register_module_full_backward_pre_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_post_hook(self, hook)`

Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, incompatible_keys) -> None

The ``module`` argument is the current module that this hook is registered
on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
is a ``list`` of ``str`` containing the missing keys and
``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling :func:`load_state_dict` with
``strict=True`` are affected by modifications the hook makes to
``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
set of keys will result in an error being thrown when ``strict=True``, and
clearing out both missing and unexpected keys will avoid an error.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_pre_hook(self, hook)`

Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

Arguments:
    hook (Callable): Callable hook that will be invoked before
        loading the state dict.

##### `register_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Alias for :func:`add_module`.

##### `register_parameter(self, name: str, param: Optional[torch.nn.parameter.Parameter]) -> None`

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Args:
    name (str): name of the parameter. The parameter can be accessed
        from this module using the given name
    param (Parameter or None): parameter to be added to the module. If
        ``None``, then operations that run on parameters, such as :attr:`cuda`,
        are ignored. If ``None``, the parameter is **not** included in the
        module's :attr:`state_dict`.

##### `register_state_dict_post_hook(self, hook)`

Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the ``state_dict`` inplace.

##### `register_state_dict_pre_hook(self, hook)`

Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the ``state_dict``
call is made.

##### `requires_grad_(self: ~T, requires_grad: bool = True) -> ~T`

Change if autograd should record operations on parameters in this module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See :ref:`locally-disable-grad-doc` for a comparison between
`.requires_grad_()` and several similar mechanisms that may be confused with it.

Args:
    requires_grad (bool): whether autograd should record operations on
                          parameters in this module. Default: ``True``.

Returns:
    Module: self

##### `set_extra_state(self, state: Any) -> None`

Set extra state contained in the loaded `state_dict`.

This function is called from :func:`load_state_dict` to handle any extra state
found within the `state_dict`. Implement this function and a corresponding
:func:`get_extra_state` for your module if you need to store extra state within its
`state_dict`.

Args:
    state (dict): Extra state from the `state_dict`

##### `set_submodule(self, target: str, module: 'Module', strict: bool = False) -> None`

Set the submodule given by ``target`` if it exists, otherwise throw an error.

.. note::
    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
    the method will only attempt to replace an existing submodule and throw an error if
    the submodule does not exist.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(3, 3, 3)
            )
            (linear): Linear(3, 3)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To override the ``Conv2d`` with a new submodule ``Linear``, you
could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
where ``strict`` could be ``True`` or ``False``

To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

In the above if you set ``strict=True`` and call
``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
will be raised because ``net_b`` does not have a submodule named ``conv``.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)
    module: The module to set the submodule to.
    strict: If ``False``, the method will replace an existing submodule
        or create a new submodule if the parent module exists. If ``True``,
        the method will only attempt to replace an existing submodule and throw an error
        if the submodule doesn't already exist.

Raises:
    ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
    AttributeError: If at any point along the path resulting from
        the ``target`` string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `share_memory(self: ~T) -> ~T`

See :meth:`torch.Tensor.share_memory_`.

##### `state_dict(self, *args, destination=None, prefix='', keep_vars=False)`

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to ``None`` are not included.

.. note::
    The returned object is a shallow copy. It contains references
    to the module's parameters and buffers.

.. warning::
    Currently ``state_dict()`` also accepts positional arguments for
    ``destination``, ``prefix`` and ``keep_vars`` in order. However,
    this is being deprecated and keyword arguments will be enforced in
    future releases.

.. warning::
    Please avoid the use of argument ``destination`` as it is not
    designed for end-users.

Args:
    destination (dict, optional): If provided, the state of module will
        be updated into the dict and the same object is returned.
        Otherwise, an ``OrderedDict`` will be created and returned.
        Default: ``None``.
    prefix (str, optional): a prefix added to parameter and buffer
        names to compose the keys in state_dict. Default: ``''``.
    keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
        returned in the state dict are detached from autograd. If it's
        set to ``True``, detaching will not be performed.
        Default: ``False``.

Returns:
    dict:
        a dictionary containing a whole state of the module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> module.state_dict().keys()
    ['bias', 'weight']

##### `to(self, *args, **kwargs)`

Move and/or cast the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)
   :noindex:

.. function:: to(dtype, non_blocking=False)
   :noindex:

.. function:: to(tensor, non_blocking=False)
   :noindex:

.. function:: to(memory_format=torch.channels_last)
   :noindex:

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point or complex :attr:`dtype`\ s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

Args:
    device (:class:`torch.device`): the desired device of the parameters
        and buffers in this module
    dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
        the parameters and buffers in this module
    tensor (torch.Tensor): Tensor whose dtype and device are the desired
        dtype and device for all parameters and buffers in this module
    memory_format (:class:`torch.memory_format`): the desired memory
        format for 4D parameters and buffers in this module (keyword
        only argument)

Returns:
    Module: self

Examples::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)

    >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.3741+0.j,  0.2382+0.j],
            [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
    >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
    tensor([[0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

##### `to_empty(self: ~T, *, device: Union[int, str, torch.device, NoneType], recurse: bool = True) -> ~T`

Move the parameters and buffers to the specified device without copying storage.

Args:
    device (:class:`torch.device`): The desired device of the parameters
        and buffers in this module.
    recurse (bool): Whether parameters and buffers of submodules should
        be recursively moved to the specified device.

Returns:
    Module: self

##### `train(self: ~T, mode: bool = True) -> ~T`

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

Args:
    mode (bool): whether to set training mode (``True``) or evaluation
                 mode (``False``). Default: ``True``.

Returns:
    Module: self

##### `type(self: ~T, dst_type: Union[torch.dtype, str]) -> ~T`

Casts all parameters and buffers to :attr:`dst_type`.

.. note::
    This method modifies the module in-place.

Args:
    dst_type (type or string): the desired type

Returns:
    Module: self

##### `xpu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `zero_grad(self, set_to_none: bool = True) -> None`

Reset gradients of all model parameters.

See similar function under :class:`torch.optim.Optimizer` for more context.

Args:
    set_to_none (bool): instead of setting to zero, set the grads to None.
        See :meth:`torch.optim.Optimizer.zero_grad` for details.

---

### class `EnhancedTrajectoryCoherenceAnalyzer`

```python
EnhancedTrajectoryCoherenceAnalyzer(coherence_threshold: float = 0.7, repetition_penalty: float = 0.5, sequence_bonus: float = 0.2, contradiction_penalty: float = 0.3)
```

Enhanced analyzer for trajectory coherence with detailed pattern recognition.

Rewards logical action sequences and penalizes repetitive or illogical patterns.

#### Methods

##### `__init__(self, coherence_threshold: float = 0.7, repetition_penalty: float = 0.5, sequence_bonus: float = 0.2, contradiction_penalty: float = 0.3)`

Initialize self.  See help(type(self)) for accurate signature.

##### `compute_coherence_reward(self, trajectory: List[Dict[str, Any]], embeddings: Optional[List[torch.Tensor]] = None) -> Tuple[float, Dict[str, Any]]`

Compute coherence reward for trajectory.

Args:
    trajectory: List of actions in trajectory
    embeddings: Optional state embeddings for semantic analysis
    
Returns:
    Tuple of (coherence_reward, metrics)

---

### class `FilePersistenceAdapter`

```python
FilePersistenceAdapter(base_path: str)
```

File-based persistence adapter using Write-Ahead Log pattern.
Uses atomic writes (tmp -> fsync -> rename) for crash safety.

#### Methods

##### `__init__(self, base_path: str)`

Initialize file-based persistence.

Args:
    base_path: Base directory for persistence files

##### `close(self)`

Close any open resources.

##### `load_snapshot(self) -> Optional[Dict[str, Any]]`

Load the latest snapshot.

##### `read_all_experiences(self) -> List[Dict[str, Any]]`

Read all experiences from the WAL.

##### `read_all_operations(self) -> List[Dict[str, Any]]`

Read all operations from the operations log.

##### `save_snapshot(self, snapshot_data: Dict[str, Any]) -> bool`

Save a complete snapshot.

##### `truncate_logs(self) -> bool`

Truncate both WAL logs after successful snapshot.

##### `write_experience(self, experience_id: str, data: Dict[str, Any]) -> bool`

Write an experience to the WAL.

##### `write_operation(self, operation: Dict[str, Any]) -> bool`

Write an operation to the operations log.

---

### class `HealthMonitor`

```python
HealthMonitor(alerter: core.modules.alerter.Alerter)
```

System health monitoring with integrated alerting.

Tracks key health indicators and triggers alerts when thresholds are exceeded.

#### Methods

##### `__init__(self, alerter: core.modules.alerter.Alerter)`

Initialize the health monitor.

Args:
    alerter: Alerter instance for sending alerts

##### `get_health_status(self) -> Dict[str, Any]`

Get current health status.

##### `record_faiss_attempt(self, success: bool)`

Record a FAISS operation attempt.

##### `record_update(self, timestamp: Optional[datetime.datetime] = None)`

Record a model update for rate calculation.

##### `update_metrics(self, new_metrics: Dict[str, Any], component: str = 'system')`

Update health metrics and check for alerts.

Args:
    new_metrics: New metric values to update
    component: Component source of the metrics

---

### class `ImageMetadataStripper`

```python
ImageMetadataStripper(config: Optional[core.modules.privacy.PrivacyConfig] = None)
```

Strips sensitive metadata from images.

Removes:
- EXIF data (including GPS coordinates)
- Camera/device information
- Timestamps
- Author/copyright information
- Software information

#### Methods

##### `__init__(self, config: Optional[core.modules.privacy.PrivacyConfig] = None)`

Initialize the metadata stripper.

Args:
    config: Privacy configuration

##### `analyze_metadata(self, image_data: bytes) -> Dict[str, Any]`

Analyze metadata in image without removing it.

Args:
    image_data: Raw image bytes
    
Returns:
    Metadata analysis report

##### `get_statistics(self) -> Dict[str, Any]`

Get processing statistics.

##### `strip_metadata(self, image_data: bytes, image_format: str = 'jpeg') -> bytes`

Strip metadata from image data.

Args:
    image_data: Raw image bytes
    image_format: Image format
    
Returns:
    Image data with metadata stripped

---

### class `InverseModel`

```python
InverseModel(state_dim: int = 768, action_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1, device: str = 'cuda')
```

Inverse dynamics model that predicts actions from state transitions.

Used to learn meaningful action representations and ensure
the state encoder captures action-relevant information.

#### Methods

##### `__call__(self, *args, **kwargs)`

##### `__delattr__(self, name)`

Implement delattr(self, name).

##### `__dir__(self)`

Default dir() implementation.

##### `__getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]`

##### `__getstate__(self)`

Helper for pickle.

##### `__init__(self, state_dim: int = 768, action_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1, device: str = 'cuda')`

Initialize inverse model.

Args:
    state_dim: Dimension of state representations
    action_dim: Dimension of action representations
    hidden_dim: Hidden layer dimension
    num_layers: Number of hidden layers
    dropout: Dropout probability
    device: Device for computation

##### `__repr__(self)`

Return repr(self).

##### `__setattr__(self, name: str, value: Union[torch.Tensor, ForwardRef('Module')]) -> None`

Implement setattr(self, name, value).

##### `__setstate__(self, state)`

##### `add_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Args:
    name (str): name of the child module. The child module can be
        accessed from this module using the given name
    module (Module): child module to be added to the module.

##### `apply(self: ~T, fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T`

Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

Args:
    fn (:class:`Module` -> None): function to be applied to each submodule

Returns:
    Module: self

Example::

    >>> @torch.no_grad()
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )

##### `bfloat16(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``bfloat16`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `buffers(self, recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]`

Return an iterator over module buffers.

Args:
    recurse (bool): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module.

Yields:
    torch.Tensor: module buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for buf in model.buffers():
    >>>     print(type(buf), buf.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `children(self) -> collections.abc.Iterator['Module']`

Return an iterator over immediate children modules.

Yields:
    Module: a child module

##### `compile(self, *args, **kwargs)`

Compile this Module's forward using :func:`torch.compile`.

This Module's `__call__` method is compiled and all arguments are passed as-is
to :func:`torch.compile`.

See :func:`torch.compile` for details on the arguments for this function.

##### `cpu(self: ~T) -> ~T`

Move all model parameters and buffers to the CPU.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `cuda(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

.. note::
    This method modifies the module in-place.

Args:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `double(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``double`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `eval(self: ~T) -> ~T`

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

See :ref:`locally-disable-grad-doc` for a comparison between
`.eval()` and several similar mechanisms that may be confused with it.

Returns:
    Module: self

##### `extra_repr(self) -> str`

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

##### `float(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``float`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor`

Predict action that caused state transition.

Args:
    state: Current state [batch_size, state_dim]
    next_state: Next state [batch_size, state_dim]
    
Returns:
    Predicted action [batch_size, action_dim]

##### `get_buffer(self, target: str) -> 'Tensor'`

Return the buffer given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the buffer
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.Tensor: The buffer referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not a
        buffer

##### `get_extra_state(self) -> Any`

Return any extra state to include in the module's state_dict.

Implement this and a corresponding :func:`set_extra_state` for your module
if you need to store extra state. This function is called when building the
module's `state_dict()`.

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:
    object: Any extra state to store in the module's state_dict

##### `get_parameter(self, target: str) -> 'Parameter'`

Return the parameter given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the Parameter
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Parameter: The Parameter referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not an
        ``nn.Parameter``

##### `get_submodule(self, target: str) -> 'Module'`

Return the submodule given by ``target`` if it exists, otherwise throw an error.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
            )
            (linear): Linear(in_features=100, out_features=200, bias=True)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To check whether or not we have the ``linear`` submodule, we
would call ``get_submodule("net_b.linear")``. To check whether
we have the ``conv`` submodule, we would call
``get_submodule("net_b.net_c.conv")``.

The runtime of ``get_submodule`` is bounded by the degree
of module nesting in ``target``. A query against
``named_modules`` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, ``get_submodule`` should always be
used.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Module: The submodule referenced by ``target``

Raises:
    AttributeError: If at any point along the path resulting from
        the target string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `half(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``half`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `ipu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `load_state_dict(self, state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)`

Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

Args:
    state_dict (dict): a dict containing parameters and
        persistent buffers.
    strict (bool, optional): whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    assign (bool, optional): When set to ``False``, the properties of the tensors
        in the current module are preserved whereas setting it to ``True`` preserves
        properties of the Tensors in the state dict. The only
        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
        for which the value from the module is preserved.
        Default: ``False``

Returns:
    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
        * **missing_keys** is a list of str containing any keys that are expected
            by this module but missing from the provided ``state_dict``.
        * **unexpected_keys** is a list of str containing the keys that are not
            expected by this module but present in the provided ``state_dict``.

Note:
    If a parameter or buffer is registered as ``None`` and its corresponding key
    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    ``RuntimeError``.

##### `modules(self) -> collections.abc.Iterator['Module']`

Return an iterator over all modules in the network.

Yields:
    Module: a module in the network

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
    ...     print(idx, '->', m)

    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)

##### `mtia(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]`

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Args:
    prefix (str): prefix to prepend to all buffer names.
    recurse (bool, optional): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module. Defaults to True.
    remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

Yields:
    (str, torch.Tensor): Tuple containing the name and buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, buf in self.named_buffers():
    >>>     if name in ['running_var']:
    >>>         print(buf.size())

##### `named_children(self) -> collections.abc.Iterator[tuple[str, 'Module']]`

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:
    (str, Module): Tuple containing a name and child module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)

##### `named_modules(self, memo: Optional[set['Module']] = None, prefix: str = '', remove_duplicate: bool = True)`

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Args:
    memo: a memo to store the set of modules already added to the result
    prefix: a prefix that will be added to the name of the module
    remove_duplicate: whether to remove the duplicated module instances in the result
        or not

Yields:
    (str, Module): Tuple of name and module

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
    ...     print(idx, '->', m)

    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

##### `named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]`

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Args:
    prefix (str): prefix to prepend to all parameter names.
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.
    remove_duplicate (bool, optional): whether to remove the duplicated
        parameters in the result. Defaults to True.

Yields:
    (str, Parameter): Tuple containing the name and parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, param in self.named_parameters():
    >>>     if name in ['bias']:
    >>>         print(param.size())

##### `parameters(self, recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]`

Return an iterator over module parameters.

This is typically passed to an optimizer.

Args:
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.

Yields:
    Parameter: module parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for param in model.parameters():
    >>>     print(type(param), param.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `register_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
the behavior of this function will change in future versions.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None`

Add a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting :attr:`persistent` to ``False``. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
:attr:`state_dict`.

Buffers can be accessed as attributes using given names.

Args:
    name (str): name of the buffer. The buffer can be accessed
        from this module using the given name
    tensor (Tensor or None): buffer to be registered. If ``None``, then operations
        that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
        the buffer is **not** included in the module's :attr:`state_dict`.
    persistent (bool): whether the buffer is part of this module's
        :attr:`state_dict`.

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> self.register_buffer('running_mean', torch.zeros(num_features))

##### `register_forward_hook(self, hook: Union[Callable[[~T, tuple[Any, ...], Any], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]]], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.

If ``with_kwargs`` is ``False`` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after :func:`forward` is called. The hook
should have the following signature::

    hook(module, args, output) -> None or modified output

If ``with_kwargs`` is ``True``, the forward hook will be passed the
``kwargs`` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature::

    hook(module, args, kwargs, output) -> None or modified output

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If ``True``, the provided ``hook`` will be fired
        before all existing ``forward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``forward`` hooks registered with
        :func:`register_module_forward_hook` will fire before all hooks
        registered by this method.
        Default: ``False``
    with_kwargs (bool): If ``True``, the ``hook`` will be passed the
        kwargs given to the forward function.
        Default: ``False``
    always_call (bool): If ``True`` the ``hook`` will be run regardless of
        whether an exception is raised while calling the Module.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_forward_pre_hook(self, hook: Union[Callable[[~T, tuple[Any, ...]], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any]], Optional[tuple[Any, dict[str, Any]]]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.


If ``with_kwargs`` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature::

    hook(module, args) -> None or modified input

If ``with_kwargs`` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature::

    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``forward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``forward_pre`` hooks registered with
        :func:`register_module_forward_pre_hook` will fire before all
        hooks registered by this method.
        Default: ``False``
    with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
        given to the forward function.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module
are computed, i.e. the hook will execute if and only if the gradients with
respect to module outputs are computed. The hook should have the following
signature::

    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of :attr:`grad_input` in
subsequent computations. :attr:`grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs or outputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``backward`` hooks registered with
        :func:`register_module_full_backward_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_pre_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature::

    hook(module, grad_output) -> tuple[Tensor] or None

The :attr:`grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of :attr:`grad_output` in
subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``backward_pre`` hooks registered with
        :func:`register_module_full_backward_pre_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_post_hook(self, hook)`

Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, incompatible_keys) -> None

The ``module`` argument is the current module that this hook is registered
on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
is a ``list`` of ``str`` containing the missing keys and
``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling :func:`load_state_dict` with
``strict=True`` are affected by modifications the hook makes to
``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
set of keys will result in an error being thrown when ``strict=True``, and
clearing out both missing and unexpected keys will avoid an error.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_pre_hook(self, hook)`

Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

Arguments:
    hook (Callable): Callable hook that will be invoked before
        loading the state dict.

##### `register_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Alias for :func:`add_module`.

##### `register_parameter(self, name: str, param: Optional[torch.nn.parameter.Parameter]) -> None`

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Args:
    name (str): name of the parameter. The parameter can be accessed
        from this module using the given name
    param (Parameter or None): parameter to be added to the module. If
        ``None``, then operations that run on parameters, such as :attr:`cuda`,
        are ignored. If ``None``, the parameter is **not** included in the
        module's :attr:`state_dict`.

##### `register_state_dict_post_hook(self, hook)`

Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the ``state_dict`` inplace.

##### `register_state_dict_pre_hook(self, hook)`

Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the ``state_dict``
call is made.

##### `requires_grad_(self: ~T, requires_grad: bool = True) -> ~T`

Change if autograd should record operations on parameters in this module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See :ref:`locally-disable-grad-doc` for a comparison between
`.requires_grad_()` and several similar mechanisms that may be confused with it.

Args:
    requires_grad (bool): whether autograd should record operations on
                          parameters in this module. Default: ``True``.

Returns:
    Module: self

##### `set_extra_state(self, state: Any) -> None`

Set extra state contained in the loaded `state_dict`.

This function is called from :func:`load_state_dict` to handle any extra state
found within the `state_dict`. Implement this function and a corresponding
:func:`get_extra_state` for your module if you need to store extra state within its
`state_dict`.

Args:
    state (dict): Extra state from the `state_dict`

##### `set_submodule(self, target: str, module: 'Module', strict: bool = False) -> None`

Set the submodule given by ``target`` if it exists, otherwise throw an error.

.. note::
    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
    the method will only attempt to replace an existing submodule and throw an error if
    the submodule does not exist.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(3, 3, 3)
            )
            (linear): Linear(3, 3)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To override the ``Conv2d`` with a new submodule ``Linear``, you
could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
where ``strict`` could be ``True`` or ``False``

To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

In the above if you set ``strict=True`` and call
``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
will be raised because ``net_b`` does not have a submodule named ``conv``.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)
    module: The module to set the submodule to.
    strict: If ``False``, the method will replace an existing submodule
        or create a new submodule if the parent module exists. If ``True``,
        the method will only attempt to replace an existing submodule and throw an error
        if the submodule doesn't already exist.

Raises:
    ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
    AttributeError: If at any point along the path resulting from
        the ``target`` string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `share_memory(self: ~T) -> ~T`

See :meth:`torch.Tensor.share_memory_`.

##### `state_dict(self, *args, destination=None, prefix='', keep_vars=False)`

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to ``None`` are not included.

.. note::
    The returned object is a shallow copy. It contains references
    to the module's parameters and buffers.

.. warning::
    Currently ``state_dict()`` also accepts positional arguments for
    ``destination``, ``prefix`` and ``keep_vars`` in order. However,
    this is being deprecated and keyword arguments will be enforced in
    future releases.

.. warning::
    Please avoid the use of argument ``destination`` as it is not
    designed for end-users.

Args:
    destination (dict, optional): If provided, the state of module will
        be updated into the dict and the same object is returned.
        Otherwise, an ``OrderedDict`` will be created and returned.
        Default: ``None``.
    prefix (str, optional): a prefix added to parameter and buffer
        names to compose the keys in state_dict. Default: ``''``.
    keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
        returned in the state dict are detached from autograd. If it's
        set to ``True``, detaching will not be performed.
        Default: ``False``.

Returns:
    dict:
        a dictionary containing a whole state of the module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> module.state_dict().keys()
    ['bias', 'weight']

##### `to(self, *args, **kwargs)`

Move and/or cast the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)
   :noindex:

.. function:: to(dtype, non_blocking=False)
   :noindex:

.. function:: to(tensor, non_blocking=False)
   :noindex:

.. function:: to(memory_format=torch.channels_last)
   :noindex:

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point or complex :attr:`dtype`\ s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

Args:
    device (:class:`torch.device`): the desired device of the parameters
        and buffers in this module
    dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
        the parameters and buffers in this module
    tensor (torch.Tensor): Tensor whose dtype and device are the desired
        dtype and device for all parameters and buffers in this module
    memory_format (:class:`torch.memory_format`): the desired memory
        format for 4D parameters and buffers in this module (keyword
        only argument)

Returns:
    Module: self

Examples::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)

    >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.3741+0.j,  0.2382+0.j],
            [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
    >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
    tensor([[0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

##### `to_empty(self: ~T, *, device: Union[int, str, torch.device, NoneType], recurse: bool = True) -> ~T`

Move the parameters and buffers to the specified device without copying storage.

Args:
    device (:class:`torch.device`): The desired device of the parameters
        and buffers in this module.
    recurse (bool): Whether parameters and buffers of submodules should
        be recursively moved to the specified device.

Returns:
    Module: self

##### `train(self: ~T, mode: bool = True) -> ~T`

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

Args:
    mode (bool): whether to set training mode (``True``) or evaluation
                 mode (``False``). Default: ``True``.

Returns:
    Module: self

##### `type(self: ~T, dst_type: Union[torch.dtype, str]) -> ~T`

Casts all parameters and buffers to :attr:`dst_type`.

.. note::
    This method modifies the module in-place.

Args:
    dst_type (type or string): the desired type

Returns:
    Module: self

##### `xpu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `zero_grad(self, set_to_none: bool = True) -> None`

Reset gradients of all model parameters.

See similar function under :class:`torch.optim.Optimizer` for more context.

Args:
    set_to_none (bool): instead of setting to zero, set the grads to None.
        See :meth:`torch.optim.Optimizer.zero_grad` for details.

---

### class `LMDBPersistenceAdapter`

```python
LMDBPersistenceAdapter(base_path: str)
```

LMDB-based persistence adapter for high-throughput scenarios.
Provides transactional guarantees and better performance than file-based.

#### Methods

##### `__init__(self, base_path: str)`

Initialize LMDB-based persistence.

Args:
    base_path: Base directory for LMDB database

##### `close(self)`

Close LMDB environment.

##### `load_snapshot(self) -> Optional[Dict[str, Any]]`

Load the latest snapshot from LMDB.

##### `read_all_experiences(self) -> List[Dict[str, Any]]`

Read all experiences from LMDB.

##### `read_all_operations(self) -> List[Dict[str, Any]]`

Read all operations from LMDB in order.

##### `save_snapshot(self, snapshot_data: Dict[str, Any]) -> bool`

Save a snapshot to LMDB.

##### `truncate_logs(self) -> bool`

Clear the operations log after snapshot.

##### `write_experience(self, experience_id: str, data: Dict[str, Any]) -> bool`

Write an experience to LMDB.

##### `write_operation(self, operation: Dict[str, Any]) -> bool`

Write an operation to LMDB.

---

### class `LoRADynamicsModel`

```python
LoRADynamicsModel(state_dim: int = 768, action_dim: int = 128, hidden_dim: int = 256, lora_rank: int = 8, lora_alpha: int = 16, dropout: float = 0.1, device: str = 'cuda')
```

Lightweight dynamics model with LoRA adapters for efficient curiosity computation.

This model uses low-rank adaptation to reduce parameters while maintaining
expressiveness for next-state prediction.

#### Methods

##### `__call__(self, *args, **kwargs)`

##### `__delattr__(self, name)`

Implement delattr(self, name).

##### `__dir__(self)`

Default dir() implementation.

##### `__getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]`

##### `__getstate__(self)`

Helper for pickle.

##### `__init__(self, state_dim: int = 768, action_dim: int = 128, hidden_dim: int = 256, lora_rank: int = 8, lora_alpha: int = 16, dropout: float = 0.1, device: str = 'cuda')`

Initialize internal Module state, shared by both nn.Module and ScriptModule.

##### `__repr__(self)`

Return repr(self).

##### `__setattr__(self, name: str, value: Union[torch.Tensor, ForwardRef('Module')]) -> None`

Implement setattr(self, name, value).

##### `__setstate__(self, state)`

##### `add_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Args:
    name (str): name of the child module. The child module can be
        accessed from this module using the given name
    module (Module): child module to be added to the module.

##### `apply(self: ~T, fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T`

Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

Args:
    fn (:class:`Module` -> None): function to be applied to each submodule

Returns:
    Module: self

Example::

    >>> @torch.no_grad()
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )

##### `bfloat16(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``bfloat16`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `buffers(self, recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]`

Return an iterator over module buffers.

Args:
    recurse (bool): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module.

Yields:
    torch.Tensor: module buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for buf in model.buffers():
    >>>     print(type(buf), buf.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `children(self) -> collections.abc.Iterator['Module']`

Return an iterator over immediate children modules.

Yields:
    Module: a child module

##### `compile(self, *args, **kwargs)`

Compile this Module's forward using :func:`torch.compile`.

This Module's `__call__` method is compiled and all arguments are passed as-is
to :func:`torch.compile`.

See :func:`torch.compile` for details on the arguments for this function.

##### `cpu(self: ~T) -> ~T`

Move all model parameters and buffers to the CPU.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `cuda(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

.. note::
    This method modifies the module in-place.

Args:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `double(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``double`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `eval(self: ~T) -> ~T`

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

See :ref:`locally-disable-grad-doc` for a comparison between
`.eval()` and several similar mechanisms that may be confused with it.

Returns:
    Module: self

##### `extra_repr(self) -> str`

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

##### `float(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``float`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor`

Predict next state given current state and action.

Args:
    state: Current state embedding [batch_size, state_dim]
    action: Action embedding [batch_size, action_dim]
    
Returns:
    Predicted next state [batch_size, state_dim]

##### `get_buffer(self, target: str) -> 'Tensor'`

Return the buffer given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the buffer
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.Tensor: The buffer referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not a
        buffer

##### `get_extra_state(self) -> Any`

Return any extra state to include in the module's state_dict.

Implement this and a corresponding :func:`set_extra_state` for your module
if you need to store extra state. This function is called when building the
module's `state_dict()`.

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:
    object: Any extra state to store in the module's state_dict

##### `get_num_trainable_params(self) -> int`

Get number of trainable parameters (LoRA only).

##### `get_parameter(self, target: str) -> 'Parameter'`

Return the parameter given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the Parameter
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Parameter: The Parameter referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not an
        ``nn.Parameter``

##### `get_submodule(self, target: str) -> 'Module'`

Return the submodule given by ``target`` if it exists, otherwise throw an error.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
            )
            (linear): Linear(in_features=100, out_features=200, bias=True)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To check whether or not we have the ``linear`` submodule, we
would call ``get_submodule("net_b.linear")``. To check whether
we have the ``conv`` submodule, we would call
``get_submodule("net_b.net_c.conv")``.

The runtime of ``get_submodule`` is bounded by the degree
of module nesting in ``target``. A query against
``named_modules`` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, ``get_submodule`` should always be
used.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Module: The submodule referenced by ``target``

Raises:
    AttributeError: If at any point along the path resulting from
        the target string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `half(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``half`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `ipu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `load_state_dict(self, state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)`

Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

Args:
    state_dict (dict): a dict containing parameters and
        persistent buffers.
    strict (bool, optional): whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    assign (bool, optional): When set to ``False``, the properties of the tensors
        in the current module are preserved whereas setting it to ``True`` preserves
        properties of the Tensors in the state dict. The only
        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
        for which the value from the module is preserved.
        Default: ``False``

Returns:
    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
        * **missing_keys** is a list of str containing any keys that are expected
            by this module but missing from the provided ``state_dict``.
        * **unexpected_keys** is a list of str containing the keys that are not
            expected by this module but present in the provided ``state_dict``.

Note:
    If a parameter or buffer is registered as ``None`` and its corresponding key
    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    ``RuntimeError``.

##### `modules(self) -> collections.abc.Iterator['Module']`

Return an iterator over all modules in the network.

Yields:
    Module: a module in the network

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
    ...     print(idx, '->', m)

    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)

##### `mtia(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]`

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Args:
    prefix (str): prefix to prepend to all buffer names.
    recurse (bool, optional): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module. Defaults to True.
    remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

Yields:
    (str, torch.Tensor): Tuple containing the name and buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, buf in self.named_buffers():
    >>>     if name in ['running_var']:
    >>>         print(buf.size())

##### `named_children(self) -> collections.abc.Iterator[tuple[str, 'Module']]`

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:
    (str, Module): Tuple containing a name and child module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)

##### `named_modules(self, memo: Optional[set['Module']] = None, prefix: str = '', remove_duplicate: bool = True)`

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Args:
    memo: a memo to store the set of modules already added to the result
    prefix: a prefix that will be added to the name of the module
    remove_duplicate: whether to remove the duplicated module instances in the result
        or not

Yields:
    (str, Module): Tuple of name and module

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
    ...     print(idx, '->', m)

    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

##### `named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]`

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Args:
    prefix (str): prefix to prepend to all parameter names.
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.
    remove_duplicate (bool, optional): whether to remove the duplicated
        parameters in the result. Defaults to True.

Yields:
    (str, Parameter): Tuple containing the name and parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, param in self.named_parameters():
    >>>     if name in ['bias']:
    >>>         print(param.size())

##### `parameters(self, recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]`

Return an iterator over module parameters.

This is typically passed to an optimizer.

Args:
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.

Yields:
    Parameter: module parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for param in model.parameters():
    >>>     print(type(param), param.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `register_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
the behavior of this function will change in future versions.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None`

Add a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting :attr:`persistent` to ``False``. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
:attr:`state_dict`.

Buffers can be accessed as attributes using given names.

Args:
    name (str): name of the buffer. The buffer can be accessed
        from this module using the given name
    tensor (Tensor or None): buffer to be registered. If ``None``, then operations
        that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
        the buffer is **not** included in the module's :attr:`state_dict`.
    persistent (bool): whether the buffer is part of this module's
        :attr:`state_dict`.

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> self.register_buffer('running_mean', torch.zeros(num_features))

##### `register_forward_hook(self, hook: Union[Callable[[~T, tuple[Any, ...], Any], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]]], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.

If ``with_kwargs`` is ``False`` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after :func:`forward` is called. The hook
should have the following signature::

    hook(module, args, output) -> None or modified output

If ``with_kwargs`` is ``True``, the forward hook will be passed the
``kwargs`` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature::

    hook(module, args, kwargs, output) -> None or modified output

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If ``True``, the provided ``hook`` will be fired
        before all existing ``forward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``forward`` hooks registered with
        :func:`register_module_forward_hook` will fire before all hooks
        registered by this method.
        Default: ``False``
    with_kwargs (bool): If ``True``, the ``hook`` will be passed the
        kwargs given to the forward function.
        Default: ``False``
    always_call (bool): If ``True`` the ``hook`` will be run regardless of
        whether an exception is raised while calling the Module.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_forward_pre_hook(self, hook: Union[Callable[[~T, tuple[Any, ...]], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any]], Optional[tuple[Any, dict[str, Any]]]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.


If ``with_kwargs`` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature::

    hook(module, args) -> None or modified input

If ``with_kwargs`` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature::

    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``forward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``forward_pre`` hooks registered with
        :func:`register_module_forward_pre_hook` will fire before all
        hooks registered by this method.
        Default: ``False``
    with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
        given to the forward function.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module
are computed, i.e. the hook will execute if and only if the gradients with
respect to module outputs are computed. The hook should have the following
signature::

    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of :attr:`grad_input` in
subsequent computations. :attr:`grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs or outputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``backward`` hooks registered with
        :func:`register_module_full_backward_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_pre_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature::

    hook(module, grad_output) -> tuple[Tensor] or None

The :attr:`grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of :attr:`grad_output` in
subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``backward_pre`` hooks registered with
        :func:`register_module_full_backward_pre_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_post_hook(self, hook)`

Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, incompatible_keys) -> None

The ``module`` argument is the current module that this hook is registered
on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
is a ``list`` of ``str`` containing the missing keys and
``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling :func:`load_state_dict` with
``strict=True`` are affected by modifications the hook makes to
``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
set of keys will result in an error being thrown when ``strict=True``, and
clearing out both missing and unexpected keys will avoid an error.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_pre_hook(self, hook)`

Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

Arguments:
    hook (Callable): Callable hook that will be invoked before
        loading the state dict.

##### `register_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Alias for :func:`add_module`.

##### `register_parameter(self, name: str, param: Optional[torch.nn.parameter.Parameter]) -> None`

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Args:
    name (str): name of the parameter. The parameter can be accessed
        from this module using the given name
    param (Parameter or None): parameter to be added to the module. If
        ``None``, then operations that run on parameters, such as :attr:`cuda`,
        are ignored. If ``None``, the parameter is **not** included in the
        module's :attr:`state_dict`.

##### `register_state_dict_post_hook(self, hook)`

Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the ``state_dict`` inplace.

##### `register_state_dict_pre_hook(self, hook)`

Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the ``state_dict``
call is made.

##### `requires_grad_(self: ~T, requires_grad: bool = True) -> ~T`

Change if autograd should record operations on parameters in this module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See :ref:`locally-disable-grad-doc` for a comparison between
`.requires_grad_()` and several similar mechanisms that may be confused with it.

Args:
    requires_grad (bool): whether autograd should record operations on
                          parameters in this module. Default: ``True``.

Returns:
    Module: self

##### `set_extra_state(self, state: Any) -> None`

Set extra state contained in the loaded `state_dict`.

This function is called from :func:`load_state_dict` to handle any extra state
found within the `state_dict`. Implement this function and a corresponding
:func:`get_extra_state` for your module if you need to store extra state within its
`state_dict`.

Args:
    state (dict): Extra state from the `state_dict`

##### `set_submodule(self, target: str, module: 'Module', strict: bool = False) -> None`

Set the submodule given by ``target`` if it exists, otherwise throw an error.

.. note::
    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
    the method will only attempt to replace an existing submodule and throw an error if
    the submodule does not exist.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(3, 3, 3)
            )
            (linear): Linear(3, 3)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To override the ``Conv2d`` with a new submodule ``Linear``, you
could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
where ``strict`` could be ``True`` or ``False``

To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

In the above if you set ``strict=True`` and call
``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
will be raised because ``net_b`` does not have a submodule named ``conv``.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)
    module: The module to set the submodule to.
    strict: If ``False``, the method will replace an existing submodule
        or create a new submodule if the parent module exists. If ``True``,
        the method will only attempt to replace an existing submodule and throw an error
        if the submodule doesn't already exist.

Raises:
    ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
    AttributeError: If at any point along the path resulting from
        the ``target`` string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `share_memory(self: ~T) -> ~T`

See :meth:`torch.Tensor.share_memory_`.

##### `state_dict(self, *args, destination=None, prefix='', keep_vars=False)`

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to ``None`` are not included.

.. note::
    The returned object is a shallow copy. It contains references
    to the module's parameters and buffers.

.. warning::
    Currently ``state_dict()`` also accepts positional arguments for
    ``destination``, ``prefix`` and ``keep_vars`` in order. However,
    this is being deprecated and keyword arguments will be enforced in
    future releases.

.. warning::
    Please avoid the use of argument ``destination`` as it is not
    designed for end-users.

Args:
    destination (dict, optional): If provided, the state of module will
        be updated into the dict and the same object is returned.
        Otherwise, an ``OrderedDict`` will be created and returned.
        Default: ``None``.
    prefix (str, optional): a prefix added to parameter and buffer
        names to compose the keys in state_dict. Default: ``''``.
    keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
        returned in the state dict are detached from autograd. If it's
        set to ``True``, detaching will not be performed.
        Default: ``False``.

Returns:
    dict:
        a dictionary containing a whole state of the module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> module.state_dict().keys()
    ['bias', 'weight']

##### `to(self, *args, **kwargs)`

Move and/or cast the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)
   :noindex:

.. function:: to(dtype, non_blocking=False)
   :noindex:

.. function:: to(tensor, non_blocking=False)
   :noindex:

.. function:: to(memory_format=torch.channels_last)
   :noindex:

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point or complex :attr:`dtype`\ s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

Args:
    device (:class:`torch.device`): the desired device of the parameters
        and buffers in this module
    dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
        the parameters and buffers in this module
    tensor (torch.Tensor): Tensor whose dtype and device are the desired
        dtype and device for all parameters and buffers in this module
    memory_format (:class:`torch.memory_format`): the desired memory
        format for 4D parameters and buffers in this module (keyword
        only argument)

Returns:
    Module: self

Examples::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)

    >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.3741+0.j,  0.2382+0.j],
            [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
    >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
    tensor([[0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

##### `to_empty(self: ~T, *, device: Union[int, str, torch.device, NoneType], recurse: bool = True) -> ~T`

Move the parameters and buffers to the specified device without copying storage.

Args:
    device (:class:`torch.device`): The desired device of the parameters
        and buffers in this module.
    recurse (bool): Whether parameters and buffers of submodules should
        be recursively moved to the specified device.

Returns:
    Module: self

##### `train(self: ~T, mode: bool = True) -> ~T`

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

Args:
    mode (bool): whether to set training mode (``True``) or evaluation
                 mode (``False``). Default: ``True``.

Returns:
    Module: self

##### `type(self: ~T, dst_type: Union[torch.dtype, str]) -> ~T`

Casts all parameters and buffers to :attr:`dst_type`.

.. note::
    This method modifies the module in-place.

Args:
    dst_type (type or string): the desired type

Returns:
    Module: self

##### `xpu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `zero_grad(self, set_to_none: bool = True) -> None`

Reset gradients of all model parameters.

See similar function under :class:`torch.optim.Optimizer` for more context.

Args:
    set_to_none (bool): instead of setting to zero, set the grads to None.
        See :meth:`torch.optim.Optimizer.zero_grad` for details.

---

### class `NormalizedRewardOrchestrator`

```python
NormalizedRewardOrchestrator(config: Dict[str, Any])
```

Central orchestrator with reward normalization and curriculum support.

Combines all reward components with proper scaling and curriculum-based weighting.

#### Methods

##### `__init__(self, config: Dict[str, Any])`

Initialize self.  See help(type(self)) for accurate signature.

##### `calculate_total_reward(self, trajectory: List[Dict[str, Any]], final_answer: Any, ground_truth: Any, state_embeddings: Optional[List[torch.Tensor]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`

Calculate total reward with all components.

Args:
    trajectory: Action trajectory
    final_answer: Model's final answer
    ground_truth: Ground truth answer
    state_embeddings: Optional state embeddings
    context: Additional context
    
Returns:
    Dictionary with all reward information

##### `update_step(self, step: int)`

Update current training step for curriculum.

---

### class `PIIRedactor`

```python
PIIRedactor(config: Optional[core.modules.privacy.PrivacyConfig] = None)
```

Detects and redacts personally identifiable information from text.

Implements comprehensive PII detection for:
- Names (persons, organizations, locations)
- Contact information (emails, phones)
- Identification numbers (SSN, passport, etc.)
- Financial information (credit cards, accounts)
- Network identifiers (IP addresses, MACs)
- Personal URLs and social media handles

#### Methods

##### `__init__(self, config: Optional[core.modules.privacy.PrivacyConfig] = None)`

Initialize the PII redactor.

Args:
    config: Privacy configuration

##### `clear_cache(self)`

Clear the redaction cache.

##### `detect_pii(self, text: str) -> Dict[str, List[str]]`

Detect PII in text without redacting.

Args:
    text: Text to analyze
    
Returns:
    Dictionary mapping PII types to detected values

##### `get_risk_assessment(self, text: str) -> Dict[str, Any]`

Assess privacy risk of text.

Args:
    text: Text to assess
    
Returns:
    Risk assessment report

##### `get_statistics(self) -> Dict[str, Any]`

Get redaction statistics.

##### `redact_text(self, text: str) -> Tuple[str, Dict[str, int]]`

Redact PII from text.

Args:
    text: Text to redact
    
Returns:
    Tuple of (redacted_text, redaction_counts)

---

### class `PerformanceAwareCuriosityModule`

```python
PerformanceAwareCuriosityModule(state_dim: int = 768, action_dim: int = 128, hidden_dim: int = 256, beta: float = 0.2, eta: float = 0.5, cache_size: int = 1000, device: str = 'cuda')
```

Performance-aware curiosity module with efficient caching and LoRA dynamics.

Implements intrinsic curiosity reward based on prediction error while
maintaining computational efficiency through caching and lightweight models.

#### Methods

##### `__call__(self, *args, **kwargs)`

##### `__delattr__(self, name)`

Implement delattr(self, name).

##### `__dir__(self)`

Default dir() implementation.

##### `__getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]`

##### `__getstate__(self)`

Helper for pickle.

##### `__init__(self, state_dim: int = 768, action_dim: int = 128, hidden_dim: int = 256, beta: float = 0.2, eta: float = 0.5, cache_size: int = 1000, device: str = 'cuda')`

Initialize internal Module state, shared by both nn.Module and ScriptModule.

##### `__repr__(self)`

Return repr(self).

##### `__setattr__(self, name: str, value: Union[torch.Tensor, ForwardRef('Module')]) -> None`

Implement setattr(self, name, value).

##### `__setstate__(self, state)`

##### `add_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Args:
    name (str): name of the child module. The child module can be
        accessed from this module using the given name
    module (Module): child module to be added to the module.

##### `apply(self: ~T, fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T`

Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

Args:
    fn (:class:`Module` -> None): function to be applied to each submodule

Returns:
    Module: self

Example::

    >>> @torch.no_grad()
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )

##### `bfloat16(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``bfloat16`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `buffers(self, recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]`

Return an iterator over module buffers.

Args:
    recurse (bool): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module.

Yields:
    torch.Tensor: module buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for buf in model.buffers():
    >>>     print(type(buf), buf.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `children(self) -> collections.abc.Iterator['Module']`

Return an iterator over immediate children modules.

Yields:
    Module: a child module

##### `compile(self, *args, **kwargs)`

Compile this Module's forward using :func:`torch.compile`.

This Module's `__call__` method is compiled and all arguments are passed as-is
to :func:`torch.compile`.

See :func:`torch.compile` for details on the arguments for this function.

##### `compute_curiosity_reward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, return_losses: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]`

Compute curiosity reward based on prediction error.

Args:
    state: Current state [batch_size, state_dim]
    action: Action taken [batch_size, action_dim]
    next_state: Resulting state [batch_size, state_dim]
    return_losses: Whether to compute and return losses
    
Returns:
    Tuple of (curiosity_reward, metrics_dict)

##### `cpu(self: ~T) -> ~T`

Move all model parameters and buffers to the CPU.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `cuda(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

.. note::
    This method modifies the module in-place.

Args:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `double(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``double`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `eval(self: ~T) -> ~T`

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

See :ref:`locally-disable-grad-doc` for a comparison between
`.eval()` and several similar mechanisms that may be confused with it.

Returns:
    Module: self

##### `extra_repr(self) -> str`

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

##### `float(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``float`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `forward(self, *input: Any) -> None`

Define the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

##### `get_buffer(self, target: str) -> 'Tensor'`

Return the buffer given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the buffer
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.Tensor: The buffer referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not a
        buffer

##### `get_extra_state(self) -> Any`

Return any extra state to include in the module's state_dict.

Implement this and a corresponding :func:`set_extra_state` for your module
if you need to store extra state. This function is called when building the
module's `state_dict()`.

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:
    object: Any extra state to store in the module's state_dict

##### `get_parameter(self, target: str) -> 'Parameter'`

Return the parameter given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the Parameter
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Parameter: The Parameter referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not an
        ``nn.Parameter``

##### `get_submodule(self, target: str) -> 'Module'`

Return the submodule given by ``target`` if it exists, otherwise throw an error.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
            )
            (linear): Linear(in_features=100, out_features=200, bias=True)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To check whether or not we have the ``linear`` submodule, we
would call ``get_submodule("net_b.linear")``. To check whether
we have the ``conv`` submodule, we would call
``get_submodule("net_b.net_c.conv")``.

The runtime of ``get_submodule`` is bounded by the degree
of module nesting in ``target``. A query against
``named_modules`` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, ``get_submodule`` should always be
used.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Module: The submodule referenced by ``target``

Raises:
    AttributeError: If at any point along the path resulting from
        the target string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `half(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``half`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `ipu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `load_state_dict(self, state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)`

Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

Args:
    state_dict (dict): a dict containing parameters and
        persistent buffers.
    strict (bool, optional): whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    assign (bool, optional): When set to ``False``, the properties of the tensors
        in the current module are preserved whereas setting it to ``True`` preserves
        properties of the Tensors in the state dict. The only
        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
        for which the value from the module is preserved.
        Default: ``False``

Returns:
    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
        * **missing_keys** is a list of str containing any keys that are expected
            by this module but missing from the provided ``state_dict``.
        * **unexpected_keys** is a list of str containing the keys that are not
            expected by this module but present in the provided ``state_dict``.

Note:
    If a parameter or buffer is registered as ``None`` and its corresponding key
    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    ``RuntimeError``.

##### `modules(self) -> collections.abc.Iterator['Module']`

Return an iterator over all modules in the network.

Yields:
    Module: a module in the network

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
    ...     print(idx, '->', m)

    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)

##### `mtia(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]`

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Args:
    prefix (str): prefix to prepend to all buffer names.
    recurse (bool, optional): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module. Defaults to True.
    remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

Yields:
    (str, torch.Tensor): Tuple containing the name and buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, buf in self.named_buffers():
    >>>     if name in ['running_var']:
    >>>         print(buf.size())

##### `named_children(self) -> collections.abc.Iterator[tuple[str, 'Module']]`

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:
    (str, Module): Tuple containing a name and child module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)

##### `named_modules(self, memo: Optional[set['Module']] = None, prefix: str = '', remove_duplicate: bool = True)`

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Args:
    memo: a memo to store the set of modules already added to the result
    prefix: a prefix that will be added to the name of the module
    remove_duplicate: whether to remove the duplicated module instances in the result
        or not

Yields:
    (str, Module): Tuple of name and module

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
    ...     print(idx, '->', m)

    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

##### `named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]`

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Args:
    prefix (str): prefix to prepend to all parameter names.
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.
    remove_duplicate (bool, optional): whether to remove the duplicated
        parameters in the result. Defaults to True.

Yields:
    (str, Parameter): Tuple containing the name and parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, param in self.named_parameters():
    >>>     if name in ['bias']:
    >>>         print(param.size())

##### `parameters(self, recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]`

Return an iterator over module parameters.

This is typically passed to an optimizer.

Args:
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.

Yields:
    Parameter: module parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for param in model.parameters():
    >>>     print(type(param), param.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `register_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
the behavior of this function will change in future versions.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None`

Add a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting :attr:`persistent` to ``False``. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
:attr:`state_dict`.

Buffers can be accessed as attributes using given names.

Args:
    name (str): name of the buffer. The buffer can be accessed
        from this module using the given name
    tensor (Tensor or None): buffer to be registered. If ``None``, then operations
        that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
        the buffer is **not** included in the module's :attr:`state_dict`.
    persistent (bool): whether the buffer is part of this module's
        :attr:`state_dict`.

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> self.register_buffer('running_mean', torch.zeros(num_features))

##### `register_forward_hook(self, hook: Union[Callable[[~T, tuple[Any, ...], Any], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]]], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.

If ``with_kwargs`` is ``False`` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after :func:`forward` is called. The hook
should have the following signature::

    hook(module, args, output) -> None or modified output

If ``with_kwargs`` is ``True``, the forward hook will be passed the
``kwargs`` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature::

    hook(module, args, kwargs, output) -> None or modified output

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If ``True``, the provided ``hook`` will be fired
        before all existing ``forward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``forward`` hooks registered with
        :func:`register_module_forward_hook` will fire before all hooks
        registered by this method.
        Default: ``False``
    with_kwargs (bool): If ``True``, the ``hook`` will be passed the
        kwargs given to the forward function.
        Default: ``False``
    always_call (bool): If ``True`` the ``hook`` will be run regardless of
        whether an exception is raised while calling the Module.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_forward_pre_hook(self, hook: Union[Callable[[~T, tuple[Any, ...]], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any]], Optional[tuple[Any, dict[str, Any]]]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.


If ``with_kwargs`` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature::

    hook(module, args) -> None or modified input

If ``with_kwargs`` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature::

    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``forward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``forward_pre`` hooks registered with
        :func:`register_module_forward_pre_hook` will fire before all
        hooks registered by this method.
        Default: ``False``
    with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
        given to the forward function.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module
are computed, i.e. the hook will execute if and only if the gradients with
respect to module outputs are computed. The hook should have the following
signature::

    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of :attr:`grad_input` in
subsequent computations. :attr:`grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs or outputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``backward`` hooks registered with
        :func:`register_module_full_backward_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_pre_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature::

    hook(module, grad_output) -> tuple[Tensor] or None

The :attr:`grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of :attr:`grad_output` in
subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``backward_pre`` hooks registered with
        :func:`register_module_full_backward_pre_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_post_hook(self, hook)`

Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, incompatible_keys) -> None

The ``module`` argument is the current module that this hook is registered
on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
is a ``list`` of ``str`` containing the missing keys and
``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling :func:`load_state_dict` with
``strict=True`` are affected by modifications the hook makes to
``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
set of keys will result in an error being thrown when ``strict=True``, and
clearing out both missing and unexpected keys will avoid an error.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_pre_hook(self, hook)`

Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

Arguments:
    hook (Callable): Callable hook that will be invoked before
        loading the state dict.

##### `register_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Alias for :func:`add_module`.

##### `register_parameter(self, name: str, param: Optional[torch.nn.parameter.Parameter]) -> None`

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Args:
    name (str): name of the parameter. The parameter can be accessed
        from this module using the given name
    param (Parameter or None): parameter to be added to the module. If
        ``None``, then operations that run on parameters, such as :attr:`cuda`,
        are ignored. If ``None``, the parameter is **not** included in the
        module's :attr:`state_dict`.

##### `register_state_dict_post_hook(self, hook)`

Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the ``state_dict`` inplace.

##### `register_state_dict_pre_hook(self, hook)`

Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the ``state_dict``
call is made.

##### `requires_grad_(self: ~T, requires_grad: bool = True) -> ~T`

Change if autograd should record operations on parameters in this module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See :ref:`locally-disable-grad-doc` for a comparison between
`.requires_grad_()` and several similar mechanisms that may be confused with it.

Args:
    requires_grad (bool): whether autograd should record operations on
                          parameters in this module. Default: ``True``.

Returns:
    Module: self

##### `set_extra_state(self, state: Any) -> None`

Set extra state contained in the loaded `state_dict`.

This function is called from :func:`load_state_dict` to handle any extra state
found within the `state_dict`. Implement this function and a corresponding
:func:`get_extra_state` for your module if you need to store extra state within its
`state_dict`.

Args:
    state (dict): Extra state from the `state_dict`

##### `set_submodule(self, target: str, module: 'Module', strict: bool = False) -> None`

Set the submodule given by ``target`` if it exists, otherwise throw an error.

.. note::
    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
    the method will only attempt to replace an existing submodule and throw an error if
    the submodule does not exist.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(3, 3, 3)
            )
            (linear): Linear(3, 3)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To override the ``Conv2d`` with a new submodule ``Linear``, you
could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
where ``strict`` could be ``True`` or ``False``

To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

In the above if you set ``strict=True`` and call
``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
will be raised because ``net_b`` does not have a submodule named ``conv``.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)
    module: The module to set the submodule to.
    strict: If ``False``, the method will replace an existing submodule
        or create a new submodule if the parent module exists. If ``True``,
        the method will only attempt to replace an existing submodule and throw an error
        if the submodule doesn't already exist.

Raises:
    ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
    AttributeError: If at any point along the path resulting from
        the ``target`` string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `share_memory(self: ~T) -> ~T`

See :meth:`torch.Tensor.share_memory_`.

##### `state_dict(self, *args, destination=None, prefix='', keep_vars=False)`

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to ``None`` are not included.

.. note::
    The returned object is a shallow copy. It contains references
    to the module's parameters and buffers.

.. warning::
    Currently ``state_dict()`` also accepts positional arguments for
    ``destination``, ``prefix`` and ``keep_vars`` in order. However,
    this is being deprecated and keyword arguments will be enforced in
    future releases.

.. warning::
    Please avoid the use of argument ``destination`` as it is not
    designed for end-users.

Args:
    destination (dict, optional): If provided, the state of module will
        be updated into the dict and the same object is returned.
        Otherwise, an ``OrderedDict`` will be created and returned.
        Default: ``None``.
    prefix (str, optional): a prefix added to parameter and buffer
        names to compose the keys in state_dict. Default: ``''``.
    keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
        returned in the state dict are detached from autograd. If it's
        set to ``True``, detaching will not be performed.
        Default: ``False``.

Returns:
    dict:
        a dictionary containing a whole state of the module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> module.state_dict().keys()
    ['bias', 'weight']

##### `to(self, *args, **kwargs)`

Move and/or cast the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)
   :noindex:

.. function:: to(dtype, non_blocking=False)
   :noindex:

.. function:: to(tensor, non_blocking=False)
   :noindex:

.. function:: to(memory_format=torch.channels_last)
   :noindex:

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point or complex :attr:`dtype`\ s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

Args:
    device (:class:`torch.device`): the desired device of the parameters
        and buffers in this module
    dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
        the parameters and buffers in this module
    tensor (torch.Tensor): Tensor whose dtype and device are the desired
        dtype and device for all parameters and buffers in this module
    memory_format (:class:`torch.memory_format`): the desired memory
        format for 4D parameters and buffers in this module (keyword
        only argument)

Returns:
    Module: self

Examples::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)

    >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.3741+0.j,  0.2382+0.j],
            [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
    >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
    tensor([[0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

##### `to_empty(self: ~T, *, device: Union[int, str, torch.device, NoneType], recurse: bool = True) -> ~T`

Move the parameters and buffers to the specified device without copying storage.

Args:
    device (:class:`torch.device`): The desired device of the parameters
        and buffers in this module.
    recurse (bool): Whether parameters and buffers of submodules should
        be recursively moved to the specified device.

Returns:
    Module: self

##### `train(self: ~T, mode: bool = True) -> ~T`

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

Args:
    mode (bool): whether to set training mode (``True``) or evaluation
                 mode (``False``). Default: ``True``.

Returns:
    Module: self

##### `type(self: ~T, dst_type: Union[torch.dtype, str]) -> ~T`

Casts all parameters and buffers to :attr:`dst_type`.

.. note::
    This method modifies the module in-place.

Args:
    dst_type (type or string): the desired type

Returns:
    Module: self

##### `xpu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `zero_grad(self, set_to_none: bool = True) -> None`

Reset gradients of all model parameters.

See similar function under :class:`torch.optim.Optimizer` for more context.

Args:
    set_to_none (bool): instead of setting to zero, set the grads to None.
        See :meth:`torch.optim.Optimizer.zero_grad` for details.

---

### class `PersistenceAdapter`

Abstract base class for persistence adapters.

#### Methods

##### `close(self)`

Close any open resources.

##### `load_snapshot(self) -> Optional[Dict[str, Any]]`

Load the latest snapshot.

##### `read_all_experiences(self) -> List[Dict[str, Any]]`

Read all experiences from storage.

##### `read_all_operations(self) -> List[Dict[str, Any]]`

Read all operations from the operations log.

##### `save_snapshot(self, snapshot_data: Dict[str, Any]) -> bool`

Save a complete snapshot.

##### `truncate_logs(self) -> bool`

Truncate both WAL logs after successful snapshot.

##### `write_experience(self, experience_id: str, data: Dict[str, Any]) -> bool`

Write an experience to persistent storage.

##### `write_operation(self, operation: Dict[str, Any]) -> bool`

Write an operation to the operations log.

---

### class `PrivacyConfig`

```python
PrivacyConfig(enable_pii_redaction: bool = True, enable_image_metadata_stripping: bool = True, enable_differential_privacy: bool = False, differential_privacy_epsilon: float = 1.0, log_redaction_stats: bool = True, redaction_placeholder_format: str = '[{category}]', hash_pii_for_consistency: bool = True, max_text_length: int = 10000, allowed_image_formats: Set[str] = <factory>) -> None
```

Configuration for privacy protection.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, enable_pii_redaction: bool = True, enable_image_metadata_stripping: bool = True, enable_differential_privacy: bool = False, differential_privacy_epsilon: float = 1.0, log_redaction_stats: bool = True, redaction_placeholder_format: str = '[{category}]', hash_pii_for_consistency: bool = True, max_text_length: int = 10000, allowed_image_formats: Set[str] = <factory>) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__repr__(self)`

Return repr(self).

---

### class `RedactionPattern`

```python
RedactionPattern(name: str, pattern: str, replacement: str, description: str, risk_level: str) -> None
```

Pattern for detecting and redacting sensitive information.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, name: str, pattern: str, replacement: str, description: str, risk_level: str) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__repr__(self)`

Return repr(self).

##### `compile(self)`

Compile the regex pattern.

---

### class `RewardComponents`

```python
RewardComponents(task_reward: float, curiosity_reward: float, coherence_reward: float, tool_penalty: float, total_reward: float, metadata: Dict[str, Any]) -> None
```

Container for all reward components.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, task_reward: float, curiosity_reward: float, coherence_reward: float, tool_penalty: float, total_reward: float, metadata: Dict[str, Any]) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__repr__(self)`

Return repr(self).

##### `to_dict(self) -> Dict[str, Any]`

Convert to dictionary.

---

### class `RewardOrchestrator`

```python
RewardOrchestrator(config: Dict[str, Any])
```

Central reward orchestrator that combines all reward components.

Manages task rewards, curiosity rewards, coherence rewards,
and tool usage penalties with normalization and curriculum.

#### Methods

##### `__init__(self, config: Dict[str, Any])`

Initialize reward orchestrator.

Args:
    config: Reward configuration

##### `calculate_reward(self, trajectory: core.data_structures.Trajectory, final_answer: Any, ground_truth: Any, state_embeddings: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]`

Calculate all reward components.

Args:
    trajectory: Reasoning trajectory
    final_answer: Model's final answer
    ground_truth: Ground truth answer (or pseudo-label)
    state_embeddings: Optional state embeddings
    
Returns:
    Dictionary with reward components and total

##### `reset_statistics(self)`

Reset reward statistics.

##### `step(self)`

Increment curriculum step.

---

### class `RunningStats`

```python
RunningStats(window_size: int = 1000)
```

Helper class for maintaining running statistics.

#### Methods

##### `__init__(self, window_size: int = 1000)`

Initialize self.  See help(type(self)) for accurate signature.

##### `normalize(self, value: float) -> float`

Normalize value using running statistics.

##### `update(self, value: float)`

Update statistics with new value.

---

### class `StateEncoder`

```python
StateEncoder(input_dim: int = 768, output_dim: int = 256, hidden_dim: int = 512, num_layers: int = 2, use_layer_norm: bool = True, device: str = 'cuda')
```

Encoder for state representations.

Transforms raw states into meaningful representations
for dynamics modeling and curiosity calculation.

#### Methods

##### `__call__(self, *args, **kwargs)`

##### `__delattr__(self, name)`

Implement delattr(self, name).

##### `__dir__(self)`

Default dir() implementation.

##### `__getattr__(self, name: str) -> Union[torch.Tensor, ForwardRef('Module')]`

##### `__getstate__(self)`

Helper for pickle.

##### `__init__(self, input_dim: int = 768, output_dim: int = 256, hidden_dim: int = 512, num_layers: int = 2, use_layer_norm: bool = True, device: str = 'cuda')`

Initialize state encoder.

Args:
    input_dim: Input state dimension
    output_dim: Output representation dimension
    hidden_dim: Hidden layer dimension
    num_layers: Number of layers
    use_layer_norm: Whether to use layer normalization
    device: Device for computation

##### `__repr__(self)`

Return repr(self).

##### `__setattr__(self, name: str, value: Union[torch.Tensor, ForwardRef('Module')]) -> None`

Implement setattr(self, name, value).

##### `__setstate__(self, state)`

##### `add_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Args:
    name (str): name of the child module. The child module can be
        accessed from this module using the given name
    module (Module): child module to be added to the module.

##### `apply(self: ~T, fn: Callable[[ForwardRef('Module')], NoneType]) -> ~T`

Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

Args:
    fn (:class:`Module` -> None): function to be applied to each submodule

Returns:
    Module: self

Example::

    >>> @torch.no_grad()
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )

##### `bfloat16(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``bfloat16`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `buffers(self, recurse: bool = True) -> collections.abc.Iterator[torch.Tensor]`

Return an iterator over module buffers.

Args:
    recurse (bool): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module.

Yields:
    torch.Tensor: module buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for buf in model.buffers():
    >>>     print(type(buf), buf.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `children(self) -> collections.abc.Iterator['Module']`

Return an iterator over immediate children modules.

Yields:
    Module: a child module

##### `compile(self, *args, **kwargs)`

Compile this Module's forward using :func:`torch.compile`.

This Module's `__call__` method is compiled and all arguments are passed as-is
to :func:`torch.compile`.

See :func:`torch.compile` for details on the arguments for this function.

##### `cpu(self: ~T) -> ~T`

Move all model parameters and buffers to the CPU.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `cuda(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

.. note::
    This method modifies the module in-place.

Args:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `double(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``double`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `eval(self: ~T) -> ~T`

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

See :ref:`locally-disable-grad-doc` for a comparison between
`.eval()` and several similar mechanisms that may be confused with it.

Returns:
    Module: self

##### `extra_repr(self) -> str`

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

##### `float(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``float`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `forward(self, state: torch.Tensor) -> torch.Tensor`

Encode state to representation.

Args:
    state: Input state tensor
    
Returns:
    Encoded representation

##### `get_buffer(self, target: str) -> 'Tensor'`

Return the buffer given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the buffer
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.Tensor: The buffer referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not a
        buffer

##### `get_extra_state(self) -> Any`

Return any extra state to include in the module's state_dict.

Implement this and a corresponding :func:`set_extra_state` for your module
if you need to store extra state. This function is called when building the
module's `state_dict()`.

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:
    object: Any extra state to store in the module's state_dict

##### `get_parameter(self, target: str) -> 'Parameter'`

Return the parameter given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the Parameter
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Parameter: The Parameter referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not an
        ``nn.Parameter``

##### `get_submodule(self, target: str) -> 'Module'`

Return the submodule given by ``target`` if it exists, otherwise throw an error.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
            )
            (linear): Linear(in_features=100, out_features=200, bias=True)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To check whether or not we have the ``linear`` submodule, we
would call ``get_submodule("net_b.linear")``. To check whether
we have the ``conv`` submodule, we would call
``get_submodule("net_b.net_c.conv")``.

The runtime of ``get_submodule`` is bounded by the degree
of module nesting in ``target``. A query against
``named_modules`` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, ``get_submodule`` should always be
used.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Module: The submodule referenced by ``target``

Raises:
    AttributeError: If at any point along the path resulting from
        the target string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `half(self: ~T) -> ~T`

Casts all floating point parameters and buffers to ``half`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `ipu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `load_state_dict(self, state_dict: collections.abc.Mapping[str, typing.Any], strict: bool = True, assign: bool = False)`

Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

Args:
    state_dict (dict): a dict containing parameters and
        persistent buffers.
    strict (bool, optional): whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    assign (bool, optional): When set to ``False``, the properties of the tensors
        in the current module are preserved whereas setting it to ``True`` preserves
        properties of the Tensors in the state dict. The only
        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
        for which the value from the module is preserved.
        Default: ``False``

Returns:
    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
        * **missing_keys** is a list of str containing any keys that are expected
            by this module but missing from the provided ``state_dict``.
        * **unexpected_keys** is a list of str containing the keys that are not
            expected by this module but present in the provided ``state_dict``.

Note:
    If a parameter or buffer is registered as ``None`` and its corresponding key
    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    ``RuntimeError``.

##### `modules(self) -> collections.abc.Iterator['Module']`

Return an iterator over all modules in the network.

Yields:
    Module: a module in the network

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
    ...     print(idx, '->', m)

    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)

##### `mtia(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.Tensor]]`

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Args:
    prefix (str): prefix to prepend to all buffer names.
    recurse (bool, optional): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module. Defaults to True.
    remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

Yields:
    (str, torch.Tensor): Tuple containing the name and buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, buf in self.named_buffers():
    >>>     if name in ['running_var']:
    >>>         print(buf.size())

##### `named_children(self) -> collections.abc.Iterator[tuple[str, 'Module']]`

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:
    (str, Module): Tuple containing a name and child module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)

##### `named_modules(self, memo: Optional[set['Module']] = None, prefix: str = '', remove_duplicate: bool = True)`

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Args:
    memo: a memo to store the set of modules already added to the result
    prefix: a prefix that will be added to the name of the module
    remove_duplicate: whether to remove the duplicated module instances in the result
        or not

Yields:
    (str, Module): Tuple of name and module

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
    ...     print(idx, '->', m)

    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

##### `named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]`

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Args:
    prefix (str): prefix to prepend to all parameter names.
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.
    remove_duplicate (bool, optional): whether to remove the duplicated
        parameters in the result. Defaults to True.

Yields:
    (str, Parameter): Tuple containing the name and parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, param in self.named_parameters():
    >>>     if name in ['bias']:
    >>>         print(param.size())

##### `parameters(self, recurse: bool = True) -> collections.abc.Iterator[torch.nn.parameter.Parameter]`

Return an iterator over module parameters.

This is typically passed to an optimizer.

Args:
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.

Yields:
    Parameter: module parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for param in model.parameters():
    >>>     print(type(param), param.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `register_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
the behavior of this function will change in future versions.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None`

Add a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting :attr:`persistent` to ``False``. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
:attr:`state_dict`.

Buffers can be accessed as attributes using given names.

Args:
    name (str): name of the buffer. The buffer can be accessed
        from this module using the given name
    tensor (Tensor or None): buffer to be registered. If ``None``, then operations
        that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
        the buffer is **not** included in the module's :attr:`state_dict`.
    persistent (bool): whether the buffer is part of this module's
        :attr:`state_dict`.

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> self.register_buffer('running_mean', torch.zeros(num_features))

##### `register_forward_hook(self, hook: Union[Callable[[~T, tuple[Any, ...], Any], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]]], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.

If ``with_kwargs`` is ``False`` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after :func:`forward` is called. The hook
should have the following signature::

    hook(module, args, output) -> None or modified output

If ``with_kwargs`` is ``True``, the forward hook will be passed the
``kwargs`` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature::

    hook(module, args, kwargs, output) -> None or modified output

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If ``True``, the provided ``hook`` will be fired
        before all existing ``forward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``forward`` hooks registered with
        :func:`register_module_forward_hook` will fire before all hooks
        registered by this method.
        Default: ``False``
    with_kwargs (bool): If ``True``, the ``hook`` will be passed the
        kwargs given to the forward function.
        Default: ``False``
    always_call (bool): If ``True`` the ``hook`` will be run regardless of
        whether an exception is raised while calling the Module.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_forward_pre_hook(self, hook: Union[Callable[[~T, tuple[Any, ...]], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any]], Optional[tuple[Any, dict[str, Any]]]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.


If ``with_kwargs`` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature::

    hook(module, args) -> None or modified input

If ``with_kwargs`` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature::

    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``forward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``forward_pre`` hooks registered with
        :func:`register_module_forward_pre_hook` will fire before all
        hooks registered by this method.
        Default: ``False``
    with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
        given to the forward function.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module
are computed, i.e. the hook will execute if and only if the gradients with
respect to module outputs are computed. The hook should have the following
signature::

    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of :attr:`grad_input` in
subsequent computations. :attr:`grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs or outputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``backward`` hooks registered with
        :func:`register_module_full_backward_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_pre_hook(self, hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle`

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature::

    hook(module, grad_output) -> tuple[Tensor] or None

The :attr:`grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of :attr:`grad_output` in
subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``backward_pre`` hooks registered with
        :func:`register_module_full_backward_pre_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_post_hook(self, hook)`

Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, incompatible_keys) -> None

The ``module`` argument is the current module that this hook is registered
on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
is a ``list`` of ``str`` containing the missing keys and
``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling :func:`load_state_dict` with
``strict=True`` are affected by modifications the hook makes to
``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
set of keys will result in an error being thrown when ``strict=True``, and
clearing out both missing and unexpected keys will avoid an error.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_pre_hook(self, hook)`

Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

Arguments:
    hook (Callable): Callable hook that will be invoked before
        loading the state dict.

##### `register_module(self, name: str, module: Optional[ForwardRef('Module')]) -> None`

Alias for :func:`add_module`.

##### `register_parameter(self, name: str, param: Optional[torch.nn.parameter.Parameter]) -> None`

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Args:
    name (str): name of the parameter. The parameter can be accessed
        from this module using the given name
    param (Parameter or None): parameter to be added to the module. If
        ``None``, then operations that run on parameters, such as :attr:`cuda`,
        are ignored. If ``None``, the parameter is **not** included in the
        module's :attr:`state_dict`.

##### `register_state_dict_post_hook(self, hook)`

Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the ``state_dict`` inplace.

##### `register_state_dict_pre_hook(self, hook)`

Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the ``state_dict``
call is made.

##### `requires_grad_(self: ~T, requires_grad: bool = True) -> ~T`

Change if autograd should record operations on parameters in this module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See :ref:`locally-disable-grad-doc` for a comparison between
`.requires_grad_()` and several similar mechanisms that may be confused with it.

Args:
    requires_grad (bool): whether autograd should record operations on
                          parameters in this module. Default: ``True``.

Returns:
    Module: self

##### `set_extra_state(self, state: Any) -> None`

Set extra state contained in the loaded `state_dict`.

This function is called from :func:`load_state_dict` to handle any extra state
found within the `state_dict`. Implement this function and a corresponding
:func:`get_extra_state` for your module if you need to store extra state within its
`state_dict`.

Args:
    state (dict): Extra state from the `state_dict`

##### `set_submodule(self, target: str, module: 'Module', strict: bool = False) -> None`

Set the submodule given by ``target`` if it exists, otherwise throw an error.

.. note::
    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
    the method will only attempt to replace an existing submodule and throw an error if
    the submodule does not exist.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(3, 3, 3)
            )
            (linear): Linear(3, 3)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To override the ``Conv2d`` with a new submodule ``Linear``, you
could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
where ``strict`` could be ``True`` or ``False``

To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

In the above if you set ``strict=True`` and call
``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
will be raised because ``net_b`` does not have a submodule named ``conv``.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)
    module: The module to set the submodule to.
    strict: If ``False``, the method will replace an existing submodule
        or create a new submodule if the parent module exists. If ``True``,
        the method will only attempt to replace an existing submodule and throw an error
        if the submodule doesn't already exist.

Raises:
    ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
    AttributeError: If at any point along the path resulting from
        the ``target`` string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `share_memory(self: ~T) -> ~T`

See :meth:`torch.Tensor.share_memory_`.

##### `state_dict(self, *args, destination=None, prefix='', keep_vars=False)`

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to ``None`` are not included.

.. note::
    The returned object is a shallow copy. It contains references
    to the module's parameters and buffers.

.. warning::
    Currently ``state_dict()`` also accepts positional arguments for
    ``destination``, ``prefix`` and ``keep_vars`` in order. However,
    this is being deprecated and keyword arguments will be enforced in
    future releases.

.. warning::
    Please avoid the use of argument ``destination`` as it is not
    designed for end-users.

Args:
    destination (dict, optional): If provided, the state of module will
        be updated into the dict and the same object is returned.
        Otherwise, an ``OrderedDict`` will be created and returned.
        Default: ``None``.
    prefix (str, optional): a prefix added to parameter and buffer
        names to compose the keys in state_dict. Default: ``''``.
    keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
        returned in the state dict are detached from autograd. If it's
        set to ``True``, detaching will not be performed.
        Default: ``False``.

Returns:
    dict:
        a dictionary containing a whole state of the module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> module.state_dict().keys()
    ['bias', 'weight']

##### `to(self, *args, **kwargs)`

Move and/or cast the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)
   :noindex:

.. function:: to(dtype, non_blocking=False)
   :noindex:

.. function:: to(tensor, non_blocking=False)
   :noindex:

.. function:: to(memory_format=torch.channels_last)
   :noindex:

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point or complex :attr:`dtype`\ s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

Args:
    device (:class:`torch.device`): the desired device of the parameters
        and buffers in this module
    dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
        the parameters and buffers in this module
    tensor (torch.Tensor): Tensor whose dtype and device are the desired
        dtype and device for all parameters and buffers in this module
    memory_format (:class:`torch.memory_format`): the desired memory
        format for 4D parameters and buffers in this module (keyword
        only argument)

Returns:
    Module: self

Examples::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)

    >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.3741+0.j,  0.2382+0.j],
            [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
    >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
    tensor([[0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

##### `to_empty(self: ~T, *, device: Union[int, str, torch.device, NoneType], recurse: bool = True) -> ~T`

Move the parameters and buffers to the specified device without copying storage.

Args:
    device (:class:`torch.device`): The desired device of the parameters
        and buffers in this module.
    recurse (bool): Whether parameters and buffers of submodules should
        be recursively moved to the specified device.

Returns:
    Module: self

##### `train(self: ~T, mode: bool = True) -> ~T`

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

Args:
    mode (bool): whether to set training mode (``True``) or evaluation
                 mode (``False``). Default: ``True``.

Returns:
    Module: self

##### `type(self: ~T, dst_type: Union[torch.dtype, str]) -> ~T`

Casts all parameters and buffers to :attr:`dst_type`.

.. note::
    This method modifies the module in-place.

Args:
    dst_type (type or string): the desired type

Returns:
    Module: self

##### `xpu(self: ~T, device: Union[int, torch.device, NoneType] = None) -> ~T`

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `zero_grad(self, set_to_none: bool = True) -> None`

Reset gradients of all model parameters.

See similar function under :class:`torch.optim.Optimizer` for more context.

Args:
    set_to_none (bool): instead of setting to zero, set the grads to None.
        See :meth:`torch.optim.Optimizer.zero_grad` for details.

---

### class `TemporalEnsembleVoting`

```python
TemporalEnsembleVoting(strategy: str = 'weighted', min_votes_required: int = 3, confidence_threshold: float = 0.5)
```

Temporal ensemble voting module for aggregating predictions.

Supports multiple voting strategies:
- Majority voting
- Weighted voting based on confidence/similarity
- Confidence-based voting
- Ensemble methods

#### Methods

##### `__init__(self, strategy: str = 'weighted', min_votes_required: int = 3, confidence_threshold: float = 0.5)`

Initialize voting module.

Args:
    strategy: Voting strategy ('majority', 'weighted', 'confidence', 'ensemble')
    min_votes_required: Minimum number of votes needed
    confidence_threshold: Minimum confidence for vote validity

##### `analyze_voting_consistency(self, voting_history: List[core.data_structures.VotingResult]) -> Dict[str, Any]`

Analyze consistency of voting over time.

Args:
    voting_history: List of past voting results
    
Returns:
    Analysis dictionary

##### `vote(self, initial_prediction: Dict[str, Any], neighbors: List[core.data_structures.Experience], **kwargs) -> core.data_structures.VotingResult`

Perform voting based on initial prediction and neighbors.

Args:
    initial_prediction: Model's initial prediction
    neighbors: List of neighbor experiences
    **kwargs: Additional arguments for specific strategies
    
Returns:
    VotingResult with final answer and metadata

---

### class `ToolMisusePenaltySystem`

```python
ToolMisusePenaltySystem(base_penalty: float = 0.1, severe_penalty_multiplier: float = 2.0)
```

System for calculating penalties for incorrect tool usage.

Enforces proper tool sequencing and parameter validation.

#### Methods

##### `__init__(self, base_penalty: float = 0.1, severe_penalty_multiplier: float = 2.0)`

Initialize self.  See help(type(self)) for accurate signature.

##### `calculate_penalties(self, trajectory: List[Dict[str, Any]], context: Dict[str, Any]) -> Tuple[float, Dict[str, int]]`

Calculate total penalties for tool misuse.

Args:
    trajectory: Action trajectory
    context: Context information (input type, etc.)
    
Returns:
    Tuple of (total_penalty, violation_counts)

---

### class `TrajectoryCoherenceAnalyzer`

```python
TrajectoryCoherenceAnalyzer(coherence_threshold: float = 0.7, repetition_penalty: float = 0.5, min_trajectory_length: int = 2)
```

Analyzes trajectory coherence and logical flow.

Rewards coherent reasoning and penalizes repetitive or
contradictory actions.

#### Methods

##### `__init__(self, coherence_threshold: float = 0.7, repetition_penalty: float = 0.5, min_trajectory_length: int = 2)`

Initialize coherence analyzer.

Args:
    coherence_threshold: Minimum coherence score
    repetition_penalty: Penalty for repetitive actions
    min_trajectory_length: Minimum length for analysis

##### `calculate_coherence_reward(self, trajectory: core.data_structures.Trajectory, embeddings: Optional[List[torch.Tensor]] = None) -> float`

Calculate coherence reward for trajectory.

Args:
    trajectory: Reasoning trajectory
    embeddings: Optional embeddings for semantic similarity
    
Returns:
    Coherence reward value

---

### class `VisualOperationRegistry`

Singleton registry for managing visual operations.

This class maintains a central registry of all available visual operations
and provides methods to register new operations and execute them by name.

#### Methods

##### `__init__(self)`

Initialize the registry.

##### `__new__(cls)`

Ensure singleton pattern - only one instance exists.

##### `__repr__(self) -> str`

String representation of the registry.

##### `clear(self) -> None`

Clear all registered operations.

Use with caution - this removes all registered operations.

##### `execute(self, operation_name: str, **kwargs) -> Any`

Execute a registered visual operation.

Args:
    operation_name: Name of the operation to execute
    **kwargs: Arguments to pass to the operation's run method
    
Returns:
    Result from the operation execution
    
Raises:
    NotImplementedError: If operation is not registered
    Exception: Any exception raised by the operation itself

##### `get_operation_class(self, operation_name: str) -> Optional[Type[core.modules.operation_registry.BaseOperation]]`

Get the class for a registered operation.

Args:
    operation_name: Name of the operation
    
Returns:
    Operation class if registered, None otherwise

##### `get_operation_metadata(self, operation_name: str) -> Optional[Dict[str, Any]]`

Get metadata for a registered operation.

Args:
    operation_name: Name of the operation
    
Returns:
    Operation metadata if registered, None otherwise

##### `has_operation(self, operation_name: str) -> bool`

Check if an operation is registered.

Args:
    operation_name: Name of the operation to check
    
Returns:
    True if operation is registered, False otherwise

##### `list_operations(self) -> Dict[str, Dict[str, Any]]`

List all registered operations with their metadata.

Returns:
    Dictionary mapping operation names to their metadata

##### `register(self, operation_name: str, operation_class: Type[core.modules.operation_registry.BaseOperation], metadata: Optional[Dict[str, Any]] = None) -> None`

Register a new visual operation.

Args:
    operation_name: Name identifier for the operation (e.g., 'SEGMENT_OBJECT_AT')
    operation_class: Class implementing the operation (must inherit from BaseOperation)
    metadata: Optional metadata about the operation (description, parameters, etc.)

Raises:
    ValueError: If operation_name already exists or operation_class is invalid

##### `unregister(self, operation_name: str) -> bool`

Unregister a visual operation.

Args:
    operation_name: Name of the operation to unregister
    
Returns:
    True if successfully unregistered, False if operation not found

---

## Functions

### `audit_log(event_type: core.modules.audit.AuditEventType, actor: str, action: str, resource: str, result: core.modules.audit.AuditResult = <AuditResult.SUCCESS: 'success'>, metadata: Optional[Dict[str, Any]] = None) -> bool`

Convenience function to log an audit event.

Args:
    event_type: Type of event
    actor: Who performed the action
    action: What action was performed
    resource: What resource was affected
    result: Result of the action
    metadata: Additional event data
    
Returns:
    True if successfully logged

---

### `create_persistence_adapter(adapter_type: str, base_path: str) -> core.modules.persistence_adapter.PersistenceAdapter`

Factory function to create persistence adapters.

Args:
    adapter_type: Type of adapter ('file' or 'lmdb')
    base_path: Base path for persistence
    
Returns:
    PersistenceAdapter instance

---

### `get_audit_logger() -> Optional[core.modules.audit.AuditLogger]`

Get the global audit logger instance.

---

### `initialize_audit_logger(config: Dict[str, Any]) -> core.modules.audit.AuditLogger`

Initialize the global audit logger.

Args:
    config: Audit configuration
    
Returns:
    Initialized audit logger

---

### `register_operation(name: str, metadata: Optional[Dict[str, Any]] = None) -> Callable`

Decorator to automatically register an operation class.

Usage:
    @register_operation("SEGMENT_OBJECT_AT")
    class SegmentObjectOperation(BaseOperation):
        ...

Args:
    name: Name to register the operation under
    metadata: Optional metadata for the operation
    
Returns:
    Decorator function

---

