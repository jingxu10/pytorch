.. meta::
   :description: A guide to torch.xpu, a PyTorch module to run XPU operations
   :keywords: optimize PyTorch, XPU

.. _xpu-semantics:

XPU semantics
==============


:mod:`torch.xpu` is used to set up and run XPU operations. It keeps track of
the currently selected GPU, and all XPU tensors you allocate will by default be
created on that device. The selected device can be changed with a
:any:`torch.xpu.device` context manager.

However, once a tensor is allocated, you can do operations on it irrespective
of the selected device, and the results will be always placed on the same
device as the tensor.

Cross-GPU operations are not allowed by default, with the exception of
:meth:`~torch.Tensor.copy_` and other methods with copy-like functionality
such as :meth:`~torch.Tensor.to` and :meth:`~torch.Tensor.xpu`.
Unless you enable peer-to-peer memory access, any attempts to launch ops on
tensors spread across different devices will raise an error.

Below you can find a small example showcasing this::

    xpu = torch.device('xpu')     # Default XPU device
    xpu0 = torch.device('xpu:0')
    xpu2 = torch.device('xpu:2')  # GPU 2 (these are 0-indexed)

    x = torch.tensor([1., 2.], device=xpu0)
    # x.device is device(type='xpu', index=0)
    y = torch.tensor([1., 2.]).xpu()
    # y.device is device(type='xpu', index=0)

    with torch.xpu.device(1):
        # allocates a tensor on GPU 1
        a = torch.tensor([1., 2.], device=xpu)

        # transfers a tensor from CPU to GPU 1
        b = torch.tensor([1., 2.]).xpu()
        # a.device and b.device are device(type='xpu', index=1)

        # You can also use ``Tensor.to`` to transfer a tensor:
        b2 = torch.tensor([1., 2.]).to(device=xpu)
        # b.device and b2.device are device(type='xpu', index=1)

        c = a + b
        # c.device is device(type='xpu', index=1)

        z = x + y
        # z.device is device(type='xpu', index=0)

        # even within a context, you can specify the device
        # (or give a GPU index to the .xpu call)
        d = torch.randn(2, device=xpu2)
        e = torch.randn(2).to(xpu2)
        f = torch.randn(2).xpu(xpu2)
        # d.device, e.device, and f.device are all device(type='xpu', index=2)

Asynchronous execution
----------------------

By default, GPU operations are asynchronous.  When you call a function that
uses the GPU, the operations are *enqueued* to the particular device, but not
necessarily executed until later.  This allows us to execute more computations
in parallel, including operations on CPU or other GPUs.

In general, the effect of asynchronous computation is invisible to the caller,
because (1) each device executes operations in the order they are queued, and
(2) PyTorch automatically performs necessary synchronization when copying data
between CPU and GPU or between two GPUs.  Hence, computation will proceed as if
every operation was executed synchronously.

As an exception, several functions such as :meth:`~torch.Tensor.to` and
:meth:`~torch.Tensor.copy_` admit an explicit :attr:`non_blocking` argument,
which lets the caller bypass synchronization when it is unnecessary.
Another exception is XPU streams, explained below.

XPU streams
^^^^^^^^^^^^

A `XPU stream`_ is a linear sequence of execution that belongs to a specific
device.  You normally do not need to create one explicitly. Please note that
there is no default stream on xpu.

Operations inside each stream are serialized in the order they are created,
but operations from different streams can execute concurrently in any
relative order, unless explicit synchronization functions (such as
:meth:`~torch.xpu.synchronize` or :meth:`~torch.xpu.Stream.wait_stream`) are
used.  For example, the following code is incorrect::

    xpu = torch.device('xpu')
    s = torch.xpu.Stream()  # Create a new stream.
    A = torch.empty((100, 100), device=xpu).normal_(0.0, 1.0)
    with torch.xpu.stream(s):
        # sum() may start execution before normal_() finishes!
        B = torch.sum(A)

Tt is the user's responsibility to ensure proper synchronization.  The fixed
version of this example is::

    xpu = torch.device('xpu')
    s = torch.xpu.Stream()  # Create a new stream.
    A = torch.empty((100, 100), device=xpu).normal_(0.0, 1.0)
    s.wait_stream(torch.xpu.current_stream(xpu))  # NEW!
    with torch.xpu.stream(s):
        B = torch.sum(A)
    A.record_stream(s)  # NEW!

There are two new additions.  The :meth:`torch.xpu.Stream.wait_stream` call
ensures that the ``normal_()`` execution has finished before we start running
``sum(A)`` on a side stream.  The :meth:`torch.Tensor.record_stream` (see for
more details) ensures that we do not deallocate A before ``sum(A)`` has
completed.  You can also manually wait on the stream at some later point in
time with ``torch.xpu.default_stream(xpu).wait_stream(s)`` (note that it
is pointless to wait immediately, since that will prevent the stream execution
from running in parallel with other work on the current stream.)  See the
documentation for :meth:`torch.Tensor.record_stream` on more details on when
to use one or another.

Note that this synchronization is necessary even when there is no
read dependency, e.g., as seen in this example::

    xpu = torch.device('xpu')
    s = torch.xpu.Stream()  # Create a new stream.
    A = torch.empty((100, 100), device=xpu)
    s.wait_stream(torch.xpu.default_stream(xpu))  # STILL REQUIRED!
    with torch.xpu.stream(s):
        A.normal_(0.0, 1.0)
        A.record_stream(s)

Despite the computation on ``s`` not reading the contents of ``A`` and no
other uses of ``A``, it is still necessary to synchronize, because ``A``
may correspond to memory reallocated by the XPU caching allocator, with
pending operations from the old (deallocated) memory.

.. _bwd-xpu-stream-semantics:

Stream semantics of backward passes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each backward XPU op runs on the same stream that was used for its corresponding forward op.
If your forward pass runs independent ops in parallel on different streams,
this helps the backward pass exploit that same parallelism.

The stream semantics of a backward call with respect to surrounding ops are the same
as for any other call. The backward pass inserts internal syncs to ensure this even when
backward ops run on multiple streams as described in the previous paragraph.
More concretely, when calling
:func:`autograd.backward<torch.autograd.backward>`,
:func:`autograd.grad<torch.autograd.grad>`, or
:meth:`tensor.backward<torch.Tensor.backward>`,
and optionally supplying XPU tensor(s) as the  initial gradient(s) (e.g.,
:func:`autograd.backward(..., grad_tensors=initial_grads)<torch.autograd.backward>`,
:func:`autograd.grad(..., grad_outputs=initial_grads)<torch.autograd.grad>`, or
:meth:`tensor.backward(..., gradient=initial_grad)<torch.Tensor.backward>`),
the acts of

1. optionally populating initial gradient(s),
2. invoking the backward pass, and
3. using the gradients

have the same stream-semantics relationship as any group of ops::

    s = torch.xpu.Stream()

    # Safe, grads are used in the same stream context as backward()
    with torch.xpu.stream(s):
        loss.backward()
        use grads

    # Unsafe
    with torch.xpu.stream(s):
        loss.backward()
    use grads

    # Safe, with synchronization
    with torch.xpu.stream(s):
        loss.backward()
    torch.xpu.current_stream().wait_stream(s)
    use grads

    # Safe, populating initial grad and invoking backward are in the same stream context
    with torch.xpu.stream(s):
        loss.backward(gradient=torch.ones_like(loss))

    # Unsafe, populating initial_grad and invoking backward are in different stream contexts,
    # without synchronization
    initial_grad = torch.ones_like(loss)
    with torch.xpu.stream(s):
        loss.backward(gradient=initial_grad)

    # Safe, with synchronization
    initial_grad = torch.ones_like(loss)
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        initial_grad.record_stream(s)
        loss.backward(gradient=initial_grad)

.. _xpu-memory-management:

Memory management
-----------------

PyTorch uses a caching memory allocator to speed up memory allocations. This
allows fast memory deallocation without device synchronizations. Calling :meth:`~torch.xpu.empty_cache`
releases all **unused** cached memory from PyTorch so that those can be used
by other GPU applications. However, the occupied GPU memory by tensors will not
be freed so it can not increase the amount of GPU memory available for PyTorch.


Best practices
--------------

Device-agnostic code
^^^^^^^^^^^^^^^^^^^^

Due to the structure of PyTorch, you may need to explicitly write
device-agnostic (CPU or GPU) code; an example may be creating a new tensor as
the initial hidden state of a recurrent neural network.

The first step is to determine whether the GPU should be used or not. A common
pattern is to use Python's ``argparse`` module to read in user arguments, and
have a flag that can be used to disable XPU, in combination with
:meth:`~torch.xpu.is_available`. In the following, ``args.device`` results in a
:class:`torch.device` object that can be used to move tensors to CPU or XPU.

::

    import argparse
    import torch

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-xpu', action='store_true',
                        help='Disable XPU')
    args = parser.parse_args()
    args.device = None
    if not args.disable_xpu and torch.xpu.is_available():
        args.device = torch.device('xpu')
    else:
        args.device = torch.device('cpu')

.. note::

    When assessing the availability of XPU in a given environment (:meth:`~torch.xpu.is_available`), PyTorch's default
    behavior is to call the XPU Runtime API method. Because this call in turn initializes the
    XPU Driver API if it is not already initialized, subsequent forks of a process that has run
    :meth:`~torch.xpu.is_available` will fail with a XPU initialization error.

Now that we have ``args.device``, we can use it to create a Tensor on the
desired device.

::

    x = torch.empty((8, 42), device=args.device)
    net = Network().to(device=args.device)

This can be used in a number of cases to produce device agnostic code. Below
is an example when using a dataloader:

::

    xpu0 = torch.device('xpu:0')  # XPU GPU 0
    for i, x in enumerate(train_loader):
        x = x.to(xpu0)

When working with multiple GPUs on a system, you can use the
``ZE_AFFINITY_MASK `` environment flag to manage which GPUs are available to
PyTorch. As mentioned above, to manually control which GPU a tensor is created
on, the best practice is to use a :any:`torch.xpu.device` context manager.

::

    print("Outside device is 0")  # On device 0 (default in most scenarios)
    with torch.xpu.device(1):
        print("Inside device is 1")  # On device 1
    print("Outside device is still 0")  # On device 0

If you have a tensor and would like to create a new tensor of the same type on
the same device, then you can use a ``torch.Tensor.new_*`` method
(see :class:`torch.Tensor`).
Whilst the previously mentioned ``torch.*`` factory functions
(:ref:`tensor-creation-ops`) depend on the current GPU context and
the attributes arguments you pass in, ``torch.Tensor.new_*`` methods preserve
the device and other attributes of the tensor.

This is the recommended practice when creating modules in which new
tensors need to be created internally during the forward pass.

::

    xpu = torch.device('xpu')
    x_cpu = torch.empty(2)
    x_gpu = torch.empty(2, device=xpu)
    x_cpu_long = torch.empty(2, dtype=torch.int64)

    y_cpu = x_cpu.new_full([3, 2], fill_value=0.3)
    print(y_cpu)

        tensor([[ 0.3000,  0.3000],
                [ 0.3000,  0.3000],
                [ 0.3000,  0.3000]])

    y_gpu = x_gpu.new_full([3, 2], fill_value=-5)
    print(y_gpu)

        tensor([[-5.0000, -5.0000],
                [-5.0000, -5.0000],
                [-5.0000, -5.0000]], device='xpu:0')

    y_cpu_long = x_cpu_long.new_tensor([[1, 2, 3]])
    print(y_cpu_long)

        tensor([[ 1,  2,  3]])


If you want to create a tensor of the same type and size of another tensor, and
fill it with either ones or zeros, :meth:`~torch.ones_like` or
:meth:`~torch.zeros_like` are provided as convenient helper functions (which
also preserve :class:`torch.device` and :class:`torch.dtype` of a Tensor).

::

    x_cpu = torch.empty(2, 3)
    x_gpu = torch.empty(2, 3)

    y_cpu = torch.ones_like(x_cpu)
    y_gpu = torch.zeros_like(x_gpu)


.. _xpu-memory-pinning:

Use pinned memory buffers
^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

    This is an advanced tip. If you overuse pinned memory, it can cause serious
    problems when running low on RAM, and you should be aware that pinning is
    often an expensive operation.

Host to GPU copies are much faster when they originate from pinned (page-locked)
memory. CPU tensors and storages expose a :meth:`~torch.Tensor.pin_memory`
method, that returns a copy of the object, with data put in a pinned region.

Also, once you pin a tensor or storage, you can use asynchronous GPU copies.
Just pass an additional ``non_blocking=True`` argument to a
:meth:`~torch.Tensor.to` or a :meth:`~torch.Tensor.xpu` call. This can be used
to overlap data transfers with computation.

You can make the :class:`~torch.utils.data.DataLoader` return batches placed in
pinned memory by passing ``pin_memory=True`` to its constructor.

.. _xpu-nn-ddp-instead:

Use nn.parallel.DistributedDataParallel instead of multiprocessing or nn.DataParallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most use cases involving batched inputs and multiple GPUs should default to
using :class:`~torch.nn.parallel.DistributedDataParallel` to utilize more
than one GPU.

There are significant caveats to using XPU models with
:mod:`~torch.multiprocessing`; unless care is taken to meet the data handling
requirements exactly, it is likely that your program will have incorrect or
undefined behavior.

It is recommended to use :class:`~torch.nn.parallel.DistributedDataParallel`,
instead of :class:`~torch.nn.DataParallel` to do multi-GPU training, even if
there is only a single node.

The difference between :class:`~torch.nn.parallel.DistributedDataParallel` and
:class:`~torch.nn.DataParallel` is: :class:`~torch.nn.parallel.DistributedDataParallel`
uses multiprocessing where a process is created for each GPU, while
:class:`~torch.nn.DataParallel` uses multithreading. By using multiprocessing,
each GPU has its dedicated process, this avoids the performance overhead caused
by GIL of Python interpreter.

If you use :class:`~torch.nn.parallel.DistributedDataParallel`, you could use
`torch.distributed.launch` utility to launch your program, see :ref:`distributed-launch`.
