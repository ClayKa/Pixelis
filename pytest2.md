pytest tests/engine/test_async_communication.py
============================================== test session starts ==============================================
platform linux -- Python 3.10.18, pytest-8.4.1, pluggy-1.6.0
rootdir: /mnt/c/Users/ClayKa/Pixelis
configfile: pyproject.toml
plugins: mock-3.14.1, hydra-core-1.3.2, cov-6.2.1, asyncio-1.1.0, anyio-4.10.0
asyncio: mode=strict, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 16 items                                                                                              

tests/engine/test_async_communication.py ..F..EFEE.EF.FF.
ERROR: Coverage failure: total of 12.05 is less than fail-under=70.00
                                                                                                          [100%]

==================================================== ERRORS =====================================================
_________________ ERROR at setup of TestAsyncCommunication.test_inference_engine_initialization _________________

self = <tests.engine.test_async_communication.TestAsyncCommunication object at 0x76c94e383130>

    @pytest.fixture
    def mock_voting_module(self):
        """Create a mock voting module."""
        voting = MagicMock()
>       result = VotingResult(
            final_answer={"answer": "test", "trajectory": []},
            confidence=0.8,
            votes=[],
            weights=[]
        )
E       TypeError: VotingResult.__init__() got an unexpected keyword argument 'votes'

tests/engine/test_async_communication.py:170: TypeError
_____________________ ERROR at setup of TestAsyncCommunication.test_adaptive_learning_rate ______________________

self = <tests.engine.test_async_communication.TestAsyncCommunication object at 0x76c94e3833a0>

    @pytest.fixture
    def mock_voting_module(self):
        """Create a mock voting module."""
        voting = MagicMock()
>       result = VotingResult(
            final_answer={"answer": "test", "trajectory": []},
            confidence=0.8,
            votes=[],
            weights=[]
        )
E       TypeError: VotingResult.__init__() got an unexpected keyword argument 'votes'

tests/engine/test_async_communication.py:170: TypeError
____________________ ERROR at setup of TestAsyncCommunication.test_infer_and_adapt_workflow _____________________

self = <tests.engine.test_async_communication.TestAsyncCommunication object at 0x76c94e381ba0>

    @pytest.fixture
    def mock_voting_module(self):
        """Create a mock voting module."""
        voting = MagicMock()
>       result = VotingResult(
            final_answer={"answer": "test", "trajectory": []},
            confidence=0.8,
            votes=[],
            weights=[]
        )
E       TypeError: VotingResult.__init__() got an unexpected keyword argument 'votes'

tests/engine/test_async_communication.py:170: TypeError
_____________ ERROR at setup of TestAsyncCommunication.test_update_task_enqueue_with_shared_memory ______________

self = <tests.engine.test_async_communication.TestAsyncCommunication object at 0x76c94e3819c0>

    @pytest.fixture
    def mock_voting_module(self):
        """Create a mock voting module."""
        voting = MagicMock()
>       result = VotingResult(
            final_answer={"answer": "test", "trajectory": []},
            confidence=0.8,
            votes=[],
            weights=[]
        )
E       TypeError: VotingResult.__init__() got an unexpected keyword argument 'votes'

tests/engine/test_async_communication.py:170: TypeError
=================================================== FAILURES ====================================================
______________________________ TestSharedMemoryManager.test_cleanup_stale_segments ______________________________

self = <tests.engine.test_async_communication.TestSharedMemoryManager object at 0x76c94e382410>

    def test_cleanup_stale_segments(self):
        """Test cleaning up stale shared memory segments."""
        manager = SharedMemoryManager(timeout_seconds=0.1)  # Very short timeout
    
        # Create some segments
        tensor1 = torch.randn(10, 10)
        tensor2 = torch.randn(20, 20)
    
        shm_info1 = manager.create_shared_tensor(tensor1)
        time.sleep(0.05)
        shm_info2 = manager.create_shared_tensor(tensor2)
    
        # Wait for first segment to become stale
        time.sleep(0.1)
    
        # Clean up stale segments
        cleaned = manager.cleanup_stale_segments(worker_alive=True)
    
        # First segment should be cleaned
        assert shm_info1.name in cleaned
        assert shm_info1.name not in manager.pending_shm
    
        # Second segment should still be there
>       assert shm_info2.name not in cleaned
E       AssertionError: assert 'pixelis_shm_fe16423407f9400586c26671bf1971ec' not in ['pixelis_shm_12e07452dc1a46
3bafc964dcd87b8475', 'pixelis_shm_fe16423407f9400586c26671bf1971ec']                                             E        +  where 'pixelis_shm_fe16423407f9400586c26671bf1971ec' = SharedMemoryInfo(name='pixelis_shm_fe16423407f
9400586c26671bf1971ec', shape=(20, 20), dtype=torch.float32, created_at=datetime.datetime(2025, 8, 17, 14, 41, 12, 362342), size_bytes=1600).name                                                                                 
tests/engine/test_async_communication.py:94: AssertionError
--------------------------------------------- Captured stdout call ----------------------------------------------
2025-08-17 14:41:12 - core.engine.inference_engine - WARNING - [Watchdog] Segment pixelis_shm_12e07452dc1a463bafc
964dcd87b8475 exceeded timeout (0.2s > 0.1s), cleaning up                                                        2025-08-17 14:41:12 - core.engine.inference_engine - WARNING - [Watchdog] Segment pixelis_shm_fe16423407f9400586c
26671bf1971ec exceeded timeout (0.1s > 0.1s), cleaning up                                                        ----------------------------------------------- Captured log call -----------------------------------------------
WARNING  core.engine.inference_engine:inference_engine.py:177 [Watchdog] Segment pixelis_shm_12e07452dc1a463bafc9
64dcd87b8475 exceeded timeout (0.2s > 0.1s), cleaning up                                                         WARNING  core.engine.inference_engine:inference_engine.py:177 [Watchdog] Segment pixelis_shm_fe16423407f9400586c2
6671bf1971ec exceeded timeout (0.1s > 0.1s), cleaning up                                                         ___________________________ TestAsyncCommunication.test_update_worker_initialization ____________________________

self = <tests.engine.test_async_communication.TestAsyncCommunication object at 0x76c94e382ce0>
mock_model = <MagicMock id='130606223951136'>

    def test_update_worker_initialization(self, mock_model):
        """Test initializing the update worker."""
        config = {
            'kl_weight': 0.01,
            'max_kl': 0.05,
            'grad_clip_norm': 1.0,
            'ema_decay': 0.999,
            'base_learning_rate': 1e-5,
            'weight_decay': 0.01
        }
    
        update_queue = mp.Queue()
        cleanup_queue = mp.Queue()
    
        worker = UpdateWorker(
            model=mock_model,
            update_queue=update_queue,
            cleanup_confirmation_queue=cleanup_queue,
            config=config
        )
    
        # Check initialization
        assert worker.model == mock_model
>       assert worker.kl_weight == 0.01
E       AttributeError: 'UpdateWorker' object has no attribute 'kl_weight'

tests/engine/test_async_communication.py:252: AttributeError
--------------------------------------------- Captured stdout call ----------------------------------------------
2025-08-17 14:41:12 - core.engine.update_worker - WARNING - No trainable parameters found!
2025-08-17 14:41:12 - core.modules.audit - INFO - Audit logger initialized at checkpoints/online_updates/audit
2025-08-17 14:41:12 - core.engine.update_worker - INFO - Update worker initialized (PID: 3513) with KL config: KL
Config(beta_update_mode='auto', initial_beta=0.01, target_kl=0.05, kl_tolerance=0.01, beta_increase_factor=1.2, beta_decrease_factor=1.2, min_beta=0.0001, max_beta=1.0)                                                          ----------------------------------------------- Captured log call -----------------------------------------------
WARNING  core.engine.update_worker:update_worker.py:253 No trainable parameters found!
INFO     core.modules.audit:audit.py:198 Audit logger initialized at checkpoints/online_updates/audit
INFO     core.engine.update_worker:update_worker.py:210 Update worker initialized (PID: 3513) with KL config: KLC
onfig(beta_update_mode='auto', initial_beta=0.01, target_kl=0.05, kl_tolerance=0.01, beta_increase_factor=1.2, beta_decrease_factor=1.2, min_beta=0.0001, max_beta=1.0)                                                           ______________________________ TestFaultTolerance.test_watchdog_cleanup_on_timeout ______________________________

self = <tests.engine.test_async_communication.TestFaultTolerance object at 0x76c94e381660>

    def test_watchdog_cleanup_on_timeout(self):
        """Test watchdog cleaning up segments after timeout."""
        config = {
            'shm_timeout': 0.1,  # Very short timeout
            'watchdog_interval': 0.05
        }
    
        # Create mock components
        model = MagicMock()
        experience_buffer = MagicMock()
        voting_module = MagicMock()
        reward_orchestrator = MagicMock()
    
        engine = InferenceEngine(
            model=model,
            experience_buffer=experience_buffer,
            voting_module=voting_module,
            reward_orchestrator=reward_orchestrator,
            config=config
        )
    
        # Create a shared memory segment
        tensor = torch.randn(10, 10)
        shm_info = engine.shm_manager.create_shared_tensor(tensor)
    
        # Start watchdog
        engine.start_watchdog()
    
        # Wait for timeout and watchdog cleanup
        time.sleep(0.3)
    
        # Check that segment was cleaned
>       assert shm_info.name not in engine.shm_manager.pending_shm
E       AssertionError: assert 'pixelis_shm_461101d8111d496c94cf2a2a460ac89d' not in {'pixelis_shm_461101d8111d49
6c94cf2a2a460ac89d': SharedMemoryInfo(name='pixelis_shm_461101d8111d496c94cf2a2a460ac89d', shape=(10, 10), dtype=torch.float32, created_at=datetime.datetime(2025, 8, 17, 14, 41, 11, 750474), size_bytes=400)}                   E        +  where 'pixelis_shm_461101d8111d496c94cf2a2a460ac89d' = SharedMemoryInfo(name='pixelis_shm_461101d8111
d496c94cf2a2a460ac89d', shape=(10, 10), dtype=torch.float32, created_at=datetime.datetime(2025, 8, 17, 14, 41, 11, 750474), size_bytes=400).name                                                                                  E        +  and   {'pixelis_shm_461101d8111d496c94cf2a2a460ac89d': SharedMemoryInfo(name='pixelis_shm_461101d8111
d496c94cf2a2a460ac89d', shape=(10, 10), dtype=torch.float32, created_at=datetime.datetime(2025, 8, 17, 14, 41, 11, 750474), size_bytes=400)} = <core.engine.inference_engine.SharedMemoryManager object at 0x76c948f75ab0>.pending_shm                                                                                                             E        +    where <core.engine.inference_engine.SharedMemoryManager object at 0x76c948f75ab0> = <core.engine.in
ference_engine.InferenceEngine object at 0x76c948f74ac0>.shm_manager                                             
tests/engine/test_async_communication.py:451: AssertionError
--------------------------------------------- Captured stdout call ----------------------------------------------
2025-08-17 14:41:11 - core.modules.alerter - INFO - Alerter initialized with channels: ['log']
2025-08-17 14:41:11 - core.modules.privacy - INFO - PII Redactor initialized with 15 patterns
2025-08-17 14:41:11 - core.modules.privacy - INFO - Image Metadata Stripper initialized
2025-08-17 14:41:11 - core.modules.privacy - INFO - Data Anonymizer initialized
2025-08-17 14:41:11 - core.engine.inference_engine - INFO - Inference Engine initialized with monitoring, alertin
g, and privacy protection                                                                                        2025-08-17 14:41:11 - core.engine.inference_engine - INFO - Started watchdog thread
2025-08-17 14:41:11 - core.engine.inference_engine - INFO - Started monitoring thread
2025-08-17 14:41:11 - core.modules.alerter - WARNING - [ALERT] inference_engine: Model update rate dropped to zer
o: update_rate=0.000                                                                                             ----------------------------------------------- Captured log call -----------------------------------------------
INFO     core.modules.alerter:alerter.py:132 Alerter initialized with channels: ['log']
INFO     core.modules.privacy:privacy.py:79 PII Redactor initialized with 15 patterns
INFO     core.modules.privacy:privacy.py:410 Image Metadata Stripper initialized
INFO     core.modules.privacy:privacy.py:564 Data Anonymizer initialized
INFO     core.engine.inference_engine:inference_engine.py:333 Inference Engine initialized with monitoring, alert
ing, and privacy protection                                                                                      INFO     core.engine.inference_engine:inference_engine.py:799 Started watchdog thread
INFO     core.engine.inference_engine:inference_engine.py:1446 Started monitoring thread
WARNING  core.modules.alerter:alerter.py:297 [ALERT] inference_engine: Model update rate dropped to zero: update_
rate=0.000                                                                                                       ___________________________________ TestFaultTolerance.test_graceful_shutdown ___________________________________

self = <tests.engine.test_async_communication.TestFaultTolerance object at 0x76c94e381120>

    def test_graceful_shutdown(self):
        """Test graceful shutdown of the system."""
        config = {}
    
        # Create components
        model = MagicMock()
        model.parameters.return_value = []
        model.named_parameters.return_value = []
    
        experience_buffer = MagicMock()
        voting_module = MagicMock()
        reward_orchestrator = MagicMock()
    
        engine = InferenceEngine(
            model=model,
            experience_buffer=experience_buffer,
            voting_module=voting_module,
            reward_orchestrator=reward_orchestrator,
            config=config
        )
    
        # Create some shared memory segments
        tensors = [torch.randn(5, 5) for _ in range(2)]
        shm_infos = [engine.shm_manager.create_shared_tensor(t) for t in tensors]
    
        # Mock worker process
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        engine.update_worker_process = mock_process
    
        # Start watchdog
        engine.watchdog_running = True
        engine.watchdog_thread = MagicMock()
    
        # Shutdown
        engine.shutdown()
    
        # Check shutdown actions
        assert engine.watchdog_running == False
>       engine.update_queue.put.assert_called_with(None)  # Shutdown signal
E       AttributeError: 'function' object has no attribute 'assert_called_with'

tests/engine/test_async_communication.py:543: AttributeError
--------------------------------------------- Captured stdout call ----------------------------------------------
2025-08-17 14:41:12 - core.modules.alerter - INFO - Alerter initialized with channels: ['log']
2025-08-17 14:41:12 - core.modules.privacy - INFO - PII Redactor initialized with 15 patterns
2025-08-17 14:41:12 - core.modules.privacy - INFO - Image Metadata Stripper initialized
2025-08-17 14:41:12 - core.modules.privacy - INFO - Data Anonymizer initialized
2025-08-17 14:41:12 - core.engine.inference_engine - INFO - Inference Engine initialized with monitoring, alertin
g, and privacy protection                                                                                        2025-08-17 14:41:12 - core.engine.inference_engine - INFO - Shutting down Inference Engine
2025-08-17 14:41:12 - core.engine.inference_engine - WARNING - Update worker didn't stop gracefully, terminating
2025-08-17 14:41:12 - core.engine.inference_engine - WARNING - [Watchdog] Worker dead, cleaning up segment: pixel
is_shm_5e48094aaf8843038a0e33de2a8668d2                                                                          2025-08-17 14:41:12 - core.engine.inference_engine - WARNING - [Watchdog] Worker dead, cleaning up segment: pixel
is_shm_f5c0a42d6a954169b986b00fb8190e49                                                                          2025-08-17 14:41:12 - core.engine.inference_engine - INFO - Final cleanup: 2 segments
2025-08-17 14:41:12 - core.engine.inference_engine - INFO - InferenceEngine Status - Requests: 0, Updates: 0, Fai
led: 0, Watchdog cleanups: 0, Pending SHM: 0, SHM bytes: 0                                                       2025-08-17 14:41:12 - core.engine.inference_engine - INFO - Inference Engine shutdown complete
----------------------------------------------- Captured log call -----------------------------------------------
INFO     core.modules.alerter:alerter.py:132 Alerter initialized with channels: ['log']
INFO     core.modules.privacy:privacy.py:79 PII Redactor initialized with 15 patterns
INFO     core.modules.privacy:privacy.py:410 Image Metadata Stripper initialized
INFO     core.modules.privacy:privacy.py:564 Data Anonymizer initialized
INFO     core.engine.inference_engine:inference_engine.py:333 Inference Engine initialized with monitoring, alert
ing, and privacy protection                                                                                      INFO     core.engine.inference_engine:inference_engine.py:1045 Shutting down Inference Engine
WARNING  core.engine.inference_engine:inference_engine.py:1060 Update worker didn't stop gracefully, terminating
WARNING  core.engine.inference_engine:inference_engine.py:175 [Watchdog] Worker dead, cleaning up segment: pixeli
s_shm_5e48094aaf8843038a0e33de2a8668d2                                                                           WARNING  core.engine.inference_engine:inference_engine.py:175 [Watchdog] Worker dead, cleaning up segment: pixeli
s_shm_f5c0a42d6a954169b986b00fb8190e49                                                                           INFO     core.engine.inference_engine:inference_engine.py:1067 Final cleanup: 2 segments
INFO     core.engine.inference_engine:inference_engine.py:967 InferenceEngine Status - Requests: 0, Updates: 0, F
ailed: 0, Watchdog cleanups: 0, Pending SHM: 0, SHM bytes: 0                                                     INFO     core.engine.inference_engine:inference_engine.py:1072 Inference Engine shutdown complete
________________________________ TestFaultTolerance.test_queue_timeout_handling _________________________________

    def single_iteration_run():
        try:
            # This should timeout
>           task = worker.update_queue.get(timeout=0.1)

tests/engine/test_async_communication.py:573: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <multiprocessing.queues.Queue object at 0x76c948b03fd0>, block = True, timeout = 0.09998669399999471

    def get(self, block=True, timeout=None):
        if self._closed:
            raise ValueError(f"Queue {self!r} is closed")
        if block and timeout is None:
            with self._rlock:
                res = self._recv_bytes()
            self._sem.release()
        else:
            if block:
                deadline = time.monotonic() + timeout
            if not self._rlock.acquire(block, timeout):
                raise Empty
            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._poll(timeout):
>                       raise Empty
E                       _queue.Empty

/home/clayka/miniconda3/envs/pixelis/lib/python3.10/multiprocessing/queues.py:114: Empty

During handling of the above exception, another exception occurred:

self = <tests.engine.test_async_communication.TestFaultTolerance object at 0x76c94e380e80>

    def test_queue_timeout_handling(self):
        """Test handling of queue timeouts."""
        config = {}
    
        update_queue = mp.Queue()
        cleanup_queue = mp.Queue()
    
        model = MagicMock()
        model.parameters.return_value = []
        model.named_parameters.return_value = []
    
        worker = UpdateWorker(
            model=model,
            update_queue=update_queue,
            cleanup_confirmation_queue=cleanup_queue,
            config=config
        )
    
        # Override run to test single iteration
        original_run = worker.run
    
        def single_iteration_run():
            try:
                # This should timeout
                task = worker.update_queue.get(timeout=0.1)
                if task is None:
                    return
                worker._process_update(task)
            except mp.queues.Empty:
                # Expected timeout
                pass
            except Exception as e:
                pytest.fail(f"Unexpected exception: {e}")
    
        # Test that timeout is handled gracefully
>       single_iteration_run()  # Should not raise

tests/engine/test_async_communication.py:584: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    def single_iteration_run():
        try:
            # This should timeout
            task = worker.update_queue.get(timeout=0.1)
            if task is None:
                return
            worker._process_update(task)
>       except mp.queues.Empty:
E       AttributeError: module 'torch.multiprocessing' has no attribute 'queues'. Did you mean: 'Queue'?

tests/engine/test_async_communication.py:577: AttributeError
--------------------------------------------- Captured stdout call ----------------------------------------------
2025-08-17 14:41:12 - core.engine.update_worker - WARNING - No trainable parameters found!
2025-08-17 14:41:12 - core.modules.audit - INFO - Audit logger initialized at checkpoints/online_updates/audit
2025-08-17 14:41:12 - core.engine.update_worker - INFO - Update worker initialized (PID: 3513) with KL config: KL
Config(beta_update_mode='auto', initial_beta=0.01, target_kl=0.05, kl_tolerance=0.01, beta_increase_factor=1.2, beta_decrease_factor=1.2, min_beta=0.0001, max_beta=1.0)                                                          ----------------------------------------------- Captured log call -----------------------------------------------
WARNING  core.engine.update_worker:update_worker.py:253 No trainable parameters found!
INFO     core.modules.audit:audit.py:198 Audit logger initialized at checkpoints/online_updates/audit
INFO     core.engine.update_worker:update_worker.py:210 Update worker initialized (PID: 3513) with KL config: KLC
onfig(beta_update_mode='auto', initial_beta=0.01, target_kl=0.05, kl_tolerance=0.01, beta_increase_factor=1.2, beta_decrease_factor=1.2, min_beta=0.0001, max_beta=1.0)                                                           =============================================== warnings summary ================================================
tests/engine/test_async_communication.py::TestSharedMemoryManager::test_create_shared_tensor
  /mnt/c/Users/ClayKa/Pixelis/core/engine/inference_engine.py:88: UserWarning: TypedStorage is deprecated. It wil
l be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()                                                                                                          storage = tensor.storage()._share_memory_()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================================ tests coverage =================================================
_______________________________ coverage: platform linux, python 3.10.18-final-0 ________________________________

Name                                         Stmts   Miss Branch BrPart   Cover   Missing
-----------------------------------------------------------------------------------------
core/engine/inference_engine.py                581    405    174     13  26.09%   81, 130-132, 142->147, 147->exi
t, 170->166, 197->exit, 202-203, 331, 355-509, 530, 547-560, 584-597, 614-717, 723-751, 765-783, 790-791, 811-942, 953-954, 958-960, 982-1005, 1014-1039, 1049->1053, 1053->1057, 1057->1065, 1059->1065, 1066->1070, 1085-1101, 1119-1199, 1218-1228, 1245-1281, 1297-1346, 1355-1389, 1398-1430, 1437-1438, 1452-1528, 1538-1558, 1567-1570      core/engine/update_worker.py                   299    189     68      7  31.88%   55, 58, 61, 135, 169, 214-216, 
234, 238-240, 256-282, 307-335, 349-500, 527-546, 555-563, 575-583, 593-654, 663-706, 715-729, 755-839, 843, 847, 860-906                                                                                                         core/models/peft_model.py                        3      3      0      0   0.00%   3-19
core/modules/alerter.py                        181    100     56      2  35.02%   43, 65-77, 193, 196->199, 209-2
16, 244-272, 285-290, 294-299, 303-332, 336-340, 344-350, 362-364, 368-369, 373-376, 421-433, 437-439, 443-450, 455-463, 470-471, 480                                                                                             core/modules/audit.py                          246    169     70      2  25.00%   86-99, 103, 118, 188, 210-220, 
245-279, 288-301, 305-319, 323-348, 365-424, 451-512, 516, 527-548, 562-570, 589-597, 602, 627-631               core/modules/dynamics_model.py                 160    160     40      0   0.00%   8-560
core/modules/experience_buffer.py              341    341    136      0   0.00%   9-782
core/modules/experience_buffer_enhanced.py     396    396    168      0   0.00%   13-885
core/modules/operation_registry.py              74     74     18      0   0.00%   8-298
core/modules/operations/base_operation.py       21     21      0      0   0.00%   8-113
core/modules/operations/get_properties.py      137    137     52      0   0.00%   7-438
core/modules/operations/read_text.py           111    111     48      0   0.00%   7-276
core/modules/operations/segment_object.py       97     97     34      0   0.00%   7-268
core/modules/operations/track_object.py        181    181     50      0   0.00%   7-519
core/modules/operations/zoom_in.py             179    179     68      0   0.00%   7-457
core/modules/persistence_adapter.py            276    276     50      0   0.00%   8-498
core/modules/privacy.py                        220    156     88      0  21.43%   236-283, 295-304, 316-351, 361-
367, 371, 379-381, 423-480, 492-539, 543, 576-609, 621-653, 661                                                  core/modules/reward_shaping.py                 223    193     84      0   9.77%   48-84, 104-137, 154-179, 193-20
3, 228-230, 247-269, 281-307, 322-353, 365-397, 415-454, 476-529, 548-558, 570-595, 604-625, 635-647, 660-670, 674, 678-679                                                                                                       core/modules/reward_shaping_enhanced.py        301    301     98      0   0.00%   8-814
core/modules/voting.py                         132    132     42      0   0.00%   8-489
core/reproducibility/artifact_manager.py       350    241    124      7  24.47%   19-20, 25-26, 68-73, 93-98, 101
, 105-110, 114-120, 128, 133, 138, 142-151, 155-165, 178->182, 180->182, 185->exit, 201->212, 204-208, 220-221, 234-272, 284-346, 357-382, 401-426, 430-442, 446-467, 471-475, 480-499, 503-517, 521-548, 552-564, 568-592, 601-618, 627-637, 646-675, 679                                                                                         core/reproducibility/config_capture.py         251    207     92      0  12.83%   18-19, 24-25, 29, 61-110, 115-1
81, 186-242, 247-263, 268-317, 322-369, 374-424, 430-477, 496-547                                                core/reproducibility/decorators.py             200    180     82      0   7.09%   45-201, 227-286, 310-436, 451-5
25                                                                                                               core/reproducibility/experiment_context.py     185    144     52      0  17.30%   19-20, 25-26, 31-32, 45-54, 58-
68, 72-104, 108-155, 186-203, 207-248, 252-297, 301-303, 312, 326-335, 344-366, 392-404, 421-444, 455-478, 488-501                                                                                                                core/reproducibility/lineage_tracker.py        234    197    108      0  10.82%   29-30, 41-51, 79-110, 127-147, 
164-184, 201-224, 236-264, 277-303, 312-372, 384-412, 421-434, 443-460, 469-488, 492-504, 508-518, 522-530, 534-541, 545-556, 560-578                                                                                             core/utils/logging_utils.py                     95     54     26      3  41.32%   37, 40, 59-65, 101-107, 117-126
, 147-157, 164-169, 176-191, 201-224                                                                             -----------------------------------------------------------------------------------------
TOTAL                                         5474   4644   1828     34  12.05%
Coverage HTML written to dir htmlcov
Coverage XML written to file coverage.xml
FAIL Required test coverage of 70% not reached. Total coverage: 12.05%
============================================ short test summary info ============================================
ERROR tests/engine/test_async_communication.py::TestAsyncCommunication::test_inference_engine_initialization - Ty
peError: VotingResult.__init__() got an unexpected keyword argument 'votes'                                      ERROR tests/engine/test_async_communication.py::TestAsyncCommunication::test_adaptive_learning_rate - TypeError: 
VotingResult.__init__() got an unexpected keyword argument 'votes'                                               ERROR tests/engine/test_async_communication.py::TestAsyncCommunication::test_infer_and_adapt_workflow - TypeError
: VotingResult.__init__() got an unexpected keyword argument 'votes'                                             ERROR tests/engine/test_async_communication.py::TestAsyncCommunication::test_update_task_enqueue_with_shared_memo
ry - TypeError: VotingResult.__init__() got an unexpected keyword argument 'votes'                               FAILED tests/engine/test_async_communication.py::TestSharedMemoryManager::test_cleanup_stale_segments - Assertion
Error: assert 'pixelis_shm_fe16423407f9400586c26671bf1971ec' not in ['pixelis_shm_12e07452dc1a463ba...           FAILED tests/engine/test_async_communication.py::TestAsyncCommunication::test_update_worker_initialization - Attr
ibuteError: 'UpdateWorker' object has no attribute 'kl_weight'                                                   FAILED tests/engine/test_async_communication.py::TestFaultTolerance::test_watchdog_cleanup_on_timeout - Assertion
Error: assert 'pixelis_shm_461101d8111d496c94cf2a2a460ac89d' not in {'pixelis_shm_461101d8111d496c9...           FAILED tests/engine/test_async_communication.py::TestFaultTolerance::test_graceful_shutdown - AttributeError: 'fu
nction' object has no attribute 'assert_called_with'                                                             FAILED tests/engine/test_async_communication.py::TestFaultTolerance::test_queue_timeout_handling - AttributeError
: module 'torch.multiprocessing' has no attribute 'queues'. Did you mean: 'Queue'?                               =============================== 5 failed, 7 passed, 1 warning, 4 errors in 17.99s ===============================
