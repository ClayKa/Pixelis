pytest -v tests/unit/test_artifact_manager.py
================================================================================== test session starts ==================================================================================
platform linux -- Python 3.10.18, pytest-8.4.1, pluggy-1.6.0 -- /home/clayka/miniconda3/envs/pixelis/bin/python3.10
cachedir: .pytest_cache
rootdir: /mnt/c/Users/ClayKa/Pixelis
configfile: pyproject.toml
plugins: mock-3.14.1, hydra-core-1.3.2, cov-6.2.1, asyncio-1.1.0, anyio-4.10.0
asyncio: mode=strict, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 20 items                                                                                                                                                                      

tests/unit/test_artifact_manager.py::TestArtifactManager::test_singleton_pattern PASSED                                                                                           [  5%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_init_run PASSED                                                                                                    [ 10%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_log_artifact FAILED                                                                                                [ 15%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_log_large_artifact FAILED                                                                                          [ 20%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_use_artifact FAILED                                                                                                [ 25%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_artifact_versioning FAILED                                                                                         [ 30%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_artifact_lineage FAILED                                                                                            [ 35%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_content_addressable_storage FAILED                                                                                 [ 40%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_wandb_integration FAILED                                                                                           [ 45%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_list_artifacts FAILED                                                                                              [ 50%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_thread_safety PASSED                                                                                               [ 55%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.DATASET] FAILED                                                                    [ 60%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.MODEL] FAILED                                                                      [ 65%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.CHECKPOINT] FAILED                                                                 [ 70%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.CONFIG] FAILED                                                                     [ 75%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.METRICS] FAILED                                                                    [ 80%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.EVALUATION] FAILED                                                                 [ 85%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.ENVIRONMENT] FAILED                                                                [ 90%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.CODE] FAILED                                                                       [ 95%]
tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.EXPERIENCE] FAILED                                                                 [100%]
ERROR: Coverage failure: total of 5.40 is less than fail-under=70.00


======================================================================================= FAILURES ========================================================================================
_________________________________________________________________________ TestArtifactManager.test_log_artifact _________________________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cae140>

    def test_log_artifact(self):
        """Test artifact logging."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        artifact_data = {"key": "value"}
        artifact = manager.log_artifact(
            name="test_artifact",
            type=ArtifactType.CONFIG,
            data=artifact_data,
            metadata={"description": "Test artifact"}
        )
    
        assert artifact is not None
>       assert artifact["name"] == "test_artifact"
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:61: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:23 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:23 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133223
2025-08-17 13:32:23 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_artifact vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133223
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_artifact vv1
______________________________________________________________________ TestArtifactManager.test_log_large_artifact ______________________________________________________________________

self = <unittest.mock._patch object at 0x786415d599f0>

    def __enter__(self):
        """Perform the patch."""
        new, spec, spec_set = self.new, self.spec, self.spec_set
        autospec, kwargs = self.autospec, self.kwargs
        new_callable = self.new_callable
        self.target = self.getter()
    
        # normalise False to None
        if spec is False:
            spec = None
        if spec_set is False:
            spec_set = None
        if autospec is False:
            autospec = None
    
        if spec is not None and autospec is not None:
            raise TypeError("Can't specify spec and autospec")
        if ((spec is not None or autospec is not None) and
            spec_set not in (True, None)):
            raise TypeError("Can't provide explicit spec_set *and* spec or autospec")
    
        original, local = self.get_original()
    
        if new is DEFAULT and autospec is None:
            inherit = False
            if spec is True:
                # set spec to the object we are replacing
                spec = original
                if spec_set is True:
                    spec_set = original
                    spec = None
            elif spec is not None:
                if spec_set is True:
                    spec_set = spec
                    spec = None
            elif spec_set is True:
                spec_set = original
    
            if spec is not None or spec_set is not None:
                if original is DEFAULT:
                    raise TypeError("Can't use 'spec' with create=True")
                if isinstance(original, type):
                    # If we're patching out a class and there is a spec
                    inherit = True
            if spec is None and _is_async_obj(original):
                Klass = AsyncMock
            else:
                Klass = MagicMock
            _kwargs = {}
            if new_callable is not None:
                Klass = new_callable
            elif spec is not None or spec_set is not None:
                this_spec = spec
                if spec_set is not None:
                    this_spec = spec_set
                if _is_list(this_spec):
                    not_callable = '__call__' not in this_spec
                else:
                    not_callable = not callable(this_spec)
                if _is_async_obj(this_spec):
                    Klass = AsyncMock
                elif not_callable:
                    Klass = NonCallableMagicMock
    
            if spec is not None:
                _kwargs['spec'] = spec
            if spec_set is not None:
                _kwargs['spec_set'] = spec_set
    
            # add a name to mocks
            if (isinstance(Klass, type) and
                issubclass(Klass, NonCallableMock) and self.attribute):
                _kwargs['name'] = self.attribute
    
            _kwargs.update(kwargs)
            new = Klass(**_kwargs)
    
            if inherit and _is_instance_mock(new):
                # we can only tell if the instance should be callable if the
                # spec is not a list
                this_spec = spec
                if spec_set is not None:
                    this_spec = spec_set
                if (not _is_list(this_spec) and not
                    _instance_callable(this_spec)):
                    Klass = NonCallableMagicMock
    
                _kwargs.pop('name')
                new.return_value = Klass(_new_parent=new, _new_name='()',
                                         **_kwargs)
        elif autospec is not None:
            # spec is ignored, new *must* be default, spec_set is treated
            # as a boolean. Should we check spec is not None and that spec_set
            # is a bool?
            if new is not DEFAULT:
                raise TypeError(
                    "autospec creates the mock for you. Can't specify "
                    "autospec and new."
                )
            if original is DEFAULT:
                raise TypeError("Can't use 'autospec' with create=True")
            spec_set = bool(spec_set)
            if autospec is True:
                autospec = original
    
            if _is_instance_mock(self.target):
                raise InvalidSpecError(
                    f'Cannot autospec attr {self.attribute!r} as the patch '
                    f'target has already been mocked out. '
                    f'[target={self.target!r}, attr={autospec!r}]')
            if _is_instance_mock(autospec):
                target_name = getattr(self.target, '__name__', self.target)
                raise InvalidSpecError(
                    f'Cannot autospec attr {self.attribute!r} from target '
                    f'{target_name!r} as it has already been mocked out. '
                    f'[target={self.target!r}, attr={autospec!r}]')
    
            new = create_autospec(autospec, spec_set=spec_set,
                                  _name=self.attribute, **kwargs)
        elif kwargs:
            # can't set keyword args when we aren't creating the mock
            # XXXX If new is a Mock we could call new.configure_mock(**kwargs)
            raise TypeError("Can't pass kwargs to a mock we aren't creating")
    
        new_attr = new
    
        self.temp_original = original
        self.is_local = local
        self._exit_stack = contextlib.ExitStack()
        try:
>           setattr(self.target, self.attribute, new_attr)
E           AttributeError: 'PosixPath' object attribute 'stat' is read-only

/home/clayka/miniconda3/envs/pixelis/lib/python3.10/unittest/mock.py:1556: AttributeError

During handling of the above exception, another exception occurred:

self = <test_artifact_manager.TestArtifactManager object at 0x786440cae350>, mocker = <pytest_mock.plugin.MockerFixture object at 0x786415d9ffd0>
tmp_path = PosixPath('/tmp/pytest-of-clayka/pytest-3/test_log_large_artifact0')

    def test_log_large_artifact(self, mocker, tmp_path): # <-- Add tmp_path fixture
        """
        Test that logging a large file artifact works correctly
        WITHOUT performing real, slow I/O operations.
        """
        manager = ArtifactManager()
    
        # Mock the slow I/O methods
        mock_hash_compute = mocker.patch.object(manager, '_compute_file_hash', return_value='mock_hash_123')
        mock_storage_upload = mocker.patch.object(manager.storage_backend, 'upload')
    
        # 1. [THE FIX] Create a REAL, EMPTY temporary file.
        #    tmp_path is a pytest fixture that provides a temporary Path object.
        large_file = tmp_path / "large_dataset.bin"
        large_file.touch() # Create the empty file
    
        # We can still mock its stat() method if we need to simulate a large size
        # without actually writing data.
>       mocker.patch.object(large_file, 'stat', return_value=MagicMock(st_size=1 * 1024**3))

tests/unit/test_artifact_manager.py:83: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/pytest_mock/plugin.py:288: in object
    return self._start_patch(
/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/pytest_mock/plugin.py:257: in _start_patch
    mocked: MockType = p.start()
/home/clayka/miniconda3/envs/pixelis/lib/python3.10/unittest/mock.py:1595: in start
    result = self.__enter__()
/home/clayka/miniconda3/envs/pixelis/lib/python3.10/unittest/mock.py:1569: in __enter__
    if not self.__exit__(*sys.exc_info()):
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <unittest.mock._patch object at 0x786415d599f0>
exc_info = (<class 'AttributeError'>, AttributeError("'PosixPath' object attribute 'stat' is read-only"), <traceback object at 0x786415d3c280>)

    def __exit__(self, *exc_info):
        """Undo the patch."""
        if self.is_local and self.temp_original is not DEFAULT:
            setattr(self.target, self.attribute, self.temp_original)
        else:
>           delattr(self.target, self.attribute)
E           AttributeError: 'PosixPath' object attribute 'stat' is read-only

/home/clayka/miniconda3/envs/pixelis/lib/python3.10/unittest/mock.py:1577: AttributeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:23 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
_________________________________________________________________________ TestArtifactManager.test_use_artifact _________________________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cae620>

    def test_use_artifact(self):
        """Test artifact retrieval."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        # First log an artifact
        artifact_data = {"data": "test"}
        logged_artifact = manager.log_artifact(
            name="test_artifact",
            type=ArtifactType.DATASET,
            data=artifact_data
        )
    
        # Then retrieve it
        retrieved = manager.use_artifact(
            name="test_artifact",
            version="v1"
        )
    
        assert retrieved is not None
>       assert retrieved["name"] == "test_artifact"
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:121: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:23 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:23 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133223
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_artifact vv1
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using cached artifact: test_artifact:v1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133223
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_artifact vv1
INFO     core.reproducibility.artifact_manager:artifact_manager.py:415 Using cached artifact: test_artifact:v1
_____________________________________________________________________ TestArtifactManager.test_artifact_versioning ______________________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440caeb60>

    def test_artifact_versioning(self):
        """Test that artifacts get versioned correctly."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        # Log same artifact multiple times
        for i in range(3):
            artifact = manager.log_artifact(
                name="versioned_artifact",
                type=ArtifactType.METRICS,
                data={"iteration": i}
            )
>           assert artifact["version"] == f"v{i+1}"
E           TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:136: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: versioned_artifact vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: versioned_artifact vv1
_______________________________________________________________________ TestArtifactManager.test_artifact_lineage _______________________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440caee30>

    def test_artifact_lineage(self):
        """Test artifact lineage tracking."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        # Create parent artifact
        parent = manager.log_artifact(
            name="parent_artifact",
            type=ArtifactType.DATASET,
            data={"parent": True}
        )
    
        # Create child artifact with lineage
        child = manager.log_artifact(
            name="child_artifact",
            type=ArtifactType.MODEL,
            data={"child": True},
>           parent_artifacts=[f"{parent['name']}:{parent['version']}"]
        )
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:155: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: parent_artifact vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: parent_artifact vv1
_________________________________________________________________ TestArtifactManager.test_content_addressable_storage __________________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440caf400>

    def test_content_addressable_storage(self):
        """Test that identical content produces same hash."""
        manager = ArtifactManager()
    
        content1 = {"data": "test", "value": 123}
        content2 = {"data": "test", "value": 123}
        content3 = {"data": "different", "value": 456}
    
>       hash1 = manager._compute_content_hash(content1)
E       AttributeError: 'ArtifactManager' object has no attribute '_compute_content_hash'. Did you mean: '_compute_file_hash'?

tests/unit/test_artifact_manager.py:169: AttributeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
______________________________________________________________________ TestArtifactManager.test_wandb_integration _______________________________________________________________________

self = <MagicMock name='init' id='132371258384848'>

    def assert_called_once(self):
        """assert that the mock was called only once.
        """
        if not self.call_count == 1:
            msg = ("Expected '%s' to have been called once. Called %s times.%s"
                   % (self._mock_name or 'mock',
                      self.call_count,
                      self._calls_repr()))
>           raise AssertionError(msg)
E           AssertionError: Expected 'init' to have been called once. Called 0 times.

/home/clayka/miniconda3/envs/pixelis/lib/python3.10/unittest/mock.py:908: AssertionError

During handling of the above exception, another exception occurred:

self = <test_artifact_manager.TestArtifactManager object at 0x786440caded0>, mock_artifact_class = <MagicMock name='Artifact' id='132371258878432'>
mock_init = <MagicMock name='init' id='132371258384848'>

    @patch("wandb.init")
    @patch("wandb.Artifact")
    def test_wandb_integration(self, mock_artifact_class, mock_init):
        """Test WandB integration when online."""
        import os
        os.environ.pop("PIXELIS_OFFLINE_MODE", None)
    
        # Reset singleton
        ArtifactManager._instance = None
    
        # Set up mocks
        mock_run = MagicMock()
        mock_run.id = "wandb_run_123"
        mock_init.return_value = mock_run
    
        mock_artifact = MagicMock()
        mock_artifact_class.return_value = mock_artifact
    
        manager = ArtifactManager()
        manager.init_run("test_run", project="test_project")
    
        # Verify WandB was initialized
>       mock_init.assert_called_once()
E       AssertionError: Expected 'init' to have been called once. Called 0 times.

tests/unit/test_artifact_manager.py:198: AssertionError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using WandB storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started WandB run: <MagicMock name='mock.init().id' id='132371258785216'>
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:223 Using WandB storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:269 Started WandB run: <MagicMock name='mock.init().id' id='132371258785216'>
________________________________________________________________________ TestArtifactManager.test_list_artifacts ________________________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440caf8e0>

    def test_list_artifacts(self):
        """Test listing all artifacts."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        # Log several artifacts
        artifacts = []
        for i in range(3):
            artifact = manager.log_artifact(
                name=f"artifact_{i}",
                type=ArtifactType.METRICS,
                data={"index": i}
            )
            artifacts.append(artifact)
    
        # List artifacts
>       all_artifacts = manager.list_artifacts()
E       AttributeError: 'ArtifactManager' object has no attribute 'list_artifacts'. Did you mean: 'log_artifact'?

tests/unit/test_artifact_manager.py:217: AttributeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: artifact_0 vv1
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: artifact_1 vv1
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: artifact_2 vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: artifact_0 vv1
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: artifact_1 vv1
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: artifact_2 vv1
___________________________________________________________ TestArtifactManager.test_all_artifact_types[ArtifactType.DATASET] ___________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cafe20>, artifact_type = <ArtifactType.DATASET: 'dataset'>

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_all_artifact_types(self, artifact_type):
        """Test logging all artifact types."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        artifact = manager.log_artifact(
            name=f"test_{artifact_type.value}",
            type=artifact_type,
            data={"type": artifact_type.value}
        )
    
>       assert artifact["type"] == artifact_type.value
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:256: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_dataset vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_dataset vv1
____________________________________________________________ TestArtifactManager.test_all_artifact_types[ArtifactType.MODEL] ____________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cafee0>, artifact_type = <ArtifactType.MODEL: 'model'>

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_all_artifact_types(self, artifact_type):
        """Test logging all artifact types."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        artifact = manager.log_artifact(
            name=f"test_{artifact_type.value}",
            type=artifact_type,
            data={"type": artifact_type.value}
        )
    
>       assert artifact["type"] == artifact_type.value
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:256: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_model vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_model vv1
_________________________________________________________ TestArtifactManager.test_all_artifact_types[ArtifactType.CHECKPOINT] __________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cf8340>, artifact_type = <ArtifactType.CHECKPOINT: 'checkpoint'>

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_all_artifact_types(self, artifact_type):
        """Test logging all artifact types."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        artifact = manager.log_artifact(
            name=f"test_{artifact_type.value}",
            type=artifact_type,
            data={"type": artifact_type.value}
        )
    
>       assert artifact["type"] == artifact_type.value
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:256: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_checkpoint vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_checkpoint vv1
___________________________________________________________ TestArtifactManager.test_all_artifact_types[ArtifactType.CONFIG] ____________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cf8400>, artifact_type = <ArtifactType.CONFIG: 'config'>

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_all_artifact_types(self, artifact_type):
        """Test logging all artifact types."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        artifact = manager.log_artifact(
            name=f"test_{artifact_type.value}",
            type=artifact_type,
            data={"type": artifact_type.value}
        )
    
>       assert artifact["type"] == artifact_type.value
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:256: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_config vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_config vv1
___________________________________________________________ TestArtifactManager.test_all_artifact_types[ArtifactType.METRICS] ___________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cf84c0>, artifact_type = <ArtifactType.METRICS: 'metrics'>

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_all_artifact_types(self, artifact_type):
        """Test logging all artifact types."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        artifact = manager.log_artifact(
            name=f"test_{artifact_type.value}",
            type=artifact_type,
            data={"type": artifact_type.value}
        )
    
>       assert artifact["type"] == artifact_type.value
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:256: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_metrics vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_metrics vv1
_________________________________________________________ TestArtifactManager.test_all_artifact_types[ArtifactType.EVALUATION] __________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cf8580>, artifact_type = <ArtifactType.EVALUATION: 'evaluation'>

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_all_artifact_types(self, artifact_type):
        """Test logging all artifact types."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        artifact = manager.log_artifact(
            name=f"test_{artifact_type.value}",
            type=artifact_type,
            data={"type": artifact_type.value}
        )
    
>       assert artifact["type"] == artifact_type.value
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:256: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_evaluation vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_evaluation vv1
_________________________________________________________ TestArtifactManager.test_all_artifact_types[ArtifactType.ENVIRONMENT] _________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cf8640>, artifact_type = <ArtifactType.ENVIRONMENT: 'environment'>

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_all_artifact_types(self, artifact_type):
        """Test logging all artifact types."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        artifact = manager.log_artifact(
            name=f"test_{artifact_type.value}",
            type=artifact_type,
            data={"type": artifact_type.value}
        )
    
>       assert artifact["type"] == artifact_type.value
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:256: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_environment vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_environment vv1
____________________________________________________________ TestArtifactManager.test_all_artifact_types[ArtifactType.CODE] _____________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cf8700>, artifact_type = <ArtifactType.CODE: 'code'>

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_all_artifact_types(self, artifact_type):
        """Test logging all artifact types."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        artifact = manager.log_artifact(
            name=f"test_{artifact_type.value}",
            type=artifact_type,
            data={"type": artifact_type.value}
        )
    
>       assert artifact["type"] == artifact_type.value
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:256: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_code vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_code vv1
_________________________________________________________ TestArtifactManager.test_all_artifact_types[ArtifactType.EXPERIENCE] __________________________________________________________

self = <test_artifact_manager.TestArtifactManager object at 0x786440cf87c0>, artifact_type = <ArtifactType.EXPERIENCE: 'experience'>

    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_all_artifact_types(self, artifact_type):
        """Test logging all artifact types."""
        manager = ArtifactManager()
        manager.init_run("test_run")
    
        artifact = manager.log_artifact(
            name=f"test_{artifact_type.value}",
            type=artifact_type,
            data={"type": artifact_type.value}
        )
    
>       assert artifact["type"] == artifact_type.value
E       TypeError: 'ArtifactMetadata' object is not subscriptable

tests/unit/test_artifact_manager.py:256: TypeError
--------------------------------------------------------------------------------- Captured stdout call ----------------------------------------------------------------------------------
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Using local storage backend for artifacts
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Started offline run: test_run_20250817_133224
2025-08-17 13:32:24 - core.reproducibility.artifact_manager - INFO - Logged data artifact: test_experience vv1
----------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------
INFO     core.reproducibility.artifact_manager:artifact_manager.py:220 Using local storage backend for artifacts
INFO     core.reproducibility.artifact_manager:artifact_manager.py:252 Started offline run: test_run_20250817_133224
INFO     core.reproducibility.artifact_manager:artifact_manager.py:330 Logged data artifact: test_experience vv1
==================================================================================== tests coverage =====================================================================================
___________________________________________________________________ coverage: platform linux, python 3.10.18-final-0 ____________________________________________________________________

Name                                         Stmts   Miss Branch BrPart  Cover   Missing
----------------------------------------------------------------------------------------
core/engine/inference_engine.py                581    581    174      0  0.00%   9-1570
core/engine/update_worker.py                   299    299     68      0  0.00%   13-906
core/models/peft_model.py                        3      3      0      0  0.00%   3-19
core/modules/alerter.py                        181    181     56      0  0.00%   8-480
core/modules/audit.py                          246    246     70      0  0.00%   11-631
core/modules/dynamics_model.py                 160    160     40      0  0.00%   8-560
core/modules/experience_buffer.py              341    341    136      0  0.00%   9-782
core/modules/experience_buffer_enhanced.py     396    396    168      0  0.00%   13-885
core/modules/operation_registry.py              74     74     18      0  0.00%   8-298
core/modules/operations/base_operation.py       21     21      0      0  0.00%   8-113
core/modules/operations/get_properties.py      137    137     52      0  0.00%   7-438
core/modules/operations/read_text.py           111    111     48      0  0.00%   7-276
core/modules/operations/segment_object.py       97     97     34      0  0.00%   7-268
core/modules/operations/track_object.py        181    181     50      0  0.00%   7-519
core/modules/operations/zoom_in.py             179    179     68      0  0.00%   7-457
core/modules/persistence_adapter.py            276    276     50      0  0.00%   8-498
core/modules/privacy.py                        220    220     88      0  0.00%   8-661
core/modules/reward_shaping.py                 223    223     84      0  0.00%   8-679
core/modules/reward_shaping_enhanced.py        301    301     98      0  0.00%   8-814
core/modules/voting.py                         132    132     42      0  0.00%   8-489
core/reproducibility/artifact_manager.py       348    173    124     19 42.80%   19-20, 25-26, 68->70, 71, 73, 101, 105-110, 114-120, 128, 138, 142-151, 155-165, 180->182, 201->212, 204
-208, 237-238, 249-250, 266->269, 287, 304-315, 321->333, 340, 357-382, 405-411, 419-426, 430-442, 446-467, 471-475, 489-499, 508-515, 530-548, 563-564, 568-592, 601-618, 627-637, 646-675                                                                                                                                                                                       core/reproducibility/config_capture.py         251    207     92      0 12.83%   18-19, 24-25, 29, 61-110, 115-181, 186-242, 247-263, 268-317, 322-369, 374-424, 430-477, 496-547
core/reproducibility/decorators.py             200    180     82      0  7.09%   45-201, 227-286, 310-436, 451-525
core/reproducibility/experiment_context.py     185    144     52      0 17.30%   19-20, 25-26, 31-32, 45-54, 58-68, 72-104, 108-155, 186-203, 207-248, 252-297, 301-303, 312, 326-335, 34
4-366, 392-404, 421-444, 455-478, 488-501                                                                                                                                                core/reproducibility/lineage_tracker.py        234    197    108      0 10.82%   29-30, 41-51, 79-110, 127-147, 164-184, 201-224, 236-264, 277-303, 312-372, 384-412, 421-434, 443-460, 4
69-488, 492-504, 508-518, 522-530, 534-541, 545-556, 560-578                                                                                                                             core/utils/logging_utils.py                     95     54     26      3 41.32%   37, 40, 59-65, 101-107, 117-126, 147-157, 164-169, 176-191, 201-224
----------------------------------------------------------------------------------------
TOTAL                                         5472   5114   1828     22  5.40%
Coverage HTML written to dir htmlcov
Coverage XML written to file coverage.xml
FAIL Required test coverage of 70% not reached. Total coverage: 5.40%
================================================================================ short test summary info ================================================================================
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_log_artifact - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_log_large_artifact - AttributeError: 'PosixPath' object attribute 'stat' is read-only
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_use_artifact - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_artifact_versioning - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_artifact_lineage - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_content_addressable_storage - AttributeError: 'ArtifactManager' object has no attribute '_compute_content_hash'. Di
d you mean: '_compute_file_hash'?                                                                                                                                                        FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_wandb_integration - AssertionError: Expected 'init' to have been called once. Called 0 times.
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_list_artifacts - AttributeError: 'ArtifactManager' object has no attribute 'list_artifacts'. Did you mean: 'log_art
ifact'?                                                                                                                                                                                  FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.DATASET] - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.MODEL] - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.CHECKPOINT] - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.CONFIG] - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.METRICS] - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.EVALUATION] - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.ENVIRONMENT] - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.CODE] - TypeError: 'ArtifactMetadata' object is not subscriptable
FAILED tests/unit/test_artifact_manager.py::TestArtifactManager::test_all_artifact_types[ArtifactType.EXPERIENCE] - TypeError: 'ArtifactMetadata' object is not subscriptable
============================================================================= 17 failed, 3 passed in 17.87s =============================================================================
