# tests/modules/test_frame_extractor.py

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from core.modules.frame_extractor import (
    FrameExtractor, OpenCVExtractor, PyAVExtractor, DecordExtractor,
    ExtractionBackend, get_default_extractor
)


@pytest.fixture
def mock_video_path(tmp_path):
    """Create a mock video file path."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"mock video content")
    return video_path


@pytest.fixture
def mock_frames():
    """Create mock frame data."""
    # Create 10 frames of 480x640 RGB
    return np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)


@pytest.fixture
def create_synthetic_video(tmp_path):
    """Create a synthetic video with known frame content."""
    def _create_video(num_frames=30, fps=10):
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not available for video creation")
            
        video_path = tmp_path / "synthetic_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (320, 240))
        
        for i in range(num_frames):
            # Create frame with frame number encoded in pixel values
            frame = np.full((240, 320, 3), i, dtype=np.uint8)
            # Add visible frame number
            cv2.putText(frame, f"Frame {i}", (50, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(frame)
        
        writer.release()
        return video_path, num_frames
    
    return _create_video


class TestFrameExtractor:
    """Test suite for frame extraction functionality."""
    
    # Test OpenCV backend
    
    @patch('core.modules.frame_extractor.OpenCVExtractor.__init__')
    def test_opencv_backend_initialization(self, mock_init):
        """Test OpenCV backend initialization."""
        mock_init.return_value = None
        
        with patch('core.modules.frame_extractor.OpenCVExtractor.extract_frames'):
            extractor = FrameExtractor(backend=ExtractionBackend.OPENCV)
            assert extractor._backend is not None
    
    def test_opencv_extract_frames(self, mock_video_path, mock_frames):
        """Test frame extraction with OpenCV backend."""
        with patch('cv2.VideoCapture') as mock_capture:
            # Setup mock video capture
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 100  # Total frames
            mock_cap.read.return_value = (True, mock_frames[0])
            mock_capture.return_value = mock_cap
            
            # Mock cv2 module
            mock_cv2 = MagicMock()
            mock_cv2.VideoCapture = mock_capture
            mock_cv2.CAP_PROP_FRAME_COUNT = 7
            mock_cv2.CAP_PROP_POS_FRAMES = 1
            mock_cv2.COLOR_BGR2RGB = 4
            mock_cv2.cvtColor = lambda x, _: x  # Pass through
            
            extractor = OpenCVExtractor()
            extractor.cv2 = mock_cv2
            
            # Extract specific frames
            frames = extractor.extract_frames(mock_video_path, [0, 5, 10])
            
            assert mock_cap.set.called
            assert mock_cap.read.called
            assert mock_cap.release.called
    
    def test_opencv_extract_frame_range(self, mock_video_path):
        """Test frame range extraction with OpenCV."""
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 100
            
            # Return different frames for each read
            frame_data = [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                for _ in range(10)
            ]
            mock_cap.read.side_effect = [(True, f) for f in frame_data]
            mock_capture.return_value = mock_cap
            
            mock_cv2 = MagicMock()
            mock_cv2.VideoCapture = mock_capture
            mock_cv2.CAP_PROP_FRAME_COUNT = 7
            mock_cv2.CAP_PROP_POS_FRAMES = 1
            mock_cv2.COLOR_BGR2RGB = 4
            mock_cv2.cvtColor = lambda x, _: x
            
            extractor = OpenCVExtractor()
            extractor.cv2 = mock_cv2
            
            # Extract range
            frames = extractor.extract_frame_range(mock_video_path, 5, 10)
            
            # Should seek to frame 5
            mock_cap.set.assert_called_with(1, 5)
            assert mock_cap.release.called
    
    def test_opencv_get_video_info(self, mock_video_path):
        """Test getting video metadata with OpenCV."""
        with patch('cv2.VideoCapture') as mock_capture:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = [30.0, 150, 1920, 1080]  # fps, frames, width, height
            mock_capture.return_value = mock_cap
            
            mock_cv2 = MagicMock()
            mock_cv2.VideoCapture = mock_capture
            mock_cv2.CAP_PROP_FPS = 5
            mock_cv2.CAP_PROP_FRAME_COUNT = 7
            mock_cv2.CAP_PROP_FRAME_WIDTH = 3
            mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
            
            extractor = OpenCVExtractor()
            extractor.cv2 = mock_cv2
            
            info = extractor.get_video_info(mock_video_path)
            
            assert info['fps'] == 30.0
            assert info['frame_count'] == 150
            assert info['width'] == 1920
            assert info['height'] == 1080
            assert info['duration'] == 5.0  # 150 frames / 30 fps
    
    # Test caching
    
    def test_frame_caching(self, mock_video_path, mock_frames):
        """Test that frame caching works correctly."""
        with patch.object(OpenCVExtractor, 'extract_frame_range') as mock_extract:
            mock_extract.return_value = mock_frames[:5]
            
            extractor = FrameExtractor(
                backend=ExtractionBackend.OPENCV,
                cache_size=10
            )
            
            # First extraction - should call backend
            frames1 = extractor.extract_frame_range(mock_video_path, 0, 4)
            assert mock_extract.call_count == 1
            
            # Second extraction - should use cache
            frames2 = extractor.extract_frame_range(mock_video_path, 0, 4)
            assert mock_extract.call_count == 1  # Still 1, not called again
            
            # Verify frames are identical
            assert np.array_equal(frames1, frames2)
    
    def test_cache_eviction(self, mock_video_path, mock_frames):
        """Test LRU cache eviction."""
        with patch.object(OpenCVExtractor, 'extract_frame_range') as mock_extract:
            mock_extract.return_value = mock_frames[:5]
            
            # Small cache size for testing eviction
            extractor = FrameExtractor(
                backend=ExtractionBackend.OPENCV,
                cache_size=2
            )
            
            # Add 3 items to cache (exceeds size of 2)
            extractor.extract_frame_range(mock_video_path, 0, 4)
            extractor.extract_frame_range(mock_video_path, 5, 9)
            extractor.extract_frame_range(mock_video_path, 10, 14)
            
            # First item should be evicted
            assert len(extractor._cache_order) <= 2
    
    def test_cache_clear(self, mock_video_path, mock_frames):
        """Test clearing the cache."""
        with patch.object(OpenCVExtractor, 'extract_frame_range') as mock_extract:
            mock_extract.return_value = mock_frames[:5]
            
            extractor = FrameExtractor(backend=ExtractionBackend.OPENCV)
            
            # Add to cache
            extractor.extract_frame_range(mock_video_path, 0, 4)
            assert len(extractor._cache) > 0
            
            # Clear cache
            extractor.clear_cache()
            assert len(extractor._cache) == 0
            assert len(extractor._cache_order) == 0
    
    # Test error handling
    
    def test_missing_video_file(self):
        """Test handling of missing video files."""
        extractor = FrameExtractor(backend=ExtractionBackend.OPENCV)
        
        with pytest.raises(FileNotFoundError):
            extractor.extract_frames(Path("/non/existent/video.mp4"), [0])
    
    def test_invalid_frame_indices(self, mock_video_path):
        """Test handling of invalid frame indices."""
        with patch.object(OpenCVExtractor, 'extract_frames') as mock_extract:
            # Simulate extracting out-of-range frames
            mock_extract.return_value = np.zeros((0, 480, 640, 3), dtype=np.uint8)
            
            extractor = FrameExtractor(backend=ExtractionBackend.OPENCV)
            
            # Should handle gracefully
            with pytest.raises(RuntimeError):
                frames = extractor.extract_frames(mock_video_path, [1000, 2000])
    
    # Test backend auto-selection
    
    def test_backend_auto_selection(self):
        """Test automatic backend selection."""
        # Mock all backends to fail except OpenCV
        with patch('core.modules.frame_extractor.DecordExtractor.__init__', 
                  side_effect=ImportError("Decord not available")):
            with patch('core.modules.frame_extractor.PyAVExtractor.__init__',
                      side_effect=ImportError("PyAV not available")):
                with patch('core.modules.frame_extractor.OpenCVExtractor.__init__',
                          return_value=None):
                    
                    extractor = FrameExtractor(backend=ExtractionBackend.AUTO)
                    assert isinstance(extractor._backend, OpenCVExtractor)
    
    def test_no_backend_available(self):
        """Test error when no backend is available."""
        with patch('core.modules.frame_extractor.DecordExtractor.__init__',
                  side_effect=ImportError()):
            with patch('core.modules.frame_extractor.PyAVExtractor.__init__',
                      side_effect=ImportError()):
                with patch('core.modules.frame_extractor.OpenCVExtractor.__init__',
                          side_effect=ImportError()):
                    
                    with pytest.raises(RuntimeError, match="No video processing backend"):
                        extractor = FrameExtractor(backend=ExtractionBackend.AUTO)
    
    # Test preloading
    
    def test_preload_frames(self, mock_video_path, mock_frames):
        """Test preloading multiple frame ranges."""
        with patch.object(OpenCVExtractor, 'extract_frame_range') as mock_extract:
            mock_extract.return_value = mock_frames[:5]
            
            extractor = FrameExtractor(backend=ExtractionBackend.OPENCV)
            
            # Preload multiple ranges
            ranges = [(0, 4), (10, 14), (20, 24)]
            extractor.preload_frames(mock_video_path, ranges)
            
            # Should have called extract for each range
            assert mock_extract.call_count == len(ranges)
    
    # Test singleton
    
    def test_default_extractor_singleton(self):
        """Test that get_default_extractor returns singleton."""
        extractor1 = get_default_extractor()
        extractor2 = get_default_extractor()
        
        assert extractor1 is extractor2
    
    # Test frame data validation
    
    def test_frame_shape_validation(self, mock_video_path):
        """Test that extracted frames have correct shape."""
        with patch.object(OpenCVExtractor, 'extract_frames') as mock_extract:
            # Return frames with correct shape
            frames = np.zeros((3, 480, 640, 3), dtype=np.uint8)
            mock_extract.return_value = frames
            
            extractor = FrameExtractor(backend=ExtractionBackend.OPENCV)
            result = extractor.extract_frames(mock_video_path, [0, 1, 2])
            
            assert result.shape == (3, 480, 640, 3)
            assert result.dtype == np.uint8
    
    def test_extract_single_frame(self, mock_video_path):
        """Test extracting a single frame."""
        with patch.object(OpenCVExtractor, 'extract_frames') as mock_extract:
            single_frame = np.zeros((1, 480, 640, 3), dtype=np.uint8)
            mock_extract.return_value = single_frame
            
            extractor = FrameExtractor(backend=ExtractionBackend.OPENCV)
            result = extractor.extract_frames(mock_video_path, [42])
            
            assert result.shape[0] == 1
            mock_extract.assert_called_once_with(mock_video_path, [42])
    
    def test_extract_empty_range(self, mock_video_path):
        """Test extracting with empty frame list."""
        extractor = FrameExtractor(backend=ExtractionBackend.OPENCV)
        
        with pytest.raises(Exception):
            # Should raise an error for empty frame list
            extractor.extract_frames(mock_video_path, [])


class TestFrameAwareLoader:
    """Test frame-aware loader functionality."""
    
    def test_frame_aware_mixin_initialization(self):
        """Test FrameAwareLoaderMixin initialization."""
        from core.dataloaders.frame_aware_loader import FrameAwareLoaderMixin
        
        class TestLoader(FrameAwareLoaderMixin):
            def __init__(self, extract_frames=False):
                super().__init__(extract_frames=extract_frames)
        
        # Without frame extraction
        loader1 = TestLoader(extract_frames=False)
        assert loader1.extract_frames == False
        assert loader1._frame_extractor is None
        
        # With frame extraction
        with patch('core.modules.frame_extractor.FrameExtractor'):
            loader2 = TestLoader(extract_frames=True)
            assert loader2.extract_frames == True
            assert loader2._frame_extractor is not None
    
    def test_sampling_strategies(self):
        """Test different frame sampling strategies."""
        from core.dataloaders.frame_aware_loader import FrameAwareLoaderMixin
        
        class TestLoader(FrameAwareLoaderMixin):
            def __init__(self):
                self.extract_frames = True
                self._frame_extractor = MagicMock()
        
        loader = TestLoader()
        video_path = Path("/test/video.mp4")
        
        # Test 'all' strategy
        loader._frame_extractor.extract_frame_range.return_value = np.zeros((31, 480, 640, 3))
        frames = loader._extract_frames_for_sample(video_path, 0, 30, "all")
        loader._frame_extractor.extract_frame_range.assert_called_with(video_path, 0, 30)
        
        # Test 'uniform_8' strategy
        loader._frame_extractor.extract_frames.return_value = np.zeros((8, 480, 640, 3))
        frames = loader._extract_frames_for_sample(video_path, 0, 100, "uniform_8")
        loader._frame_extractor.extract_frames.assert_called()
        
        # Test 'endpoints' strategy
        loader._frame_extractor.extract_frames.return_value = np.zeros((2, 480, 640, 3))
        frames = loader._extract_frames_for_sample(video_path, 10, 50, "endpoints")
        loader._frame_extractor.extract_frames.assert_called_with(video_path, [10, 50])
        
        # Test 'middle' strategy
        loader._frame_extractor.extract_frames.return_value = np.zeros((1, 480, 640, 3))
        frames = loader._extract_frames_for_sample(video_path, 10, 50, "middle")
        loader._frame_extractor.extract_frames.assert_called_with(video_path, [30])
    
    def test_add_frame_data_to_sample(self):
        """Test adding frame data to sample dictionary."""
        from core.dataloaders.frame_aware_loader import FrameAwareLoaderMixin
        
        class TestLoader(FrameAwareLoaderMixin):
            pass
        
        loader = TestLoader()
        
        # Create sample
        sample = {
            'sample_id': 'test',
            'annotations': {
                'action_segment': {
                    'start_frame': 10,
                    'end_frame': 20
                }
            }
        }
        
        # Add frame data
        frames = np.zeros((11, 480, 640, 3), dtype=np.uint8)
        updated_sample = loader._add_frame_data_to_sample(sample, frames, "uniform_8")
        
        assert 'frames' in updated_sample
        assert updated_sample['frames']['num_frames'] == 11
        assert updated_sample['frames']['shape'] == (11, 480, 640, 3)
        assert updated_sample['frames']['sampling_strategy'] == "uniform_8"
        assert updated_sample['frames']['original_range'] == [10, 20]


class TestPyAVExtractor:
    """Test suite for PyAV backend."""
    
    def test_pyav_extract_frames(self, mock_video_path, mock_frames):
        """Test frame extraction with PyAV backend."""
        with patch('av.open') as mock_open:
            # Create mock container
            mock_container = MagicMock()
            mock_stream = MagicMock()
            mock_stream.frames = 100
            mock_stream.average_rate = 30
            mock_stream.width = 640
            mock_stream.height = 480
            mock_container.streams.video = [mock_stream]
            
            # Create mock frames
            mock_av_frames = []
            for i in range(10):
                frame = MagicMock()
                frame.index = i
                frame.to_ndarray.return_value = mock_frames[i]
                mock_av_frames.append(frame)
            
            mock_container.decode.return_value = iter(mock_av_frames)
            mock_open.return_value.__enter__.return_value = mock_container
            
            # Test extraction
            extractor = PyAVExtractor()
            frames = extractor.extract_frames(mock_video_path, [0, 5, 9])
            
            assert frames.shape[0] == 3
            assert mock_container.seek.called
    
    def test_pyav_get_video_info(self, mock_video_path):
        """Test getting video metadata with PyAV."""
        with patch('av.open') as mock_open:
            mock_container = MagicMock()
            mock_stream = MagicMock()
            mock_stream.frames = 150
            mock_stream.average_rate = 30
            mock_stream.width = 1920
            mock_stream.height = 1080
            mock_container.streams.video = [mock_stream]
            mock_open.return_value.__enter__.return_value = mock_container
            
            extractor = PyAVExtractor()
            info = extractor.get_video_info(mock_video_path)
            
            assert info['fps'] == 30
            assert info['frame_count'] == 150
            assert info['width'] == 1920
            assert info['height'] == 1080
            assert info['duration'] == 5.0  # 150 / 30


class TestDecordExtractor:
    """Test suite for Decord backend."""
    
    def test_decord_extract_frames(self, mock_video_path, mock_frames):
        """Test frame extraction with Decord backend."""
        with patch('decord.VideoReader') as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.__len__.return_value = 100
            mock_reader.__getitem__.return_value = mock_frames[0]
            mock_reader.get_batch.return_value = mock_frames[:3]
            mock_reader_class.return_value = mock_reader
            
            with patch('decord.cpu') as mock_cpu:
                mock_cpu.return_value = 0
                
                extractor = DecordExtractor()
                frames = extractor.extract_frames(mock_video_path, [0, 5, 9])
                
                assert mock_reader.get_batch.called
    
    def test_decord_extract_frame_range(self, mock_video_path):
        """Test frame range extraction with Decord."""
        with patch('decord.VideoReader') as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.__len__.return_value = 100
            
            # Create frames for range
            range_frames = np.random.randint(0, 255, (6, 480, 640, 3), dtype=np.uint8)
            mock_reader.get_batch.return_value = range_frames
            mock_reader_class.return_value = mock_reader
            
            with patch('decord.cpu'):
                extractor = DecordExtractor()
                frames = extractor.extract_frame_range(mock_video_path, 5, 10)
                
                # Should extract 6 frames (5, 6, 7, 8, 9, 10)
                assert frames.shape[0] == 6
    
    def test_decord_gpu_acceleration(self):
        """Test Decord with GPU acceleration."""
        with patch('decord.VideoReader') as mock_reader_class:
            with patch('decord.gpu') as mock_gpu:
                mock_gpu.return_value = 0
                
                extractor = DecordExtractor(gpu_id=0)
                assert extractor._ctx is not None


class TestFrameExtractionAccuracy:
    """Test suite for frame extraction accuracy across backends."""
    
    def test_frame_accuracy_sequential(self, create_synthetic_video):
        """Test that frames are extracted in correct sequence."""
        video_path, num_frames = create_synthetic_video(30, 10)
        
        # Test with available backend
        extractor = FrameExtractor(backend=ExtractionBackend.AUTO)
        
        # Extract specific frames
        indices = [0, 10, 20, 29]
        frames = extractor.extract_frames(video_path, indices)
        
        # Verify we got the right number of frames
        assert frames.shape[0] == len(indices)
        
        # Check frame values (approximately - compression may alter values)
        for i, idx in enumerate(indices):
            # The first pixel should roughly match the frame index
            assert abs(frames[i, 0, 0, 0] - idx) < 10  # Allow for compression artifacts
    
    def test_frame_accuracy_range(self, create_synthetic_video):
        """Test that frame ranges are extracted correctly."""
        video_path, num_frames = create_synthetic_video(20, 5)
        
        extractor = FrameExtractor(backend=ExtractionBackend.AUTO)
        
        # Extract a range
        frames = extractor.extract_frame_range(video_path, 5, 14)
        
        # Should get exactly 10 frames (5 through 14 inclusive)
        assert frames.shape[0] == 10
    
    def test_sampling_accuracy(self, create_synthetic_video):
        """Test that sampling strategies produce expected results."""
        from core.dataloaders.frame_aware_loader import FrameAwareLoaderMixin
        
        video_path, _ = create_synthetic_video(100, 25)
        
        class TestLoader(FrameAwareLoaderMixin):
            def __init__(self):
                self.extract_frames = True
                self._frame_extractor = FrameExtractor(backend=ExtractionBackend.AUTO)
        
        loader = TestLoader()
        
        # Test uniform sampling
        frames = loader._extract_frames_for_sample(video_path, 0, 99, "uniform_8")
        assert frames.shape[0] == 8
        
        # Test endpoints
        frames = loader._extract_frames_for_sample(video_path, 10, 90, "endpoints")
        assert frames.shape[0] == 2
        
        # Test middle
        frames = loader._extract_frames_for_sample(video_path, 20, 80, "middle")
        assert frames.shape[0] == 1


class TestConcurrentAccess:
    """Test concurrent access to frame extraction."""
    
    def test_concurrent_extraction(self, mock_video_path, mock_frames):
        """Test that multiple threads can extract frames concurrently."""
        import threading
        import queue
        
        with patch.object(OpenCVExtractor, 'extract_frames') as mock_extract:
            mock_extract.return_value = mock_frames[:3]
            
            extractor = FrameExtractor(backend=ExtractionBackend.OPENCV)
            results = queue.Queue()
            
            def extract_worker(indices):
                try:
                    frames = extractor.extract_frames(mock_video_path, indices)
                    results.put(frames.shape)
                except Exception as e:
                    results.put(e)
            
            # Start multiple threads
            threads = []
            for i in range(5):
                t = threading.Thread(target=extract_worker, args=([i, i+1, i+2],))
                threads.append(t)
                t.start()
            
            # Wait for completion
            for t in threads:
                t.join()
            
            # Check results
            for _ in range(5):
                result = results.get()
                assert isinstance(result, tuple)
                assert result == (3, 480, 640, 3)
    
    def test_cache_thread_safety(self, mock_video_path):
        """Test that cache is thread-safe."""
        import threading
        
        with patch.object(OpenCVExtractor, 'extract_frame_range') as mock_extract:
            mock_extract.return_value = np.zeros((5, 480, 640, 3))
            
            extractor = FrameExtractor(backend=ExtractionBackend.OPENCV, cache_size=10)
            
            def cache_worker():
                for i in range(10):
                    extractor.extract_frame_range(mock_video_path, i*5, i*5+4)
            
            threads = [threading.Thread(target=cache_worker) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Cache should still be valid
            assert len(extractor._cache) > 0
            assert len(extractor._cache_order) <= 10


class TestPerformance:
    """Performance benchmarks for frame extraction."""
    
    def test_extraction_performance(self, create_synthetic_video):
        """Benchmark frame extraction speed."""
        import time
        
        video_path, _ = create_synthetic_video(100, 30)
        extractor = FrameExtractor(backend=ExtractionBackend.AUTO)
        
        # Warm up
        extractor.extract_frames(video_path, [0])
        
        # Benchmark single frame extraction
        start = time.time()
        for i in range(10):
            extractor.extract_frames(video_path, [i * 10])
        single_time = time.time() - start
        
        # Benchmark batch extraction
        start = time.time()
        extractor.extract_frames(video_path, list(range(0, 100, 10)))
        batch_time = time.time() - start
        
        # Batch should be faster than individual
        assert batch_time < single_time
        
        # Print performance info
        print(f"\nPerformance: Single={single_time:.3f}s, Batch={batch_time:.3f}s")
        print(f"Speedup: {single_time/batch_time:.2f}x")
    
    def test_cache_performance(self, create_synthetic_video):
        """Test cache performance impact."""
        import time
        
        video_path, _ = create_synthetic_video(50, 25)
        extractor = FrameExtractor(backend=ExtractionBackend.AUTO, cache_size=10)
        
        # First extraction (no cache)
        start = time.time()
        frames1 = extractor.extract_frame_range(video_path, 10, 20)
        uncached_time = time.time() - start
        
        # Second extraction (with cache)
        start = time.time()
        frames2 = extractor.extract_frame_range(video_path, 10, 20)
        cached_time = time.time() - start
        
        # Cache should be significantly faster
        assert cached_time < uncached_time / 2
        
        # Verify same results
        assert np.array_equal(frames1, frames2)
        
        print(f"\nCache Performance: Uncached={uncached_time:.3f}s, Cached={cached_time:.3f}s")
        print(f"Cache Speedup: {uncached_time/cached_time:.1f}x")