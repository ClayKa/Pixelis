# Pixelis Test Suite Documentation

*Generated on: 2025-08-15 03:42:04*

## Summary Statistics

- **Total Test Files**: 13
- **Total Test Functions**: 189
- **Total Test Classes**: 34
- **Total Standalone Tests**: 17

## Table of Contents

1. [tests/engine/test_async_communication.py](#tests-engine-test_async_communication-py) (15 tests)
2. [tests/engine/test_inference_engine.py](#tests-engine-test_inference_engine-py) (14 tests)
3. [tests/engine/test_ipc.py](#tests-engine-test_ipc-py) (19 tests)
4. [tests/engine/test_update_worker.py](#tests-engine-test_update_worker-py) (27 tests)
5. [tests/modules/test_experience_buffer.py](#tests-modules-test_experience_buffer-py) (20 tests)
6. [tests/modules/test_model_init.py](#tests-modules-test_model_init-py) (12 tests)
7. [tests/modules/test_voting.py](#tests-modules-test_voting-py) (14 tests)
8. [tests/test_basic.py](#tests-test_basic-py) (5 tests)
9. [tests/test_experimental_protocol.py](#tests-test_experimental_protocol-py) (14 tests)
10. [tests/test_integration.py](#tests-test_integration-py) (7 tests)
11. [tests/test_rft_training.py](#tests-test_rft_training-py) (26 tests)
12. [tests/test_sft_curriculum.py](#tests-test_sft_curriculum-py) (4 tests)
13. [tests/unit/test_artifact_manager.py](#tests-unit-test_artifact_manager-py) (12 tests)

## Detailed Test Listing

### tests/engine/test_async_communication.py <a id='tests-engine-test_async_communication-py'></a>

**Module**: `tests.engine.test_async_communication`
**Total Tests**: 15

#### Test Classes:

**`TestAsyncCommunication`** (5 tests)

- `test_inference_engine_initialization` (line 191)
  > Test initializing the inference engine.
- `test_update_worker_initialization` (line 229)
  > Test initializing the update worker.
- `test_adaptive_learning_rate` (line 259)
  > Test adaptive learning rate calculation.
- `test_queue_communication` (line 335)
  > Test basic queue communication between processes.
- `test_update_task_enqueue_with_shared_memory` (line 364)
  > Test enqueueing update task with shared memory transfer.

**`TestFaultTolerance`** (5 tests)

- `test_watchdog_cleanup_on_timeout` (line 419)
  > Test watchdog cleaning up segments after timeout.
- `test_worker_failure_recovery` (line 458)
  > Test recovery when update worker fails.
- `test_graceful_shutdown` (line 504)
  > Test graceful shutdown of the system.
- `test_queue_timeout_handling` (line 549)
  > Test handling of queue timeouts.
- `test_shared_memory_reconstruction_error` (line 586)
  > Test handling errors in shared memory reconstruction.

**`TestSharedMemoryManager`** (5 tests)

- `test_create_shared_tensor` (line 31)
  > Test creating a shared memory segment for a tensor.
- `test_cuda_tensor_transfer` (line 52)
  > Test transferring CUDA tensors to shared memory.
- `test_cleanup_stale_segments` (line 71)
  > Test cleaning up stale shared memory segments.
- `test_cleanup_on_worker_death` (line 100)
  > Test cleaning up all segments when worker dies.
- `test_get_status` (line 117)
  > Test getting status of the shared memory manager.

---

### tests/engine/test_inference_engine.py <a id='tests-engine-test_inference_engine-py'></a>

**Module**: `tests.engine.test_inference_engine`
**Total Tests**: 14

#### Test Classes:

**`TestInferenceEngine`** (9 tests)

- `test_initialization` (line 131)
  > Test inference engine initialization.
- `test_should_trigger_update` (line 151)
  > Test confidence gating mechanism.
- `test_calculate_adaptive_lr` (line 165)
  > Test adaptive learning rate calculation.
- `test_should_request_human_review` (line 186)
  > Test HIL sampling logic.
- `test_infer_and_adapt` (line 209)
  > Test the main inference and adaptation loop.
- `test_process_human_review_decision` (line 252)
  > Test processing human review decisions.
- `test_watchdog_cleanup` (line 274)
  > Test watchdog cleanup functionality.
- `test_stats_tracking` (line 286)
  > Test statistics tracking.
- `test_adaptive_lr_bounds` (line 300)
  > Test that adaptive LR respects bounds across all confidence values.

**`TestIntegration`** (1 tests)

- `test_voting_result_provenance` (line 320)
  > Test that VotingResult properly maintains provenance.

**`TestSharedMemoryManager`** (4 tests)

- `test_create_shared_tensor` (line 33)
  > Test creating a shared memory segment for a tensor.
- `test_mark_cleaned` (line 50)
  > Test marking a segment as cleaned.
- `test_cleanup_stale_segments` (line 61)
  > Test cleanup of stale segments.
- `test_get_status` (line 80)
  > Test getting status of shared memory manager.

---

### tests/engine/test_ipc.py <a id='tests-engine-test_ipc-py'></a>

**Module**: `tests.engine.test_ipc`
**Total Tests**: 19

#### Test Classes:

**`TestCleanupMechanisms`** (3 tests)

- `test_cleanup_confirmation_flow` (line 367)
  > Test the cleanup confirmation flow.
- `test_timeout_based_cleanup` (line 388)
  > Test timeout-based cleanup of stale segments.
- `test_forced_cleanup_on_shutdown` (line 415)
  > Test forced cleanup of all segments on shutdown.

**`TestEdgeCases`** (4 tests)

- `test_empty_tensor_transfer` (line 442)
  > Test transferring empty tensors.
- `test_scalar_tensor_transfer` (line 457)
  > Test transferring scalar tensors.
- `test_concurrent_access` (line 471)
  > Test concurrent access to shared memory manager.
- `test_queue_full_handling` (line 499)
  > Test handling of full queues.

**`TestProcessCommunication`** (3 tests)

- `test_tensor_transfer_between_processes` (line 248)
  > Test transferring tensor between processes via shared memory.
- `test_bidirectional_communication` (line 295)
  > Test bidirectional communication between processes.
- `test_process_error_handling` (line 340)
  > Test error handling in process communication.

**`TestQueueCommunication`** (4 tests)

- `test_basic_queue_operations` (line 167)
  > Test basic queue put/get operations.
- `test_queue_timeout` (line 191)
  > Test queue timeout behavior.
- `test_queue_with_none_signal` (line 206)
  > Test using None as shutdown signal.
- `test_multiple_queues` (line 225)
  > Test using multiple queues for bidirectional communication.

**`TestSharedMemoryTransfer`** (5 tests)

- `test_basic_tensor_transfer` (line 71)
  > Test basic tensor transfer via shared memory.
- `test_large_tensor_transfer` (line 89)
  > Test transferring large tensors.
- `test_multiple_tensor_transfer` (line 105)
  > Test transferring multiple tensors simultaneously.
- `test_tensor_dtype_preservation` (line 132)
  > Test that tensor dtypes are preserved during transfer.
- `test_gpu_to_cpu_transfer` (line 147)
  > Test transferring GPU tensors to shared memory.

---

### tests/engine/test_update_worker.py <a id='tests-engine-test_update_worker-py'></a>

**Module**: `tests.engine.test_update_worker`
**Total Tests**: 27

#### Test Classes:

**`TestIntegration`** (2 tests)

- `test_worker_queue_processing` (line 513)
  > Test worker processing tasks from queue.
- `test_ema_synchronization` (line 559)
  > Test EMA model synchronization.

**`TestKLConfig`** (4 tests)

- `test_valid_config` (line 149)
  > Test creating a valid KL configuration.
- `test_invalid_mode` (line 160)
  > Test invalid beta update mode.
- `test_invalid_target_kl` (line 165)
  > Test invalid target KL.
- `test_invalid_beta_bounds` (line 170)
  > Test invalid beta bounds.

**`TestSharedMemoryReconstructor`** (2 tests)

- `test_reconstruct_tensor` (line 179)
  > Test reconstructing tensor from shared memory info.
- `test_reconstruct_with_custom_dtype` (line 192)
  > Test reconstruction with custom dtype.

**`TestUpdateWorker`** (18 tests)

- `test_initialization` (line 209)
  > Test worker initialization.
- `test_create_ema_model` (line 218)
  > Test EMA model creation.
- `test_create_optimizer` (line 227)
  > Test optimizer creation.
- `test_calculate_kl_penalty` (line 238)
  > Test KL divergence penalty calculation.
- `test_calculate_kl_penalty_no_original` (line 251)
  > Test KL penalty with no original logits.
- `test_update_ema_model` (line 262)
  > Test EMA model update.
- `test_update_kl_tracking` (line 283)
  > Test KL divergence tracking.
- `test_adjust_beta_increase` (line 292)
  > Test beta adjustment when KL is too high.
- `test_adjust_beta_decrease` (line 305)
  > Test beta adjustment when KL is too low.
- `test_adjust_beta_bounds` (line 318)
  > Test beta adjustment respects bounds.
- `test_save_ema_snapshot` (line 336)
  > Test saving EMA model snapshot.
- `test_cleanup_old_snapshots` (line 356)
  > Test cleanup of old model snapshots.
- `test_process_update_success` (line 374)
  > Test successful update processing.
- `test_process_update_high_kl_skip` (line 388)
  > Test skipping update when KL is too high.
- `test_process_update_with_shared_memory` (line 400)
  > Test processing update with shared memory.
- `test_log_update` (line 433)
  > Test update logging.
- `test_get_statistics` (line 472)
  > Test getting worker statistics.
- `test_shutdown` (line 488)
  > Test graceful shutdown.

#### Standalone Tests:

- `test_config` (line 74)
  > Create test configuration.

---

### tests/modules/test_experience_buffer.py <a id='tests-modules-test_experience_buffer-py'></a>

**Module**: `tests.modules.test_experience_buffer`
**Total Tests**: 20

#### Test Classes:

**`TestExperienceBuffer`** (16 tests)

- `test_basic_add_and_get` (line 101)
  > Test basic add and get operations.
- `test_duplicate_prevention` (line 119)
  > Test that duplicate experiences are rejected.
- `test_priority_calculation` (line 136)
  > Test multi-factor priority calculation.
- `test_priority_sampling` (line 158)
  > Test priority-based sampling.
- `test_hybrid_embedding_creation` (line 179)
  > Test hybrid embedding creation.
- `test_knn_search` (line 208)
  > Test k-NN search functionality.
- `test_value_tracking` (line 232)
  > Test value tracking with retrieval and success counts.
- `test_persistence_save_and_load` (line 269)
  > Test persistence with save and load.
- `test_wal_recovery` (line 296)
  > Test Write-Ahead Log recovery after crash.
- `test_concurrent_writes` (line 330)
  > Test concurrent write safety.
- `test_faiss_backend_fallback` (line 369)
  > Test FAISS backend fallback from GPU to CPU.
- `test_lmdb_persistence` (line 391)
  > Test LMDB persistence adapter.
- `test_buffer_overflow` (line 421)
  > Test buffer behavior when full.
- `test_statistics` (line 445)
  > Test statistics calculation.
- `test_index_rebuild` (line 468)
  > Test asynchronous index rebuild.
- `test_empty_buffer_operations` (line 493)
  > Test operations on empty buffer.

**`TestPersistenceAdapters`** (4 tests)

- `test_file_adapter_basic` (line 528)
  > Test file-based persistence adapter.
- `test_file_adapter_snapshot` (line 558)
  > Test file adapter snapshot functionality.
- `test_lmdb_adapter_basic` (line 579)
  > Test LMDB persistence adapter.
- `test_adapter_factory` (line 615)
  > Test persistence adapter factory.

---

### tests/modules/test_model_init.py <a id='tests-modules-test_model_init-py'></a>

**Module**: `tests.modules.test_model_init`
**Total Tests**: 12

#### Test Classes:

**`TestDynamicLoRAConfig`** (4 tests)

- `test_load_config` (line 73)
  > Test loading LoRA configuration
- `test_get_layer_ranks` (line 80)
  > Test getting layer-specific ranks
- `test_get_compression_ratio` (line 89)
  > Test getting compression ratio
- `test_create_lora_config` (line 96)
  > Test creating LoraConfig with dynamic ranks

**`TestLoRAInsertion`** (2 tests)

- `test_lora_layer_insertion` (line 169)
  > Test that LoRA layers are correctly inserted
- `test_heterogeneous_ranks` (line 207)
  > Test that different layers get different ranks as configured

**`TestPerformanceAssertions`** (3 tests)

- `test_memory_usage` (line 257)
  > Test that memory usage is below threshold
- `test_inference_latency` (line 285)
  > Test that inference latency is below threshold
- `test_memory_with_gradient_checkpointing` (line 342)
  > Test memory usage with gradient checkpointing enabled

**`TestSVDArtifactPersistence`** (3 tests)

- `test_save_singular_value_plots` (line 407)
  > Test saving singular value decay plots
- `test_save_raw_svd_data` (line 428)
  > Test saving raw SVD data
- `test_save_delta_weights` (line 460)
  > Test saving delta weight matrices

---

### tests/modules/test_voting.py <a id='tests-modules-test_voting-py'></a>

**Module**: `tests.modules.test_voting`
**Total Tests**: 14

#### Test Classes:

**`TestTemporalEnsembleVoting`** (14 tests)

- `test_initialization` (line 77)
  > Test voting module initialization.
- `test_majority_voting` (line 93)
  > Test majority voting strategy.
- `test_weighted_voting` (line 113)
  > Test weighted voting strategy.
- `test_confidence_voting` (line 130)
  > Test confidence-based voting strategy.
- `test_ensemble_voting` (line 147)
  > Test ensemble voting strategy.
- `test_insufficient_votes` (line 163)
  > Test handling of insufficient votes.
- `test_calculate_vote_weight` (line 180)
  > Test vote weight calculation.
- `test_calculate_agreement_factor` (line 192)
  > Test agreement factor calculation.
- `test_voting_result_validation` (line 211)
  > Test VotingResult validation and methods.
- `test_analyze_voting_consistency` (line 243)
  > Test voting consistency analysis.
- `test_empty_neighbors` (line 269)
  > Test voting with no neighbors.
- `test_missing_trajectory_data` (line 281)
  > Test handling of neighbors with missing trajectory data.
- `test_provenance_required_fields` (line 301)
  > Test that all required provenance fields are present.
- `test_confidence_bounds` (line 319)
  > Test that confidence values are within valid bounds.

---

### tests/test_basic.py <a id='tests-test_basic-py'></a>

**Module**: `tests.test_basic`
**Total Tests**: 5

#### Standalone Tests:

- `test_visual_operations` (line 16)
  > Test visual operations registry and execution.
- `test_data_structures` (line 63)
  > Test core data structures.
- `test_configuration` (line 109)
  > Test configuration system.
- `test_voting_module` (line 136)
  > Test temporal ensemble voting.
- `test_dynamics_model` (line 180)
  > Test dynamics model for curiosity.

---

### tests/test_experimental_protocol.py <a id='tests-test_experimental_protocol-py'></a>

**Module**: `tests.test_experimental_protocol`
**Total Tests**: 14

#### Test Classes:

**`TestExperimentReproducibility`** (2 tests)

- `test_environment_capture` (line 384)
  > Test that environment is properly captured
- `test_configuration_versioning` (line 401)
  > Test configuration versioning for reproducibility

**`TestExperimentalProtocol`** (9 tests)

- `test_multi_seed_requirement` (line 47)
  > Test that multi-seed runs are enforced
- `test_aggregated_reporting_format` (line 70)
  > Test that results are properly formatted as mean ± std
- `test_statistical_significance_testing` (line 100)
  > Test statistical significance testing between experiments
- `test_experiment_config_validation` (line 140)
  > Test experiment configuration validation
- `test_reproducibility_with_seeds` (line 165)
  > Test that same seed produces consistent initialization
- `test_registry_management` (line 175)
  > Test experiment registry operations
- `test_metric_aggregation_edge_cases` (line 218)
  > Test metric aggregation with edge cases
- `test_comparison_table_generation` (line 244)
  > Test generation of comparison tables
- `test_parallel_seed_execution` (line 291)
  > Test parallel seed execution configuration

**`TestStatisticalMethods`** (3 tests)

- `test_paired_t_test` (line 323)
  > Test paired t-test implementation
- `test_bootstrap_confidence_intervals` (line 342)
  > Test bootstrap confidence interval calculation
- `test_bonferroni_correction` (line 364)
  > Test Bonferroni correction for multiple comparisons

---

### tests/test_integration.py <a id='tests-test_integration-py'></a>

**Module**: `tests.test_integration`
**Total Tests**: 7

#### Standalone Tests:

- `test_visual_operations` (line 30)
  > Test visual operations registry and execution.
- `test_data_structures` (line 62)
  > Test core data structures.
- `test_configuration` (line 103)
  > Test configuration system.
- `test_experience_buffer` (line 128)
  > Test experience buffer functionality.
- `test_voting_module` (line 181)
  > Test temporal ensemble voting.
- `test_reward_orchestrator` (line 222)
  > Test reward calculation system.
- `test_dynamics_model` (line 267)
  > Test dynamics model for curiosity.

---

### tests/test_rft_training.py <a id='tests-test_rft_training-py'></a>

**Module**: `tests.test_rft_training`
**Total Tests**: 26

#### Test Classes:

**`TestEnhancedCoherenceAnalyzer`** (5 tests)

- `test_initialization` (line 224)
  > Test analyzer initialization.
- `test_empty_trajectory` (line 236)
  > Test handling of empty trajectory.
- `test_repetition_detection` (line 245)
  > Test detection of repetitive actions.
- `test_good_sequence_detection` (line 261)
  > Test detection of good action sequences.
- `test_embedding_similarity_analysis` (line 277)
  > Test coherence based on embedding similarity.

**`TestEnhancedCuriosityModule`** (3 tests)

- `test_initialization` (line 143)
  > Test module initialization.
- `test_curiosity_reward_computation` (line 160)
  > Test curiosity reward calculation.
- `test_caching_mechanism` (line 187)
  > Test LRU caching for efficiency.

**`TestEnhancedRewardOrchestrator`** (4 tests)

- `test_initialization` (line 381)
  > Test orchestrator initialization.
- `test_total_reward_calculation` (line 400)
  > Test calculation of total reward.
- `test_running_statistics_normalization` (line 448)
  > Test normalization with running statistics.
- `test_curriculum_weight_adjustment` (line 466)
  > Test curriculum-based weight adjustment.

**`TestGRPOTrainer`** (2 tests)

- `test_initialization` (line 512)
  > Test GRPO trainer initialization.
- `test_group_advantage_normalization` (line 539)
  > Test group-based advantage normalization.

**`TestIntegration`** (2 tests)

- `test_end_to_end_reward_calculation` (line 616)
  > Test complete reward calculation pipeline.
- `test_memory_efficiency` (line 665)
  > Test memory efficiency of LoRA implementation.

**`TestLightweightDynamicsModel`** (3 tests)

- `test_initialization` (line 65)
  > Test model initialization.
- `test_forward_pass` (line 84)
  > Test forward pass through dynamics model.
- `test_lora_parameter_efficiency` (line 111)
  > Test that LoRA reduces parameter count.

**`TestToolMisusePenaltyCalculator`** (5 tests)

- `test_initialization` (line 303)
  > Test calculator initialization.
- `test_no_violations` (line 311)
  > Test trajectory with no violations.
- `test_missing_prerequisite` (line 328)
  > Test penalty for missing prerequisite.
- `test_track_on_static_image` (line 345)
  > Test severe penalty for tracking on static image.
- `test_out_of_bounds_coordinates` (line 361)
  > Test penalty for out-of-bounds coordinates.

**`TestUtilityFunctions`** (2 tests)

- `test_parse_trajectory` (line 574)
  > Test trajectory parsing from response.
- `test_extract_answer` (line 595)
  > Test answer extraction from response.

---

### tests/test_sft_curriculum.py <a id='tests-test_sft_curriculum-py'></a>

**Module**: `tests.test_sft_curriculum`
**Total Tests**: 4

#### Standalone Tests:

- `test_curriculum_dataset` (line 25)
  > Test the CurriculumDataset class.
- `test_curriculum_manager` (line 100)
  > Test the CurriculumManager class.
- `test_curriculum_callback` (line 197)
  > Test the CurriculumCallback class.
- `test_integration` (line 247)
  > Test integration of all components.

---

### tests/unit/test_artifact_manager.py <a id='tests-unit-test_artifact_manager-py'></a>

**Module**: `tests.unit.test_artifact_manager`
**Total Tests**: 12

#### Test Classes:

**`TestArtifactManager`** (12 tests)

- `test_singleton_pattern` (line 31)
  > Test that ArtifactManager follows singleton pattern.
- `test_init_run` (line 37)
  > Test run initialization.
- `test_log_artifact` (line 47)
  > Test artifact logging.
- `test_log_large_artifact` (line 65)
  > Test large artifact logging with file.
- `test_use_artifact` (line 86)
  > Test artifact retrieval.
- `test_artifact_versioning` (line 109)
  > Test that artifacts get versioned correctly.
- `test_artifact_lineage` (line 123)
  > Test artifact lineage tracking.
- `test_content_addressable_storage` (line 146)
  > Test that identical content produces same hash.
- `test_wandb_integration` (line 163)
  > Test WandB integration when online.
- `test_list_artifacts` (line 186)
  > Test listing all artifacts.
- `test_thread_safety` (line 211)
  > Test thread-safe singleton access.
- `test_all_artifact_types` (line 230)
  > Test logging all artifact types.

---

## Test Organization by Category

### Unit Tests (12 tests)

- `tests/unit/test_artifact_manager.py` (12 tests)

### Integration Tests (7 tests)

- `tests/test_integration.py` (7 tests)

### Engine Tests (75 tests)

- `tests/engine/test_async_communication.py` (15 tests)
- `tests/engine/test_inference_engine.py` (14 tests)
- `tests/engine/test_ipc.py` (19 tests)
- `tests/engine/test_update_worker.py` (27 tests)

### Module Tests (46 tests)

- `tests/modules/test_experience_buffer.py` (20 tests)
- `tests/modules/test_model_init.py` (12 tests)
- `tests/modules/test_voting.py` (14 tests)

### Training Tests (30 tests)

- `tests/test_rft_training.py` (26 tests)
- `tests/test_sft_curriculum.py` (4 tests)

### Protocol Tests (14 tests)

- `tests/test_experimental_protocol.py` (14 tests)

## PyTest Commands

### Run All Tests
```bash
pytest tests/
```

### Run Individual Test Files
```bash
pytest tests/engine/test_async_communication.py
pytest tests/engine/test_inference_engine.py
pytest tests/engine/test_ipc.py
pytest tests/engine/test_update_worker.py
pytest tests/modules/test_experience_buffer.py
pytest tests/modules/test_model_init.py
pytest tests/modules/test_voting.py
pytest tests/test_basic.py
pytest tests/test_experimental_protocol.py
pytest tests/test_integration.py
pytest tests/test_rft_training.py
pytest tests/test_sft_curriculum.py
pytest tests/unit/test_artifact_manager.py
```

### Run Specific Test Classes
```bash
pytest tests/engine/test_async_communication.py::TestSharedMemoryManager
pytest tests/engine/test_async_communication.py::TestAsyncCommunication
pytest tests/engine/test_async_communication.py::TestFaultTolerance
pytest tests/engine/test_inference_engine.py::TestSharedMemoryManager
pytest tests/engine/test_inference_engine.py::TestInferenceEngine
pytest tests/engine/test_inference_engine.py::TestIntegration
pytest tests/engine/test_ipc.py::TestSharedMemoryTransfer
pytest tests/engine/test_ipc.py::TestQueueCommunication
pytest tests/engine/test_ipc.py::TestProcessCommunication
pytest tests/engine/test_ipc.py::TestCleanupMechanisms
pytest tests/engine/test_ipc.py::TestEdgeCases
pytest tests/engine/test_update_worker.py::TestKLConfig
pytest tests/engine/test_update_worker.py::TestSharedMemoryReconstructor
pytest tests/engine/test_update_worker.py::TestUpdateWorker
pytest tests/engine/test_update_worker.py::TestIntegration
pytest tests/modules/test_experience_buffer.py::TestExperienceBuffer
pytest tests/modules/test_experience_buffer.py::TestPersistenceAdapters
pytest tests/modules/test_model_init.py::TestDynamicLoRAConfig
pytest tests/modules/test_model_init.py::TestLoRAInsertion
pytest tests/modules/test_model_init.py::TestPerformanceAssertions
pytest tests/modules/test_model_init.py::TestSVDArtifactPersistence
pytest tests/modules/test_voting.py::TestTemporalEnsembleVoting
pytest tests/test_experimental_protocol.py::TestExperimentalProtocol
pytest tests/test_experimental_protocol.py::TestStatisticalMethods
pytest tests/test_experimental_protocol.py::TestExperimentReproducibility
pytest tests/test_rft_training.py::TestLightweightDynamicsModel
pytest tests/test_rft_training.py::TestEnhancedCuriosityModule
pytest tests/test_rft_training.py::TestEnhancedCoherenceAnalyzer
pytest tests/test_rft_training.py::TestToolMisusePenaltyCalculator
pytest tests/test_rft_training.py::TestEnhancedRewardOrchestrator
pytest tests/test_rft_training.py::TestGRPOTrainer
pytest tests/test_rft_training.py::TestUtilityFunctions
pytest tests/test_rft_training.py::TestIntegration
pytest tests/unit/test_artifact_manager.py::TestArtifactManager
```

### Run Specific Test Functions
```bash
pytest tests/engine/test_async_communication.py::TestSharedMemoryManager::test_create_shared_tensor
pytest tests/engine/test_async_communication.py::TestAsyncCommunication::test_inference_engine_initialization
pytest tests/engine/test_async_communication.py::TestFaultTolerance::test_watchdog_cleanup_on_timeout
pytest tests/engine/test_inference_engine.py::TestSharedMemoryManager::test_create_shared_tensor
pytest tests/engine/test_inference_engine.py::TestInferenceEngine::test_initialization
# ... (use same pattern for other tests)
```
