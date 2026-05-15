python scripts/1_generate_specialized_datasets.py --config-name test_generation_manifest
2025-08-31 14:08:40,049 - __main__ - INFO - ============================================================
2025-08-31 14:08:40,049 - __main__ - INFO - STAGE 1: SPECIALIZED DATASET GENERATION
2025-08-31 14:08:40,049 - __main__ - INFO - ============================================================
2025-08-31 14:08:40,049 - __main__ - INFO - Starting Stage 1 generation. Dry run: False
2025-08-31 14:08:40,050 - __main__ - INFO - Initializing all available dataloaders from manifest...
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'sa1b_for_zoomin'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'flickr30k'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'mind2web_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'textcaps_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'unsplash_lite'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'starqa_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'didemo_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'msrvtt_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'activitynet_captions_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'assembly101_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'coco2017_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'lvis_v1_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'part_imagenet_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'sa1b_for_segmentation'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'infographics_vqa_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'docvqa_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'hiertext_train'
2025-08-31 14:08:40,051 - __main__ - INFO - -> Successfully initialized loader for 'icdar_2019_art_train'
2025-08-31 14:08:40,052 - __main__ - INFO - -> Successfully initialized loader for 'mot20_train'
2025-08-31 14:08:40,052 - __main__ - INFO - -> Successfully initialized loader for 'uvo_dense_train'
2025-08-31 14:08:40,052 - __main__ - INFO - -> Successfully initialized loader for 'epic_kitchens_visor_train'
2025-08-31 14:08:40,052 - __main__ - INFO - -> Successfully initialized loader for 'vis2022_train'
2025-08-31 14:08:40,052 - __main__ - INFO - Total initialized loaders: 22
2025-08-31 14:08:40,052 - __main__ - INFO - TrajectoryAugmenter initialized with proportions: {'golden_positive': 0.6, 'trap_samples': 0.2, 'self_correction': 0.2}
2025-08-31 14:08:40,052 - __main__ - INFO - Processing 1 tasks...
2025-08-31 14:08:40,052 - __main__ - INFO - 
============================================================
2025-08-31 14:08:40,052 - __main__ - INFO - Processing task: detail_perception_task
2025-08-31 14:08:40,052 - __main__ - INFO - ============================================================
2025-08-31 14:08:40,052 - __main__ - INFO - Task 'detail_perception_task' requires these datasources: ['sa1b_for_zoomin', 'flickr30k', 'mind2web_train', 'textcaps_
train', 'unsplash_lite']                                                                                                                                           2025-08-31 14:08:40,052 - __main__ - INFO - Injecting 5 loaders into 'detail_perception_task'
2025-08-31 14:08:40,052 - __main__ - INFO - Preparing configuration for DetailPerceptionTaskGenerator task 'detail_perception_task'
2025-08-31 14:08:40,052 - __main__ - INFO - Target: 10 unique samples for task 'detail_perception_task'...
2025-08-31 14:08:40,054 - __main__ - INFO - Loaded checkpoint: 0 unique samples from 0 total
2025-08-31 14:08:40,055 - __main__ - INFO - Generation round 1: Generating 10 samples (need 10 more unique samples)...
2025-08-31 14:08:40,055 - core.data_generation.base_generator - INFO - Validation strictness set to: strict
2025-08-31 14:08:40,063 - core.data_generation.base_generator - INFO - Loaded prompt template from prompts/detail_perception.md
2025-08-31 14:08:40,063 - core.data_generation.base_generator - INFO - API Key 'OPENROUTER_API_KEY' found and loaded successfully.
2025-08-31 14:08:40,101 - core.data_generation.base_generator - INFO - Initialized API client for 'detail_perception_task' with base URL: https://openrouter.ai/api
/v1                                                                                                                                                                2025-08-31 14:08:40,101 - core.data_generation.detail_perception - INFO - --- Parsing Style Cookbook ---
2025-08-31 14:08:40,102 - core.data_generation.detail_perception - INFO - Pattern 1 found 40 potential style blocks.
2025-08-31 14:08:40,105 - core.data_generation.detail_perception - INFO - Total styles parsed: 40
2025-08-31 14:08:40,106 - core.data_generation.detail_perception - INFO - Parsed 40 creative styles from prompt template
Generating NEW 'detail_perception_task':   0%|                                                                                              | 0/10 [00:00<?, ?it/s]
2025-08-31 14:08:40,112 - core.data_generation.detail_perception - INFO - === Building V4 Context Block for DetailPerceptionTask ===                               2025-08-31 14:08:40,112 - core.data_generation.detail_perception - INFO - Selected Difficulty: Medium
2025-08-31 14:08:40,112 - core.data_generation.detail_perception - INFO - Using loader: Textcaps Train
2025-08-31 14:08:40,112 - core.data_generation.detail_perception - INFO - Selected Style: 'The Direct Inquirer', Difficulty: 'Medium'
2025-08-31 14:08:40,112 - core.data_generation.detail_perception - INFO - ✓ Successfully constructed placeholders and metadata
2025-08-31 14:08:40,113 - core.data_generation.base_generator - INFO - --- Preparing to call LLM API ---
2025-08-31 14:08:44,194 - httpx - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2025-08-31 14:08:46,268 - core.data_generation.detail_perception - WARNING - Validation failed: ZOOM-IN action is missing a valid 'parameters' dictionary. Got: {}
2025-08-31 14:08:46,268 - core.data_generation.detail_perception - INFO - === Building V4 Context Block for DetailPerceptionTask ===
2025-08-31 14:08:46,268 - core.data_generation.detail_perception - INFO - Selected Difficulty: Medium
2025-08-31 14:08:46,268 - core.data_generation.detail_perception - INFO - Using loader: Mind2Web Train
2025-08-31 14:08:46,269 - core.data_generation.detail_perception - INFO - Selected Style: 'The AI Trainer', Difficulty: 'Medium'
2025-08-31 14:08:46,269 - core.data_generation.detail_perception - INFO - ✓ Successfully constructed placeholders and metadata
2025-08-31 14:08:46,269 - core.data_generation.base_generator - INFO - --- Preparing to call LLM API ---
2025-08-31 14:08:47,764 - httpx - INFO - HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2025-08-31 14:08:50,147 - core.data_generation.detail_perception - WARNING - Validation failed: ZOOM-IN action is missing a valid 'parameters' dictionary. Got: {}
2025-08-31 14:08:50,147 - core.data_generation.detail_perception - INFO - === Building V4 Context Block for DetailPerceptionTask ===
2025-08-31 14:08:50,147 - core.data_generation.detail_perception - INFO - Selected Difficulty: Easy
2025-08-31 14:08:50,147 - core.data_generation.detail_perception - INFO - Using loader: Sa1B For Zoomin
2025-08-31 14:08:50,147 - core.data_generation.detail_perception - INFO - Selected Style: 'The Translator', Difficulty: 'Easy'
2025-08-31 14:08:50,147 - core.data_generation.detail_perception - INFO - ✓ Successfully constructed placeholders and metadata
2025-08-31 14:08:50,148 - core.data_generation.base_generator - INFO - --- Preparing to call LLM API ---
^C2025-08-31 14:08:52,264 - core.data_generation.base_generator - INFO - Generation interrupted by user.
Generating NEW 'detail_perception_task':   0%|                                                                                              | 0/10 [00:12<?, ?it/s]
Traceback (most recent call last):
  File "/mnt/c/Users/ClayKa/Pixelis/scripts/1_generate_specialized_datasets.py", line 681, in <module>
    main()
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/hydra/_internal/hydra.py", line 119, in run
    ret = run_job(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
  File "/mnt/c/Users/ClayKa/Pixelis/scripts/1_generate_specialized_datasets.py", line 674, in main
    generator.generate_all_datasets(tasks=task_list)
  File "/mnt/c/Users/ClayKa/Pixelis/scripts/1_generate_specialized_datasets.py", line 551, in generate_all_datasets
    output_file = self.generate_task_dataset(task_name, task_config)
  File "/mnt/c/Users/ClayKa/Pixelis/scripts/1_generate_specialized_datasets.py", line 457, in generate_task_dataset
    golden_samples = generator.generate(num_samples=samples_to_generate)
  File "/mnt/c/Users/ClayKa/Pixelis/core/data_generation/base_generator.py", line 560, in generate
    llm_response = self._call_llm_api(final_prompt, context_placeholders=context_placeholders)
  File "/mnt/c/Users/ClayKa/Pixelis/core/data_generation/base_generator.py", line 346, in _call_llm_api
    response = self.api_client.chat.completions.create(**payload)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/openai/_utils/_utils.py", line 287, in wrapper
    return func(*args, **kwargs)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/openai/resources/chat/completions/completions.py", line 1147, in create
    return self._post(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/openai/_base_client.py", line 1259, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/openai/_base_client.py", line 982, in request
    response = self._client.send(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpcore/_sync/http_proxy.py", line 343, in handle_request
    return self._connection.handle_request(request)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpcore/_sync/http11.py", line 217, in _receive_event
    data = self._network_stream.read(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/httpcore/_backends/sync.py", line 128, in read
    return self._sock.recv(max_bytes)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/ssl.py", line 1292, in recv
    return self.read(buflen)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/ssl.py", line 1165, in read
    return self._sslobj.read(len)
KeyboardInterrupt

