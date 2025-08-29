python scripts/1_generate_specialized_datasets_v2.py
Could not import dataloaders: cannot import name 'STARQALoader' from 'core.dataloaders.starqa_loader' (/mnt/c/Users/ClayKa/Pixelis/core/dataloaders/starqa_loa
der.py). Using mock loaders.                                                                                                                                  [2025-08-29 13:04:50,808][__main__][INFO] - ============================================================
[2025-08-29 13:04:50,809][__main__][INFO] - Starting Data Generation Pipeline (New Architecture)
[2025-08-29 13:04:50,810][__main__][INFO] - ============================================================
[2025-08-29 13:04:50,811][__main__][INFO] - GENERATION MODE - Creating Datasets
[2025-08-29 13:04:50,812][__main__][INFO] - ----------------------------------------
[2025-08-29 13:04:50,813][__main__][INFO] - 
Generating dataset for task: detail_perception_task
[2025-08-29 13:04:50,814][__main__][WARNING] - Could not import some dataloaders: cannot import name 'STARQALoader' from 'core.dataloaders.starqa_loader' (/mn
t/c/Users/ClayKa/Pixelis/core/dataloaders/starqa_loader.py). Using mock loaders.                                                                              [2025-08-29 13:04:50,816][__main__][INFO] - Loaded 5 dataloaders for detail_perception_task
[2025-08-29 13:04:50,863][core.data_generation.base_generator][INFO] - BaseGenerator initialized with config from configs/data_generation_manifest.yaml
[2025-08-29 13:04:50,863][core.data_generation.specialized_generator][INFO] - SpecializedGenerator initialized for task: detail_perception_task
[2025-08-29 13:04:50,864][core.data_generation.specialized_generator][INFO] - Generating 20000 samples for detail_perception_task
[2025-08-29 13:04:50,864][core.data_generation.specialized_generator][INFO] - Sampling plan: {'sa1b_shard_7': 9000, 'flickr30k': 5000, 'mind2web_train': 4000,
 'textcaps_train': 1000, 'unsplash_lite': 1000}                                                                                                               Generating detail_perception_task:   0%|                                                                                            | 0/20000 [00:00<?, ?it/s]
[2025-08-29 13:04:50,870][core.data_generation.specialized_generator][WARNING] - No loader available for source: sa1b_shard_7                                 [2025-08-29 13:04:50,870][core.data_generation.specialized_generator][WARNING] - No loader available for source: flickr30k
[2025-08-29 13:04:50,870][core.data_generation.specialized_generator][WARNING] - No loader available for source: mind2web_train
[2025-08-29 13:04:50,871][core.data_generation.specialized_generator][WARNING] - No loader available for source: textcaps_train
[2025-08-29 13:04:50,871][core.data_generation.specialized_generator][WARNING] - No loader available for source: unsplash_lite
Generating detail_perception_task: 100%|███████████████████████████████████████████████████████████████████████████| 20000/20000 [00:00<00:00, 9230422.54it/s]
[2025-08-29 13:04:50,872][__main__][WARNING] - ⚠️ No samples generated for detail_perception_task
[2025-08-29 13:04:50,872][__main__][INFO] - 
Generating dataset for task: temporal_localization_task
[2025-08-29 13:04:50,872][__main__][WARNING] - Could not import some dataloaders: cannot import name 'STARQALoader' from 'core.dataloaders.starqa_loader' (/mn
t/c/Users/ClayKa/Pixelis/core/dataloaders/starqa_loader.py). Using mock loaders.                                                                              [2025-08-29 13:04:50,873][__main__][INFO] - Loaded 5 dataloaders for temporal_localization_task
[2025-08-29 13:04:50,913][core.data_generation.base_generator][INFO] - BaseGenerator initialized with config from configs/data_generation_manifest.yaml
[2025-08-29 13:04:50,913][core.data_generation.specialized_generator][INFO] - SpecializedGenerator initialized for task: temporal_localization_task
[2025-08-29 13:04:50,913][core.data_generation.specialized_generator][INFO] - Generating 25000 samples for temporal_localization_task
[2025-08-29 13:04:50,913][core.data_generation.specialized_generator][INFO] - Sampling plan: {'starqa_train': 10000, 'didemo_train': 3750, 'msrvtt_train': 500
0, 'activitynet_captions_train': 3750, 'assembly101_train': 2500}                                                                                             Generating temporal_localization_task:   0%|                                                                                        | 0/25000 [00:00<?, ?it/s]
[2025-08-29 13:04:50,914][core.data_generation.specialized_generator][WARNING] - No loader available for source: starqa_train                                 [2025-08-29 13:04:50,914][core.data_generation.specialized_generator][WARNING] - No loader available for source: didemo_train
[2025-08-29 13:04:50,914][core.data_generation.specialized_generator][WARNING] - No loader available for source: msrvtt_train
[2025-08-29 13:04:50,914][core.data_generation.specialized_generator][WARNING] - No loader available for source: activitynet_captions_train
[2025-08-29 13:04:50,914][core.data_generation.specialized_generator][WARNING] - No loader available for source: assembly101_train
Generating temporal_localization_task: 100%|██████████████████████████████████████████████████████████████████████| 25000/25000 [00:00<00:00, 27428093.12it/s]
[2025-08-29 13:04:50,915][__main__][WARNING] - ⚠️ No samples generated for temporal_localization_task
[2025-08-29 13:04:50,915][__main__][INFO] - 
Generating dataset for task: geometric_reasoning_task
[2025-08-29 13:04:50,915][__main__][WARNING] - Could not import some dataloaders: cannot import name 'STARQALoader' from 'core.dataloaders.starqa_loader' (/mn
t/c/Users/ClayKa/Pixelis/core/dataloaders/starqa_loader.py). Using mock loaders.                                                                              [2025-08-29 13:04:50,916][__main__][INFO] - Loaded 4 dataloaders for geometric_reasoning_task
[2025-08-29 13:04:50,947][core.data_generation.base_generator][INFO] - BaseGenerator initialized with config from configs/data_generation_manifest.yaml
[2025-08-29 13:04:50,947][core.data_generation.specialized_generator][INFO] - SpecializedGenerator initialized for task: geometric_reasoning_task
[2025-08-29 13:04:50,948][core.data_generation.specialized_generator][INFO] - Generating 30000 samples for geometric_reasoning_task
[2025-08-29 13:04:50,948][core.data_generation.specialized_generator][INFO] - Sampling plan: {'coco2017_train': 12000, 'lvis_v1_train': 9000, 'sa1b_shard_7': 
6000, 'part_imagenet_train': 3000}                                                                                                                            Generating geometric_reasoning_task:   0%|                                                                                          | 0/30000 [00:00<?, ?it/s]
[2025-08-29 13:04:50,948][core.data_generation.specialized_generator][WARNING] - No loader available for source: coco2017_train                               [2025-08-29 13:04:50,948][core.data_generation.specialized_generator][WARNING] - No loader available for source: lvis_v1_train
[2025-08-29 13:04:50,949][core.data_generation.specialized_generator][WARNING] - No loader available for source: sa1b_shard_7
[2025-08-29 13:04:50,949][core.data_generation.specialized_generator][WARNING] - No loader available for source: part_imagenet_train
Generating geometric_reasoning_task: 100%|████████████████████████████████████████████████████████████████████████| 30000/30000 [00:00<00:00, 41789810.69it/s]
[2025-08-29 13:04:50,949][__main__][WARNING] - ⚠️ No samples generated for geometric_reasoning_task
[2025-08-29 13:04:50,949][__main__][INFO] - 
Generating dataset for task: contextual_reading_task
[2025-08-29 13:04:50,949][__main__][WARNING] - Could not import some dataloaders: cannot import name 'STARQALoader' from 'core.dataloaders.starqa_loader' (/mn
t/c/Users/ClayKa/Pixelis/core/dataloaders/starqa_loader.py). Using mock loaders.                                                                              [2025-08-29 13:04:50,950][__main__][INFO] - Loaded 4 dataloaders for contextual_reading_task
[2025-08-29 13:04:50,982][core.data_generation.base_generator][INFO] - BaseGenerator initialized with config from configs/data_generation_manifest.yaml
[2025-08-29 13:04:50,982][core.data_generation.specialized_generator][INFO] - SpecializedGenerator initialized for task: contextual_reading_task
[2025-08-29 13:04:50,982][core.data_generation.specialized_generator][INFO] - Generating 20000 samples for contextual_reading_task
[2025-08-29 13:04:50,982][core.data_generation.specialized_generator][INFO] - Sampling plan: {'infographics_vqa_train': 6000, 'docvqa_train': 6000, 'hiertext_
train': 4000, 'icdar_2019_art_train': 4000}                                                                                                                   Generating contextual_reading_task:   0%|                                                                                           | 0/20000 [00:00<?, ?it/s]
[2025-08-29 13:04:50,983][core.data_generation.specialized_generator][WARNING] - No loader available for source: infographics_vqa_train                       [2025-08-29 13:04:50,983][core.data_generation.specialized_generator][WARNING] - No loader available for source: docvqa_train
[2025-08-29 13:04:50,983][core.data_generation.specialized_generator][WARNING] - No loader available for source: hiertext_train
[2025-08-29 13:04:50,983][core.data_generation.specialized_generator][WARNING] - No loader available for source: icdar_2019_art_train
Generating contextual_reading_task: 100%|█████████████████████████████████████████████████████████████████████████| 20000/20000 [00:00<00:00, 29873960.11it/s]
[2025-08-29 13:04:50,983][__main__][WARNING] - ⚠️ No samples generated for contextual_reading_task
[2025-08-29 13:04:50,984][__main__][INFO] - 
Generating dataset for task: spatio_temporal_tracking_task
[2025-08-29 13:04:50,984][__main__][WARNING] - Could not import some dataloaders: cannot import name 'STARQALoader' from 'core.dataloaders.starqa_loader' (/mn
t/c/Users/ClayKa/Pixelis/core/dataloaders/starqa_loader.py). Using mock loaders.                                                                              [2025-08-29 13:04:50,984][__main__][INFO] - Loaded 5 dataloaders for spatio_temporal_tracking_task
[2025-08-29 13:04:51,015][core.data_generation.base_generator][INFO] - BaseGenerator initialized with config from configs/data_generation_manifest.yaml
[2025-08-29 13:04:51,016][core.data_generation.specialized_generator][INFO] - SpecializedGenerator initialized for task: spatio_temporal_tracking_task
[2025-08-29 13:04:51,016][core.data_generation.specialized_generator][INFO] - Generating 30000 samples for spatio_temporal_tracking_task
[2025-08-29 13:04:51,016][core.data_generation.specialized_generator][INFO] - Sampling plan: {'uvo_dense_train': 7500, 'epic_kitchens_visor_train': 9000, 'you
tube_vos_2022_train': 6000, 'mot20_train': 4500, 'uvo_sparse_train': 3000}                                                                                    Generating spatio_temporal_tracking_task:   0%|                                                                                     | 0/30000 [00:00<?, ?it/s]
[2025-08-29 13:04:51,017][core.data_generation.specialized_generator][WARNING] - No loader available for source: uvo_dense_train                              [2025-08-29 13:04:51,017][core.data_generation.specialized_generator][WARNING] - No loader available for source: epic_kitchens_visor_train
[2025-08-29 13:04:51,017][core.data_generation.specialized_generator][WARNING] - No loader available for source: youtube_vos_2022_train
[2025-08-29 13:04:51,017][core.data_generation.specialized_generator][WARNING] - No loader available for source: mot20_train
[2025-08-29 13:04:51,017][core.data_generation.specialized_generator][WARNING] - No loader available for source: uvo_sparse_train
Generating spatio_temporal_tracking_task: 100%|███████████████████████████████████████████████████████████████████| 30000/30000 [00:00<00:00, 31528218.49it/s]
[2025-08-29 13:04:51,018][__main__][WARNING] - ⚠️ No samples generated for spatio_temporal_tracking_task
[2025-08-29 13:04:51,018][__main__][INFO] - 
Generating dataset for task: zoom_in_replication_task
[2025-08-29 13:04:51,018][__main__][WARNING] - Could not import some dataloaders: cannot import name 'STARQALoader' from 'core.dataloaders.starqa_loader' (/mn
t/c/Users/ClayKa/Pixelis/core/dataloaders/starqa_loader.py). Using mock loaders.                                                                              [2025-08-29 13:04:51,018][__main__][INFO] - Loaded 1 dataloaders for zoom_in_replication_task
[2025-08-29 13:04:51,098][core.data_generation.base_generator][INFO] - BaseGenerator initialized with config from configs/data_generation_manifest.yaml
[2025-08-29 13:04:51,098][core.data_generation.specialized_generator][INFO] - SpecializedGenerator initialized for task: zoom_in_replication_task
[2025-08-29 13:04:51,099][core.data_generation.specialized_generator][INFO] - Generating 8000 samples for zoom_in_replication_task
[2025-08-29 13:04:51,099][core.data_generation.specialized_generator][INFO] - Sampling plan: {'sa1b_shard_7': 8000}
Generating zoom_in_replication_task:   0%|                                                                                           | 0/8000 [00:00<?, ?it/s]
[2025-08-29 13:04:51,100][core.data_generation.base_generator][WARNING] - Unresolved placeholders in prompt: ['{\n  "question": "string",\n  "trajectory": [\n    { "type": "thought", "content": "string" }', '{ "type": "action", "name": "string (ZOOM-IN or SELECT-FRAME)", "parameters": { ... }', '{ "type": "thought", "content": "string" }']                                                                                                                                     Generating zoom_in_replication_task:   0%|                                                           | 1/8000 [00:11<24:42:54, 11.12s/it, Success=1, Failed=0]
[2025-08-29 13:05:02,223][core.data_generation.base_generator][WARNING] - Unresolved placeholders in prompt: ['{\n  "question": "string",\n  "trajectory": [\n    { "type": "thought", "content": "string" }', '{ "type": "action", "name": "string (ZOOM-IN or SELECT-FRAME)", "parameters": { ... }', '{ "type": "thought", "content": "string" }']                                                                                                                                     Generating zoom_in_replication_task:   0%|                                                           | 2/8000 [00:24<27:49:38, 12.53s/it, Success=2, Failed=0]
[2025-08-29 13:05:15,730][core.data_generation.base_generator][WARNING] - Unresolved placeholders in prompt: ['{\n  "question": "string",\n  "trajectory": [\n    { "type": "thought", "content": "string" }', '{ "type": "action", "name": "string (ZOOM-IN or SELECT-FRAME)", "parameters": { ... }', '{ "type": "thought", "content": "string" }']                                                                                                                                     Generating zoom_in_replication_task:   0%|                                                           | 3/8000 [00:31<22:06:07,  9.95s/it, Success=3, Failed=0]
[2025-08-29 13:05:22,615][core.data_generation.base_generator][WARNING] - Unresolved placeholders in prompt: ['{\n  "question": "string",\n  "trajectory": [\n    { "type": "thought", "content": "string" }', '{ "type": "action", "name": "string (ZOOM-IN or SELECT-FRAME)", "parameters": { ... }', '{ "type": "thought", "content": "string" }']                                                                                                                                     Generating zoom_in_replication_task:   0%|                                                           | 3/8000 [00:39<29:30:40, 13.28s/it, Success=3, Failed=0]
Traceback (most recent call last):
  File "/mnt/c/Users/ClayKa/Pixelis/scripts/1_generate_specialized_datasets_v2.py", line 452, in <module>
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
  File "/mnt/c/Users/ClayKa/Pixelis/scripts/1_generate_specialized_datasets_v2.py", line 448, in main
    pipeline.run(dry_run=dry_run, specific_tasks=specific_tasks if specific_tasks else None)
  File "/mnt/c/Users/ClayKa/Pixelis/scripts/1_generate_specialized_datasets_v2.py", line 70, in run
    self._run_generation(specific_tasks)
  File "/mnt/c/Users/ClayKa/Pixelis/scripts/1_generate_specialized_datasets_v2.py", line 163, in _run_generation
    samples = generator.generate_dataset(num_samples)
  File "/mnt/c/Users/ClayKa/Pixelis/core/data_generation/specialized_generator.py", line 110, in generate_dataset
    source_samples = self._generate_from_source(source_name, count, pbar)
  File "/mnt/c/Users/ClayKa/Pixelis/core/data_generation/specialized_generator.py", line 205, in _generate_from_source
    api_response = self.generate(self.task_name, context)
  File "/mnt/c/Users/ClayKa/Pixelis/core/data_generation/base_generator.py", line 149, in generate
    response = self._call_api(formatted_prompt)
  File "/mnt/c/Users/ClayKa/Pixelis/core/data_generation/base_generator.py", line 239, in _call_api
    response = requests.post(
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/requests/api.py", line 115, in post
    return request("post", url, data=data, json=json, **kwargs)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/requests/api.py", line 59, in request
    return session.request(method=method, url=url, **kwargs)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/requests/sessions.py", line 746, in send
    r.content
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/requests/models.py", line 902, in content
    self._content = b"".join(self.iter_content(CONTENT_CHUNK_SIZE)) or b""
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/requests/models.py", line 820, in generate
    yield from self.raw.stream(chunk_size, decode_content=True)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/urllib3/response.py", line 1088, in stream
    yield from self.read_chunked(amt, decode_content=decode_content)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/urllib3/response.py", line 1248, in read_chunked
    self._update_chunk_length()
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/site-packages/urllib3/response.py", line 1167, in _update_chunk_length
    line = self._fp.fp.readline()  # type: ignore[union-attr]
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/socket.py", line 717, in readinto
    return self._sock.recv_into(b)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/ssl.py", line 1307, in recv_into
    return self.read(nbytes, buffer)
  File "/home/clayka/miniconda3/envs/pixelis/lib/python3.10/ssl.py", line 1163, in read
    return self._sslobj.read(len, buffer)
KeyboardInterrupt

