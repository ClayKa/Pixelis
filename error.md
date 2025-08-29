============================= test session starts ==============================
platform linux -- Python 3.10.18, pytest-8.4.1, pluggy-1.6.0
rootdir: /mnt/c/Users/ClayKa/Pixelis
configfile: pyproject.toml
testpaths: tests
plugins: mock-3.14.1, hydra-core-1.3.2, cov-6.2.1, asyncio-1.1.0, anyio-4.10.0
asyncio: mode=auto, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 1553 items

tests/data_generation/test_augmenter.py ...........                      [  0%]
tests/dataloaders/test_activitynet_captions_loader.py .........          [  1%]
tests/dataloaders/test_assembly101_loader.py ...............             [  2%]
tests/dataloaders/test_base_loader.py ............                       [  3%]
tests/dataloaders/test_coco_segment_loader.py ..................         [  4%]
tests/dataloaders/test_didemo_loader.py ...........                      [  4%]
tests/dataloaders/test_docvqa_loader.py ..........                       [  5%]
tests/dataloaders/test_edge_cases.py ...........                         [  6%]
tests/dataloaders/test_epic_kitchens_loader.py ......................... [  7%]
..                                                                       [  7%]
tests/dataloaders/test_flickr30k_loader.py .........                     [  8%]
tests/dataloaders/test_hiertext_loader.py ............                   [  9%]
tests/dataloaders/test_icdar_art_loader.py ......                        [  9%]
tests/dataloaders/test_infographics_vqa_loader.py .........              [ 10%]
tests/dataloaders/test_lvis_segment_loader.py .......................    [ 11%]
tests/dataloaders/test_mind2web_loader.py ..............                 [ 12%]
tests/dataloaders/test_mot_loader.py ...........................         [ 14%]
tests/dataloaders/test_mot_sliding_window_loader.py .......              [ 14%]
tests/dataloaders/test_msrvtt_loader.py .............................    [ 16%]
tests/dataloaders/test_part_imagenet_loader.py ................          [ 17%]
tests/dataloaders/test_sa1b_loader.py ............                       [ 18%]
tests/dataloaders/test_sa1b_segment_loader.py ..................         [ 19%]
tests/dataloaders/test_starqa_loader.py ...........................      [ 21%]
tests/dataloaders/test_textcaps_loader.py ...........                    [ 22%]
tests/dataloaders/test_timestamp_utils.py ........................       [ 23%]
tests/dataloaders/test_unsplash_lite_loader.py ..............            [ 24%]
tests/dataloaders/test_uvo_loader.py ...............................     [ 26%]
tests/dataloaders/test_youtube_vos_loader.py ........................... [ 28%]
..                                                                       [ 28%]
tests/engine/test_async_communication.py ......................          [ 29%]
tests/engine/test_inference_engine.py .................................. [ 32%]
...........................................                              [ 34%]
tests/engine/test_ipc.py ...................                             [ 36%]
tests/engine/test_update_worker.py ..................................... [ 38%]
...