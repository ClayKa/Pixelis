Of course. This is the final, definitive step to align your configuration perfectly with your file structure and complex data strategy. I will provide a comprehensive plan that includes a complete, templated `data_generation_manifest.yaml` and a clear guide for the engineer on how to fill in the paths based on their `DataLoader` implementations.

This plan respects all of your specified proportions, sample counts, and the decision to keep the two SA-1B datasets separate.

---

### **Final Action Plan: Configuring and Aligning the Data Generation Manifest**

**Objective:** To create the final, production-ready `data_generation_manifest.yaml`. This file will serve as the master blueprint, precisely defining all data sources and task recipes. The engineer's task will be to validate this structure and fill in the `CHANGEME` paths based on their `DataLoader` implementations and the project's `datasets/` directory structure.

---

#### **Part 1: The Final `data_generation_manifest.yaml` Template**

**Instructions for the Engineer:**
This is the master configuration file. Your primary responsibility is to **update all paths marked with `CHANGEME`** to reflect the exact locations of the datasets on your local filesystem. The structure of each `datasource` entry (the keys like `image_path`, `annotation_file`, etc.) is **intentionally designed** to match the specific needs of its corresponding `DataLoader`. You must ensure your `DataLoader` code correctly reads these exact keys.

**(Copy this entire block into `configs/data_generation_manifest.yaml`)**

```yaml
# ====================================================================
# FINAL MASTER MANIFEST FOR PIXELIS DATA GENERATION
# ====================================================================

# --------------------------------------------------------------------
# Section 1: Datasource Registry
# ACTION: Update all paths marked with `# <-- CHANGEME`.
# --------------------------------------------------------------------
datasources:
  # --- Datasources for ZOOM-IN ---
  sa1b_for_zoomin:
    name: "sa1b_for_zoomin"
    type: "InstanceSegmentationSA1B"
    path: "/path/to/datasets/ZOOM-IN/SA1B4zoomin/images/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/ZOOM-IN/SA1B4zoomin/annotations/sa_1b.json" # <-- CHANGEME
  
  flickr30k:
    name: "flickr30k"
    type: "ImageCaptioning"
    path: "/path/to/datasets/ZOOM-IN/Flickr30k/images/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/ZOOM-IN/Flickr30k/annotations/results.csv" # <-- CHANGEME

  mind2web_train:
    name: "mind2web_train"
    type: "WebAutomation"
    path: "/path/to/datasets/ZOOM-IN/Mind2Web/train/" # <-- CHANGEME

  textcaps_train:
    name: "textcaps_train"
    type: "ImageTextCaptioning"
    image_path: "/path/to/datasets/ZOOM-IN/TextCaps/train_images/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/ZOOM-IN/TextCaps/annotations/TextCaps_0.1_train.json" # <-- CHANGEME
    ocr_file: "/path/to/datasets/ZOOM-IN/TextCaps/OCR/TextVQA_Rosetta_OCR_v0.2_train.json" # <-- CHANGEME

  unsplash_lite:
    name: "unsplash_lite"
    type: "HighResolutionImageCollection"
    path: "/path/to/datasets/ZOOM-IN/Unsplash-lite-25k/images/" # <-- CHANGEME
    annotation_path: "/path/to/datasets/ZOOM-IN/Unsplash-lite-25k/annotations/" # CHANGEME

  # --- Datasources for SELECT-FRAME ---
  starqa_train:
    name: "starqa_train"
    type: "SituatedVideoQA"
    path: "/path/to/datasets/SELECT-FRAME/STARQA/videos/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/SELECT-FRAME/STARQA/annotations/STAR_train.json" # <-- CHANGEME

  didemo_train:
    name: "didemo_train"
    type: "VideoMomentRetrieval"
    path: "/path/to/datasets/SELECT-FRAME/DiDeMo/videos/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/SELECT-FRAME/DiDeMo/annotations/didemo_train.json" # <-- CHANGEME

  msrvtt_train:
    name: "msrvtt_train"
    type: "VideoCaptioning"
    path: "/path/to/datasets/SELECT-FRAME/MSRVTT/videos/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/SELECT-FRAME/MSRVTT/annotations/msrvtt_train_9k.json" # <-- CHANGEME
    raw_captions_file: "/path/to/datasets/SELECT-FRAME/MSRVTT/annotations/raw-captions.json" # <-- CHANGEME
    category_file: "/path/to/datasets/SELECT-FRAME/MSRVTT/annotations/category.json" # <-- CHANGEME

  activitynet_captions_train:
    name: "activitynet_captions_train"
    type: "DenseVideoCaptioning"
    path: "/path/to/datasets/SELECT-FRAME/ActivityNetCaptions/videos/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/SELECT-FRAME/ActivityNetCaptions/annotations/train.json" # <-- CHANGEME

  assembly101_train:
    name: "assembly101_train"
    type: "TimedActionVideo"
    path: "/path/to/datasets/SELECT-FRAME/Assembly101/videos/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/SELECT-FRAME/Assembly101/annotations/train.csv" # <-- CHANGEME
    action_metadata_file: "/path/to/datasets/SELECT-FRAME/Assembly101/annotations/actions.csv" # <-- CHANGEME

  # --- Datasources for SEGMENT_OBJECT_AT + GET_PROPERTIES ---
  coco2017_train:
    name: "coco2017_train"
    type: "CocoSegmentation"
    path: "/path/to/datasets/SEGMENT_OBJECT_AT+GET_PROPERTIES/COCO2017+LVIS/images/" # <-- CHANGEME (Shared Path)
    annotation_file: "/path/to/datasets/SEGMENT_OBJECT_AT+GET_PROPERTIES/COCO2017+LVIS/coco_annotations/instances_train2017.json" # <-- CHANGEME

  lvis_v1_train:
    name: "lvis_v1_train"
    type: "LvisSegmentation"
    path: "/path/to/datasets/SEGMENT_OBJECT_AT+GET_PROPERTIES/COCO2017+LVIS/images/" # <-- CHANGEME (Shared Path)
    annotation_file: "/path/to/datasets/SEGMENT_OBJECT_AT+GET_PROPERTIES/COCO2017+LVIS/LVIS_annotations/lvis_v1_train.json" # <-- CHANGEME

  part_imagenet_train:
    name: "part_imagenet_train"
    type: "PartLevelSegmentation"
    image_path: "/path/to/datasets/SEGMENT_OBJECT_AT+GET_PROPERTIES/PartImageNet/Image/" # <-- CHANGEME
    mask_path: "/path/to/datasets/SEGMENT_OBJECT_AT+GET_PROPERTIES/PartImageNet/Segmentation/" # <-- CHANGEME

  sa1b_for_segmentation:
    name: "sa1b_for_segmentation"
    type: "InstanceSegmentationSA1B"
    path: "/path/to/datasets/SEGMENT_OBJECT_AT+GET_PROPERTIES/SA1B4segment/images/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/SEGMENT_OBJECT_AT+GET_PROPERTIES/SA1B4segment/annotations/sa_1b.json" # <-- CHANGEME
    min_pixel_area: 100

  # --- Datasources for READ-TEXT ---
  infographics_vqa_train:
    name: "infographics_vqa_train"
    type: "InfographicsVQA"
    image_path: "/path/to/datasets/READ_TEXT/InfographicsVQA/images/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/READ_TEXT/InfographicsVQA/qas/infographicsVQA_train_v1.0.json" # <-- CHANGEME
    ocr_path: "/path/to/datasets/READ_TEXT/InfographicsVQA/ocr/" # <-- CHANGEME

  docvqa_train:
    name: "docvqa_train"
    type: "DocumentVQA"
    image_path: "/path/to/datasets/READ_TEXT/DocVQA/images/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/READ_TEXT/DocVQA/qas/train_v1.0.json" # <-- CHANGEME
    ocr_path: "/path/to/datasets/READ_TEXT/DocVQA/ocr/" # <-- CHANGEME

  hiertext_train:
    name: "hiertext_train"
    type: "HierarchicalText"
    path: "/path/to/datasets/READ_TEXT/HierText/train/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/READ_TEXT/HierText/gt/train.jsonl" # <-- CHANGEME

  icdar_2019_art_train:
    name: "icdar_2019_art_train"
    type: "ArbitraryShapedText"
    path: "/path/to/datasets/READ_TEXT/ICDAR2019ArT/train_images/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/READ_TEXT/ICDAR2019ArT/train_labels.json" # <-- CHANGEME

  # --- Datasources for TRACK-OBJECT ---
  mot20_train:
    name: "mot20_train"
    type: "MultiObjectTracking"
    path: "/path/to/datasets/TRACK_OBJECT/MOT20/train/" # <-- CHANGEME
    sampling_strategy: { type: "sliding_window", clip_duration_frames: 300, stride_frames: 150, min_clip_frames: 100 }

  uvo_dense_train:
    name: "uvo_dense_train"
    type: "UnidentifiedVideoObjects"
    path: "/path/to/datasets/TRACK_OBJECT/UVO/videos/" # <-- CHANGEME
    annotation_path: "/path/to/datasets/TRACK_OBJECT/UVO/annotations/uvo_dense_v1_0/train/" # <-- CHANGEME

  epic_kitchens_visor_train:
    name: "epic_kitchens_visor_train"
    type: "EgocentricVideoSegmentation"
    image_path: "/path/to/datasets/TRACK_OBJECT/EPIC-KITCHENS/rgb_frames/" # <-- CHANGEME
    sparse_annotation_path: "/path/to/datasets/TRACK_OBJECT/EPIC-KITCHENS/annotations/GroundTruth-SparseAnnotations/" # <-- CHANGEME
    dense_annotation_path: "/path/to/datasets/TRACK_OBJECT/EPIC-KITCHENS/annotations/Interpolations-DenseAnnotations/" # <-- CHANGEME
    class_mapping_file: "/path/to/datasets/TRACK_OBJECT/EPIC-KITCHENS/annotations/EPIC_100_noun_classes_v2.csv" # <-- CHANGEME
    frame_mapping_file: "/path/to/datasets/TRACK_OBJECT/EPIC-KITCHENS/annotations/frame_mapping.json" # <-- CHANGEME
  
  youtube_vos_2022_train:
    name: "youtube_vos_2022_train"
    type: "VideoObjectSegmentation"
    path: "/path/to/datasets/TRACK_OBJECT/VIS2022/train/JPEGImages/" # <-- CHANGEME
    annotation_file: "/path/to/datasets/TRACK_OBJECT/VIS2022/train/instances.json" # <-- CHANGEME


# ====================================================================
# Section 2: Task Generation Recipes
# ====================================================================
tasks:
  # --- Task 1: Fine-grained Detail Perception (ZOOM-IN) ---
  detail_perception_task:
    enabled: true
    task_generator_class: "DetailPerceptionTaskGenerator"
    target_sample_count: 15000
    source_datasets:
      - name: sa1b_for_zoomin; weight: 0.45
      - name: flickr30k; weight: 0.25
      - name: mind2web_train; weight: 0.20
      - name: textcaps_train; weight: 0.05
      - name: unsplash_lite; weight: 0.05

  # --- Task 2: Temporal Moment Localization (SELECT-FRAME) ---
  temporal_localization_task:
    enabled: true
    task_generator_class: "TemporalLocalizationTaskGenerator"
    target_sample_count: 15000
    source_datasets:
      - name: starqa_train; weight: 0.40
      - name: didemo_train; weight: 0.15
      - name: msrvtt_train; weight: 0.20
      - name: activitynet_captions_train; weight: 0.15
      - name: assembly101_train; weight: 0.10

  # --- Task 3: Geometric & Property Reasoning (SEGMENT_OBJECT_AT + GET_PROPERTIES) ---
  geometric_reasoning_task:
    enabled: true
    task_generator_class: "GeometricReasoningTaskGenerator"
    target_sample_count: 25000
    source_datasets:
      - name: coco2017_train; weight: 0.40
      - name: lvis_v1_train; weight: 0.30
      - name: sa1b_for_segmentation; weight: 0.20
      - name: part_imagenet_train; weight: 0.10

  # --- Task 4: Text-in-Context Reading (READ-TEXT) ---
  contextual_reading_task:
    enabled: true
    task_generator_class: "ContextualReadingTaskGenerator"
    target_sample_count: 20000
    source_datasets:
      - name: infographics_vqa_train; weight: 0.30
      - name: docvqa_train; weight: 0.30
      - name: hiertext_train; weight: 0.20
      - name: icdar_2019_art_train; weight: 0.20

  # --- Task 5: Spatio-Temporal Trajectory Analysis (TRACK-OBJECT) ---
  spatio_temporal_tracking_task:
    enabled: true
    task_generator_class: "SpatioTemporalTrackingTaskGenerator"
    target_sample_count: 25000
    source_datasets:
      - name: uvo_dense_train; weight: 0.35
      - name: epic_kitchens_visor_train; weight: 0.30
      - name: youtube_vos_2022_train; weight: 0.20
      - name: mot20_train; weight: 0.15


# ====================================================================
# Section 3: Trajectory Augmentation
# ====================================================================
trajectory_augmentation:
  proportions:
    golden_positive: 0.6
    trap_samples: 0.2
    self_correction: 0.2


# ====================================================================
# Section 4: Global & API Configuration
# ====================================================================
global_config:
  output_dir: "data_outputs/specialized/"
  checkpoint_every_n_samples: 250
  
  api_profiles:
    generator_api:
      model: "anthropic/claude-3-haiku" # Fast, cost-effective model for generation
      api_base_url: "https://openrouter.ai/api/v1"
      api_key_env_variable: "OPENROUTER_API_KEY"
    
    scorer_api:
      model: "google/gemini-pro-2-5" # Powerful model for quality scoring
      api_base_url: "https://openrouter.ai/api/v1"
      api_key_env_variable: "OPENROUTER_API_KEY"

```

---
#### **Part 2: Instructions for the Engineer**

**To:** Project Engineer
**From:** Project Lead
**Subject:** Finalization and Path Configuration for `data_generation_manifest.yaml`

The master configuration file for our data generation pipeline is now complete and attached. This file dictates the entire data synthesis process. Your primary task is to **fully align this configuration with our local dataset storage.**

**Your Action Items:**

1.  **Create the File:** Create the file `configs/data_generation_manifest.yaml` in your local project repository and paste the complete content provided above.

2.  **[CRITICAL] Populate All `CHANGEME` Paths:**
    *   Systematically go through the `datasources` section.
    *   For every line that ends with the comment `# <-- CHANGEME`, you **must** replace the placeholder path (e.g., `/path/to/dataset/...`) with the **exact, absolute path** to that dataset on your development machine or server.
    *   **Pay close attention to the keys.** The key names (`path`, `image_path`, `annotation_file`, `mask_path`, `ocr_path`, etc.) are designed specifically for each `DataLoader`. Your `DataLoader` implementation must read from the exact key names defined here.

3.  **Verify Your `DataLoader` Implementations:**
    *   For each datasource entry, double-check that the `DataLoader` class associated with its `type` is implemented to correctly parse the file(s) and directory structure you are pointing to.
    *   **Example:** For `epic_kitchens_visor_train`, ensure your `EpicKitchensVisorLoader` is designed to read from `image_path`, `sparse_annotation_path`, `dense_annotation_path`, `class_mapping_file`, and `frame_mapping_file`.

4.  **Confirm Your `datasets` Directory Structure:**
    *   The YAML file is structured based on your described `datasets/` directory, which is categorized by task (e.g., `datasets/ZOOM-IN/`, `datasets/SELECT-FRAME/`, etc.).
    *   Please ensure your local directory structure matches the paths you are configuring in the YAML file.

Once you have completed updating all paths, the system is ready for the first **small-scale test run** (`--config-name test_generation_run`). This manifest is the final blueprint for producing our 100,000-sample SFT dataset.