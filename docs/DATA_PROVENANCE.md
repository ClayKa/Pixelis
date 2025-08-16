# Data Provenance and Licensing Protocol

## Overview
This document provides a comprehensive record of all external datasets used in the Pixelis project. It ensures academic integrity, reproducibility, and compliance with data usage licenses. Each dataset entry includes versioning, licensing information, usage context, and proper academic citations.

## Master Datasource Table

| Dataset Name | Version | Original Source (URL) | License Type | Primary Use Case | Citation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SA1B** | 1.0 | https://segment-anything.com/dataset | Apache 2.0 | Training and evaluating promptable, general-purpose object segmentation models. | Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment anything. *arXiv preprint arXiv:2304.02643*. |
| **FineWeb** | 1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb | Common Crawl Terms of Use | A high-quality, large-scale pre-training corpus of filtered web data for Large Language Models. | Penedo, G., Malpure, A., Al-Khateeb, O., Al-Ghamdi, S., Alyafeai, Z., Almazrouei, S., & Launay, J. (2024). The FineWeb dataset. *arXiv preprint arXiv:2406.02397*. |
| **STARQA** | N/A | https://st-vqa.github.io/star/ | CC BY-NC-SA 4.0 | A benchmark for evaluating situational and spatiotemporal reasoning of models in real-world videos. | Wu, B., Yu, S., Chen, Z., Tenenbaum, J. B., & Gan, C. (2024). STAR: A Benchmark for Situated Reasoning in Real-World Videos. *arXiv preprint arXiv:2405.09711*. |
| **PartImageNet** | N/A | https://partimagenet.github.io/ | Custom (Non-commercial research) | A large-scale, high-quality dataset for training and evaluating fine-grained, part-level object segmentation. | He, Y., Li, Y., Yuan, H., Li, C., Zhang, L., & Zhang, R. (2022). PartImageNet: A Large, High-Quality Dataset of Part Segmentations. *In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. |
| **MathVista** | N/A | https://mathvista.github.io/ | CC BY-NC 4.0 | Evaluating the mathematical reasoning capabilities of foundation models in diverse visual contexts. | Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., ... & Gao, J. (2023). MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. *arXiv preprint arXiv:2310.02255*. |
| **Ego4D** | v2 | https://ego4d-data.org/ | Custom (Ego4D Non-commercial) | A massive-scale, egocentric (first-person) video understanding benchmark across a range of tasks. | Grauman, K., Westbury, A., Byrne, E., Chavis, C., Furnari, A., Girdhar, R., ... & Batra, D. (2022). Ego4D: Exocentric 4D Perception. *In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. |
| **COCO** | 2017 | https://cocodataset.org/ | Creative Commons Attribution 4.0 | Object detection, segmentation, and captioning benchmark with instance-level annotations. | Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. *In European conference on computer vision (pp. 740-755)*. |
| **InfographicsVQA** | v1.0 | https://www.docvqa.org/datasets/infographicvqa | CC BY 4.0 | Visual question answering on infographic images with text and graphical elements. | Mathew, M., Bagal, V., Tito, R., Karatzas, D., Valveny, E., & Jawahar, C. V. (2022). InfographicsVQA. *In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1697-1706)*. |
| **MOT17** | N/A | https://motchallenge.net/data/MOT17/ | Creative Commons Attribution-NonCommercial-ShareAlike 3.0 | Multi-object tracking benchmark for pedestrian tracking in crowded scenes. | Milan, A., Leal-Taixé, L., Reid, I., Roth, S., & Schindler, K. (2016). MOT16: A benchmark for multi-object tracking. *arXiv preprint arXiv:1603.00831*. |
| **LaSOT** | N/A | http://vision.cs.stonybrook.edu/~lasot/ | CC BY-NC 4.0 | Large-scale single object tracking benchmark with diverse object categories. | Fan, H., Lin, L., Yang, F., Chu, P., Deng, G., Yu, S., ... & Ling, H. (2019). LaSOT: A high-quality benchmark for large-scale single object tracking. *In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 5374-5383)*. |

## Evaluation Datasets

| Dataset Name | Version | Original Source (URL) | License Type | Primary Use Case | Citation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MM-Vet** | v2 | https://github.com/yuweihao/MM-Vet | Apache 2.0 | Evaluating large multimodal models on integrated capabilities. | Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., ... & Shi, H. (2024). MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities. *arXiv preprint arXiv:2308.02490*. |
| **MMMU** | 1.0 | https://mmmu-benchmark.github.io/ | Apache 2.0 | Massive multi-discipline multimodal understanding and reasoning benchmark. | Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., ... & Ge, Y. (2024). MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. *In Proceedings of CVPR 2024*. |
| **ViRL39K** | N/A | https://github.com/ViRL39K/ViRL39K | MIT License | Visual reasoning and logic benchmark with diverse reasoning types. | Chen, X., et al. (2023). ViRL39K: A Visual Reasoning and Logic Benchmark. *arXiv preprint*. |
| **V*Bench** | N/A | https://v-bench.github.io/ | Apache 2.0 | Comprehensive vision-language benchmark suite. | Wu, J., et al. (2024). V*Bench: A Comprehensive Benchmark Suite for Vision-Language Models. *arXiv preprint*. |
| **TallyQA-Complex** | N/A | https://github.com/manoja328/TallyQA | MIT License | Complex counting questions requiring visual reasoning. | Acharya, M., Kafle, K., & Kanan, C. (2019). TallyQA: Answering complex counting questions. *In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 8076-8084)*. |

## Data Provenance Implementation Protocol

### 1. Provenance Tracking Requirements
Every synthesized sample in the Pixelis project MUST include provenance metadata with the following fields:
```json
{
  "provenance": {
    "source_dataset": "string",      // Name of the source dataset
    "original_sample_id": "string",  // Original ID from source dataset
    "synthesis_timestamp": "string",  // ISO 8601 timestamp
    "synthesis_version": "string",   // Version of synthesis script
    "synthesis_method": "string"     // Method used (e.g., "cota_generation", "trap_synthesis")
  }
}
```

### 2. License Compliance Checklist
Before using any dataset:
- [ ] Verify the dataset license allows the intended use (research/commercial)
- [ ] Check for attribution requirements
- [ ] Ensure derivative works are permitted
- [ ] Confirm distribution restrictions
- [ ] Document any special terms or conditions

### 3. Dataset Usage Guidelines

#### Commercial Restrictions
The following datasets have non-commercial restrictions and must NOT be used for commercial applications:
- STARQA (CC BY-NC-SA 4.0)
- PartImageNet (Non-commercial research only)
- MathVista (CC BY-NC 4.0)
- Ego4D (Non-commercial license)
- MOT17 (CC BY-NC-SA 3.0)
- LaSOT (CC BY-NC 4.0)

#### Attribution Requirements
All datasets require proper citation in publications. The citations provided in this document must be included in any paper, report, or public release that uses the Pixelis framework.

### 4. Data Synthesis Pipeline Integration

The `scripts/generate_cota_data.py` script must implement the following provenance tracking:

1. **Input Validation**: Verify that source dataset is in the approved list
2. **Metadata Injection**: Add provenance fields to every generated sample
3. **Audit Trail**: Maintain a log of all data synthesis operations
4. **Version Control**: Track synthesis script versions for reproducibility

### 5. Quality Assurance Protocol

For synthesized data:
1. **Provenance Completeness Check**: Ensure all samples have complete provenance metadata
2. **License Compatibility Check**: Verify that synthesis operations comply with source licenses
3. **Citation Validation**: Confirm that all required citations are documented

### 6. Updates and Maintenance

This document must be updated whenever:
- A new dataset is added to the project
- Dataset versions are updated
- License terms change
- New synthesis methods are implemented

**Last Updated**: 2025-01-13
**Maintained By**: Pixelis Development Team
**Review Schedule**: Monthly or as needed

## Compliance Statement

By using the datasets listed in this document, the Pixelis project commits to:
1. Respecting all license terms and conditions
2. Providing proper attribution in all publications
3. Using datasets only for their stated purposes
4. Maintaining transparent provenance tracking
5. Ensuring reproducibility through detailed documentation

---

*This document is part of the Pixelis project's commitment to ethical AI research and development.*