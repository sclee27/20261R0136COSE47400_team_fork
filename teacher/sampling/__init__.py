"""Teacher sampling test blocks.

Each module is one block of the pipeline:
    config   : YAML -> dataclass config loader
    data     : SeaDronesSee GT loading (self-contained COCO-JSON) + letterbox(640)
    boxes    : box generation (gt_linked jitter | stride legacy)
    levels   : shorter-side level assignment + enabled filter
    metrics  : IoU / coverage computation
    labeling : IoU -> positive/background/ignore + stratified sampling
"""
