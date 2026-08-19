# ComfyUI MiniMax H3 Reference to Video + First Frame

Experimental custom node for ComfyUI MiniMax H3.

This node extends ComfyUI's built-in **MiniMax H3 Reference to Video** node by adding an independent `first_frame` input.

The first frame is supplied in an I2V-style manner:

- to the tokenizer as image conditioning
- as a MiniMax keyframe at frame 0

This makes it possible to combine:

- a short reference video for motion / identity continuity
- the previous clip's final frame as the starting frame of the next clip

It was created mainly for chained long-form MiniMax H3 workflows.

## Installation

Download or clone this repository into:

`ComfyUI/custom_nodes/`

Example:

`ComfyUI/custom_nodes/ComfyUI-MinimaxH3-RefFirstFrame/`

The folder should contain:

- `__init__.py`
- `README.md`
- `LICENSE`

Restart ComfyUI after installation.

## Node

**MiniMax H3 Reference to Video + First Frame**

The node is based on ComfyUI's built-in MiniMax H3 reference-to-video implementation, with additional first-frame conditioning.

## Tested use case

Used successfully for chained MiniMax H3 video generation with:

- short reference-video conditioning
- previous clip final frame as `first_frame`
- hybrid FL2VA / Ref2VA models such as `minimax_h3_hybrid_fl2va_ref2va_b30-49-int8`

## Notes

This is an experimental custom node and may depend on changes in ComfyUI's built-in MiniMax H3 implementation.

If ComfyUI changes the internal MiniMax H3 node API, this node may also require an update.

## Credits

Based on / adapted from ComfyUI's MiniMax H3 implementation:

https://github.com/Comfy-Org/ComfyUI

Developed with assistance from ChatGPT.

## License

GPL-3.0
