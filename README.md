# MiniMax H3 Reference to Video + First Frame

Experimental custom node for ComfyUI MiniMax H3.

This node extends the behavior of ComfyUI's built-in
MiniMaxH3ReferenceToVideo node by adding an independent first_frame input.

The first frame is supplied both:
- to the tokenizer as I2V-style image conditioning
- as a MiniMax keyframe at frame 0

This is useful for chained long-form H3 workflows where a short reference
video is used for temporal/identity continuity while the final frame of the
previous clip is used as the starting frame of the next clip.

Based on / adapted from ComfyUI's MiniMax H3 implementation.
ComfyUI:
https://github.com/Comfy-Org/ComfyUI

Developed with assistance from ChatGPT.

License: GPL-3.0
