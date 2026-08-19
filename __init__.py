import math
import torchaudio
import nodes
import node_helpers

from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_minimax_h3 import (
    _empty_av_latent,
    _resize,
    adapt_canvas,
    CANVAS_MULTIPLE,
    REF_IMAGE_SHORT_EDGE,
    FPS,
)


class MiniMaxH3ReferenceToVideoFirstFrame(io.ComfyNode):
    """
    Ref2V + true-ish first frame support.

    Adds first_frame in two ways:
    1) send it to the tokenizer as an image (I2V-style)
    2) add it as minimax_keyframes[frame 0] (I2V-style)
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceToVideoFirstFrame",
            description="MiniMax H3 Ref2V with extra first_frame conditioning similar to I2V.",
            display_name="MiniMax H3 Reference to Video + First Frame",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip="Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 'max' uses the reference pipeline's 2048px short edge for best identity fidelity."
                ),
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip="Extra start image, handled similarly to I2V first_frame."
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Reference image (downscaled to 2048 short edge if larger, never upscaled)"
                        ),
                        prefix="ref_image_",
                        min=0,
                        max=9
                    )
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_video",
                            tooltip="Reference video frames at 24 fps (2-15s)"
                        ),
                        prefix="ref_video_",
                        min=0,
                        max=3
                    )
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Soundtrack of the same-numbered reference video"
                        ),
                        prefix="ref_video_audio_",
                        min=0,
                        max=3
                    )
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_audio",
                            tooltip="Standalone reference audio"
                        ),
                        prefix="ref_audio_",
                        min=0,
                        max=3
                    )
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output()
            ],
        )

    @staticmethod
    def _encode_ref_audio(audio_vae, audio):
        waveform = audio["waveform"]  # [B, C, L]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
        return z, z.shape[-1]

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        audio_vae,
        prompt,
        width,
        height,
        length,
        ref_image_size="match",
        first_frame=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None
    ) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # tokenizer-side reference presentation
        ref_blocks = []  # DiT-side minimax_refs payload

        # -----------------------------
        # regular ref images
        # -----------------------------
        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]

            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))

            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

            resized = _resize(img[:1], tw, th, "disabled")
            z = vae.encode(resized)

            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({
                "kind": "image",
                "latent_h": th // 16,
                "latent_w": tw // 16,
                "latent": z
            })

        # -----------------------------
        # regular ref videos (+ optional paired audio)
        # -----------------------------
        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue

            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])

            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)

            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)

            frames = _resize(video_frames, cw, ch, "disabled")

            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]

            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")

            while n % 17 != 5:
                n -= 1
            frames = frames[:n]

            z = vae.encode(frames)

            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})

            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]

            ref_items.append({
                "type": "video",
                "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(len(sample_idx))]
            })

            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": z.shape[2],
                "latent_h": ch // 16,
                "latent_w": cw // 16,
                "ref_audio_t": ref_audio_t,
                "latent": z,
                "audio_latent": audio_latent
            })

        # -----------------------------
        # standalone ref audios
        # -----------------------------
        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({
                "kind": "audio",
                "ref_audio_t": ref_audio_t,
                "audio_latent": audio_latent
            })

        # -----------------------------
        # I2V-style first frame
        # -----------------------------
        images = []
        keyframes = []

        if first_frame is not None:
            ff = _resize(first_frame[:1], width, height, "disabled")
            images.append(ff)
            keyframes.append({
                "resolved_frame_index": 0,
                "image": ff
            })

        # Prefer true I2V-style path:
        #   - first_frame goes to tokenizer via images=
        #   - refs go via minimax_ref_items=
        # If the tokenizer rejects images=, fall back to appending ff as the last image ref.
        try:
            tokens = clip.tokenize(
                prompt,
                images=images,
                minimax_ref_items=ref_items
            )
        except TypeError:
            token_ref_items = list(ref_items)
            if first_frame is not None:
                token_ref_items.append({"type": "image", "data": ff})
            tokens = clip.tokenize(
                prompt,
                minimax_ref_items=token_ref_items
            )

        cond = clip.encode_from_tokens_scheduled(tokens)

        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_refs": ref_blocks
            })

        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })

        return io.NodeOutput(cond, latent)


class MiniMaxH3RefFirstExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3ReferenceToVideoFirstFrame,
        ]


async def comfy_entrypoint() -> MiniMaxH3RefFirstExtension:
    return MiniMaxH3RefFirstExtension()