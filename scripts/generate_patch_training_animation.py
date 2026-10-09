"""Generate an animation of multiscale 2-D patch extraction.

The animation starts from a 252 x 252 field. At the native resolution a
40 x 40 window moves in raster order with stride 20. It then Gaussian-blurs
the field, keeps every s-th pixel for s=2,...,8, and repeats the scan while
keeping the patch and stride fixed in sampled-grid pixels.

The exact number of valid windows is shown at every scale. The sampled grids
at s=7 and s=8 are smaller than 40 x 40, so, matching ``build_dataset_ffno``,
those scales are skipped without padding.

Dependencies: NumPy, Pillow, and an ``ffmpeg`` executable on PATH.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


SOURCE_SIZE = 252
PATCH_SIZE = 40
STRIDE = 20
SCALES = tuple(range(1, 9))

WIDTH, HEIGHT = 1280, 720
DEFAULT_FPS = 18
PANEL_SIDE = 438
PANEL_Y = 150
LEFT_X, RIGHT_X = 82, 760

BACKGROUND = "#0b1223"
FOREGROUND = "#f3f6fb"
MUTED = "#aab8cf"
BORDER = "#40516c"
CYAN = "#58d8eb"
GOLD = "#ffd174"
CORAL = "#ff8292"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable system font, falling back to Pillow's default font."""

    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_TITLE = _font(37, bold=True)
FONT_HEADING = _font(23, bold=True)
FONT_BODY = _font(21)
FONT_SMALL = _font(17)
FONT_COUNTER = _font(28, bold=True)


@dataclass(frozen=True)
class FrameSpec:
    """One animation frame at a scale and fractional patch-list position."""

    scale: int
    patch_progress: float | None


def make_source_field() -> Image.Image:
    """Create a deterministic image with smooth and high-frequency detail."""

    yy, xx = np.mgrid[0:SOURCE_SIZE, 0:SOURCE_SIZE].astype(np.float32)
    x = xx / SOURCE_SIZE
    y = yy / SOURCE_SIZE
    field = (
        0.34 * np.sin(2 * np.pi * (1.6 * x + 0.55 * y))
        + 0.19 * np.cos(2 * np.pi * (4.1 * y - 0.75 * x))
        + 0.12 * np.sin(2 * np.pi * (8.0 * x + 5.5 * y))
        + 0.86 * np.exp(-((x - 0.30) ** 2 + (y - 0.28) ** 2) / 0.018)
        - 0.70 * np.exp(-((x - 0.72) ** 2 + (y - 0.70) ** 2) / 0.026)
    )
    # A sharp diagonal makes the benefit of pre-decimation blur easy to see.
    field += 0.42 * (np.abs(y - (0.54 * x + 0.18)) < 0.012)
    field = np.clip((field - field.min()) / np.ptp(field), 0.0, 1.0)
    rgb = np.stack(
        (
            20 + 220 * field,
            45 + 150 * np.sin(np.pi * field) ** 2,
            100 + 130 * (1.0 - field),
        ),
        axis=-1,
    )
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def blur_and_decimate(source: Image.Image, scale: int) -> Image.Image:
    """Gaussian prefilter and select every ``scale``-th source pixel."""

    if scale == 1:
        return source.copy()
    # The animation uses sigma=s/2 as an anti-aliasing prefilter. PIL calls
    # its Gaussian standard deviation ``radius``.
    blurred = source.filter(ImageFilter.GaussianBlur(radius=scale / 2))
    return Image.fromarray(np.asarray(blurred)[::scale, ::scale], mode="RGB")


def patch_positions(grid_size: int) -> list[tuple[int, int]]:
    """Return all valid (x, y) patch starts in row-major order."""

    if grid_size < PATCH_SIZE:
        return []
    starts = range(0, grid_size - PATCH_SIZE + 1, STRIDE)
    return [(x, y) for y in starts for x in starts]


def frame_plan(fps: int) -> list[FrameSpec]:
    """Build a plan that visits every valid patch at every scale."""

    frames: list[FrameSpec] = []
    for scale in SCALES:
        grid_size = math.ceil(SOURCE_SIZE / scale)
        positions = patch_positions(grid_size)
        # Briefly establish each scale before scanning.
        frames.extend(FrameSpec(scale, 0.0 if positions else None) for _ in range(round(0.65 * fps)))
        if positions:
            # Two frames per transition make the square visibly move rather
            # than teleport, while the integer position is still visited.
            for index in range(len(positions)):
                frames.append(FrameSpec(scale, float(index)))
                if index + 1 < len(positions):
                    frames.append(FrameSpec(scale, index + 0.5))
            frames.extend(FrameSpec(scale, float(len(positions) - 1)) for _ in range(round(0.45 * fps)))
        else:
            frames.extend(FrameSpec(scale, None) for _ in range(round(1.15 * fps)))
    return frames


def centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text((center[0] - width / 2, center[1] - height / 2), text, font=font, fill=fill)


def paste_panel(
    canvas: Image.Image,
    image: Image.Image,
    x: int,
    heading: str,
    *,
    nearest: bool,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text((x, PANEL_Y - 37), heading, font=FONT_HEADING, fill=FOREGROUND)
    resized = image.resize(
        (PANEL_SIDE, PANEL_SIDE),
        Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC,
    )
    canvas.paste(resized, (x, PANEL_Y))
    draw.rectangle(
        (x - 2, PANEL_Y - 2, x + PANEL_SIDE + 2, PANEL_Y + PANEL_SIDE + 2),
        outline=BORDER,
        width=2,
    )


def interpolated_position(
    positions: list[tuple[int, int]], progress: float | None
) -> tuple[float, float] | None:
    if not positions or progress is None:
        return None
    lo = min(int(progress), len(positions) - 1)
    hi = min(lo + 1, len(positions) - 1)
    fraction = progress - math.floor(progress)
    return (
        positions[lo][0] * (1 - fraction) + positions[hi][0] * fraction,
        positions[lo][1] * (1 - fraction) + positions[hi][1] * fraction,
    )


def hex_rgb(color: str) -> tuple[int, int, int]:
    """Convert a six-digit hex color to an RGB tuple."""

    value = color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def draw_patch(
    draw: ImageDraw.ImageDraw,
    panel_x: int,
    grid_size: int,
    position: tuple[float, float],
    patch_size: float,
    color: str,
) -> None:
    unit = PANEL_SIDE / grid_size
    x, y = position
    rect = (
        panel_x + x * unit,
        PANEL_Y + y * unit,
        panel_x + (x + patch_size) * unit,
        PANEL_Y + (y + patch_size) * unit,
    )
    draw.rectangle(rect, fill=(*hex_rgb(color), 44), outline=color, width=5)


def draw_scale_track(draw: ImageDraw.ImageDraw, active_scale: int) -> None:
    x0, x1, y = 386, 894, 655
    draw.line((x0, y, x1, y), fill=BORDER, width=3)
    for scale in SCALES:
        x = x0 + (scale - 1) * (x1 - x0) / (len(SCALES) - 1)
        active = scale == active_scale
        radius = 14 if active else 9
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=CYAN if active else BORDER)
        centered_text(draw, (x, y + 28), str(scale), FONT_SMALL, FOREGROUND if active else MUTED)
    draw.text((307, y - 11), "scale", font=FONT_SMALL, fill=MUTED)


def render_frame(
    source: Image.Image,
    pyramid: dict[int, Image.Image],
    spec: FrameSpec,
    frame_number: int,
    total_frames: int,
) -> Image.Image:
    scale = spec.scale
    sampled = pyramid[scale]
    grid_size = sampled.width
    positions = patch_positions(grid_size)
    moving_position = interpolated_position(positions, spec.patch_progress)

    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((55, 27), "Multiscale patch extraction", font=FONT_TITLE, fill=FOREGROUND)
    draw.text(
        (57, 77),
        "252 × 252 source  •  Gaussian blur  •  every s-th pixel  •  patch 40 × 40  •  stride 20",
        font=FONT_BODY,
        fill=MUTED,
    )

    paste_panel(canvas, source, LEFT_X, "Source image  ·  252 × 252", nearest=False)
    ordinal = "1st" if scale == 1 else "2nd" if scale == 2 else "3rd" if scale == 3 else f"{scale}th"
    operation = "native grid" if scale == 1 else f"blur → select every {ordinal} pixel"
    paste_panel(canvas, sampled, RIGHT_X, f"Scale {scale}  ·  {operation}", nearest=True)
    draw = ImageDraw.Draw(canvas, "RGBA")

    if moving_position is not None:
        # Right: the fixed 40 x 40 patch in sampled-grid coordinates.
        draw_patch(draw, RIGHT_X, grid_size, moving_position, PATCH_SIZE, GOLD)
        # Left: the corresponding physical footprint in source coordinates.
        source_position = (moving_position[0] * scale, moving_position[1] * scale)
        draw_patch(draw, LEFT_X, SOURCE_SIZE, source_position, PATCH_SIZE * scale, CYAN)
        current = min(int(spec.patch_progress or 0) + 1, len(positions))
        per_side = int(math.sqrt(len(positions)))
        status = f"position {current} / {len(positions)}   ({per_side} × {per_side} valid starts)"
        centered_text(draw, (RIGHT_X + PANEL_SIDE / 2, 618), status, FONT_COUNTER, GOLD)
        centered_text(
            draw,
            (LEFT_X + PANEL_SIDE / 2, 618),
            f"source footprint: {PATCH_SIZE * scale} × {PATCH_SIZE * scale}",
            FONT_BODY,
            CYAN,
        )
    else:
        overlay = (RIGHT_X + 33, PANEL_Y + 164, RIGHT_X + PANEL_SIDE - 33, PANEL_Y + 276)
        draw.rounded_rectangle(overlay, radius=16, fill=(19, 31, 52, 235), outline=CORAL, width=3)
        centered_text(
            draw,
            (RIGHT_X + PANEL_SIDE / 2, PANEL_Y + 202),
            "Scale skipped — grid < 40 × 40",
            FONT_HEADING,
            CORAL,
        )
        centered_text(
            draw,
            (RIGHT_X + PANEL_SIDE / 2, PANEL_Y + 244),
            f"sampled image is only {grid_size} × {grid_size}",
            FONT_BODY,
            MUTED,
        )
        centered_text(draw, (RIGHT_X + PANEL_SIDE / 2, 618), "0 valid patch positions", FONT_COUNTER, CORAL)
        centered_text(draw, (LEFT_X + PANEL_SIDE / 2, 618), "no patch is emitted at this scale", FONT_BODY, MUTED)

    draw_scale_track(draw, scale)
    progress = max(0.0, min(1.0, (frame_number + 1) / total_frames))
    draw.line((55, 700, 1225, 700), fill=BORDER, width=4)
    draw.line((55, 700, 55 + 1170 * progress, 700), fill=CYAN, width=5)
    return canvas


def encode_mp4(
    output: Path,
    source: Image.Image,
    pyramid: dict[int, Image.Image],
    plan: list[FrameSpec],
    fps: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH; install it to encode the MP4")
    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise RuntimeError("failed to open ffmpeg stdin")
        for frame_number, spec in enumerate(plan):
            frame = render_frame(source, pyramid, spec, frame_number, len(plan))
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parents[1] / "docs/media/patch_training_scales.mp4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output, help="output MP4 path")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="frames per second")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    source = make_source_field()
    pyramid = {scale: blur_and_decimate(source, scale) for scale in SCALES}
    plan = frame_plan(args.fps)

    # Save a representative poster alongside the video.
    poster = args.output.with_suffix(".png")
    poster_spec = FrameSpec(scale=2, patch_progress=12.0)
    render_frame(source, pyramid, poster_spec, 0, 1).save(poster)
    encode_mp4(args.output, source, pyramid, plan, args.fps)

    counts = {scale: len(patch_positions(pyramid[scale].width)) for scale in SCALES}
    print(f"wrote {args.output}")
    print(f"wrote {poster}")
    print("valid patch counts:", ", ".join(f"s={s}: {counts[s]}" for s in SCALES))


if __name__ == "__main__":
    main()
