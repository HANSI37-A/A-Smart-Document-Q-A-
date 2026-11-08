"""Find the figure regions on a rendered PDF page.

SAM (Segment Anything) finds object outlines. It does not know what a logo or a
chart is, so it is only used to cut figures out of the page; deciding whether a
figure is useful happens later (rag/multimodal.py and rag/image_selector.py).

pdfplumber already knows roughly where embedded images and chart drawings are.
Those rough boxes ("hints") are given to SAM as prompts, and SAM returns a
tighter outline. Without hints, SAM segments the whole page automatically."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

MIN_AREA = 0.01      # boxes smaller than 1% of the page are icons or bullets
MAX_AREA = 0.95      # a box covering the whole page is the page, not a figure
MAX_ASPECT = 8.0     # thinner than 8:1 is a line, rule or divider
MERGE_IOU = 0.5      # boxes that overlap this much are the same figure
MAX_FIGURES = 4      # per page
REFINE_KEEP = 0.5    # SAM box smaller than half the hint: SAM cut too much, keep the hint


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersection(self, other: "Box") -> float:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return max(0.0, w) * max(0.0, h)

    def iou(self, other: "Box") -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def contains(self, other: "Box", tolerance: float = 0.9) -> bool:
        """True if most of `other` lies inside this box."""
        return other.area > 0 and self.intersection(other) / other.area >= tolerance

    def union(self, other: "Box") -> "Box":
        return Box(min(self.x0, other.x0), min(self.y0, other.y0),
                   max(self.x1, other.x1), max(self.y1, other.y1))

    def scaled(self, factor: float) -> "Box":
        return Box(self.x0 * factor, self.y0 * factor, self.x1 * factor, self.y1 * factor)

    def padded(self, pad: float, width: float, height: float) -> "Box":
        return Box(max(0.0, self.x0 - pad), max(0.0, self.y0 - pad),
                   min(width, self.x1 + pad), min(height, self.y1 + pad))


def filter_boxes(boxes: list[Box], page_w: float, page_h: float,
                 max_figures: int = MAX_FIGURES) -> list[Box]:
    """Keep figure-sized boxes, merge duplicates, and order them top to bottom."""
    page_area = page_w * page_h
    kept = []
    for b in boxes:
        if not page_area or not b.width or not b.height:
            continue
        share = b.area / page_area
        aspect = max(b.width / b.height, b.height / b.width)
        if MIN_AREA <= share <= MAX_AREA and aspect <= MAX_ASPECT:
            kept.append(b)

    merged: list[Box] = []
    for b in sorted(kept, key=lambda b: b.area, reverse=True):   # big boxes absorb small ones
        for i, m in enumerate(merged):
            if m.iou(b) > MERGE_IOU or m.contains(b):
                merged[i] = m.union(b)
                break
        else:
            merged.append(b)

    merged = merged[:max_figures]
    return sorted(merged, key=lambda b: (round(b.y0), b.x0))


class RegionSegmenter(Protocol):
    """Anything with this method works, which lets tests use a fake."""
    def segment(self, image: Image.Image, hints: list[Box]) -> list[Box]: ...


class PassThroughSegmenter:
    """No SAM: use pdfplumber's boxes as they are, or the whole page if there are none.
    This is the behaviour the app had before SAM was added."""

    def segment(self, image: Image.Image, hints: list[Box]) -> list[Box]:
        boxes = filter_boxes(hints, image.width, image.height)
        return boxes or [Box(0, 0, image.width, image.height)]


class SamSegmenter:
    """MobileSAM (or any SAM model) through the `ultralytics` package.

    The model weights download on first use (MobileSAM is about 40 MB)."""

    def __init__(self, weights: str | Path = "mobile_sam.pt"):
        from ultralytics import SAM   # heavy import (PyTorch), so only when SAM is used

        weights = Path(weights)
        weights.parent.mkdir(parents=True, exist_ok=True)
        self.model = SAM(str(weights))

    def _predict(self, image: Image.Image, hints: list[Box]) -> list[Box]:
        kwargs = {"bboxes": [[b.x0, b.y0, b.x1, b.y1] for b in hints]} if hints else {}
        results = self.model(image.convert("RGB"), verbose=False, **kwargs)
        boxes = []
        for r in results:
            if r.masks is None:
                continue
            for mask in r.masks.data.cpu().numpy():
                ys, xs = mask.nonzero()
                if len(xs):
                    boxes.append(Box(float(xs.min()), float(ys.min()),
                                     float(xs.max()) + 1, float(ys.max()) + 1))
        return boxes

    def segment(self, image: Image.Image, hints: list[Box]) -> list[Box]:
        found = self._predict(image, hints)
        if hints:
            # One mask per hint, in order. SAM sometimes outlines only part of a chart
            # (the bars without the axis labels) or spills into the page background;
            # then the hint box is the better crop.
            found = [s if REFINE_KEEP * h.area <= s.area <= h.area / REFINE_KEEP else h
                     for s, h in zip(found, hints)] + hints[len(found):]
        return filter_boxes(found, image.width, image.height)
