"""
결과 갤러리 상태 및 헬퍼 모듈
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


@dataclass
class GalleryItem:
    image: Image.Image
    image_path: str
    model: str
    ratio: str
    quality: str
    prompt: str
    index: int
    status: str = "success"
    error: Optional[str] = None


class GalleryState:
    """세션 내 생성된 이미지 목록을 관리합니다."""

    def __init__(self):
        self._items: list[GalleryItem] = []

    def add(self, item: GalleryItem) -> None:
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    @property
    def items(self) -> list[GalleryItem]:
        return list(self._items)

    @property
    def images(self) -> list[Image.Image]:
        return [item.image for item in self._items if item.image is not None]

    @property
    def image_paths(self) -> list[str]:
        return [item.image_path for item in self._items if item.image_path]

    def to_gradio_gallery(self) -> list[tuple[Image.Image, str]]:
        """
        Gradio Gallery 컴포넌트에 전달할 형식으로 변환합니다.
        각 항목은 (image, caption) 튜플입니다.
        """
        result = []
        for item in self._items:
            if item.image is None:
                continue
            caption = (
                f"#{item.index + 1} | {item.model} | {item.ratio} | {item.quality}\n"
                f"{item.prompt[:80]}{'...' if len(item.prompt) > 80 else ''}"
            )
            result.append((item.image, caption))
        return result

    def get_summary(self) -> str:
        total = len(self._items)
        success = sum(1 for i in self._items if i.status == "success")
        failed = total - success
        if total == 0:
            return "아직 생성된 이미지가 없습니다."
        parts = [f"총 {total}장 생성 완료"]
        if failed:
            parts.append(f"(성공 {success}장 / 실패 {failed}장)")
        return " · ".join(parts)
