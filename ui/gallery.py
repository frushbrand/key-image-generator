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
    reference_image_paths: list = field(default_factory=list)


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

    def get_success_item_by_visual_index(self, visual_idx: int) -> Optional[GalleryItem]:
        """
        최신순(역순) 갤러리 표시 기준으로 시각적 인덱스에 해당하는 항목을 반환합니다.
        시각적 인덱스 0 = 가장 최근 항목
        """
        success_items = [i for i in self._items if i.status == "success"]
        real_idx = len(success_items) - 1 - visual_idx
        if 0 <= real_idx < len(success_items):
            return success_items[real_idx]
        return None

    def to_gradio_gallery(self) -> list[tuple[str, str]]:
        """
        Gradio Gallery 컴포넌트에 전달할 형식으로 변환합니다.
        각 항목은 (image_path, caption) 튜플이며, 최신 항목이 맨 앞(왼쪽 위)에 표시됩니다.
        파일 경로를 직접 사용하므로 Gradio가 원본 PNG를 그대로 서빙하고,
        오버레이 다운로드 버튼이 img src에서 원본 파일을 직접 내려받을 수 있습니다.
        """
        result = []
        for item in reversed(self._items):
            if not item.image_path or not os.path.exists(item.image_path):
                continue
            caption = (
                f"#{item.index + 1} | {item.model} | {item.ratio} | {item.quality}\n"
                f"{item.prompt[:80]}{'...' if len(item.prompt) > 80 else ''}"
            )
            result.append((item.image_path, caption))
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
