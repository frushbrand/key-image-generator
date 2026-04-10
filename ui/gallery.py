"""
결과 갤러리 상태 및 헬퍼 모듈
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from core.image_utils import PLACEHOLDER_IMAGE_PATH

# 갤러리에 표시되는 항목 상태 집합 (대기 중 + 성공)
_VISIBLE_STATUSES = frozenset(("success", "pending"))


@dataclass
class GalleryItem:
    image: Image.Image
    image_path: str
    model: str
    ratio: str
    quality: str
    prompt: str
    index: int
    status: str = "success"   # "success" | "pending" | "failed"
    error: Optional[str] = None
    reference_image_paths: list = field(default_factory=list)
    thumbnail_path: str = ""


class GalleryState:
    """세션 내 생성된 이미지 목록을 관리합니다.
    스레드 안전 설계: 여러 생성 작업이 동시에 진행될 수 있습니다."""

    def __init__(self):
        self._items: list[GalleryItem] = []
        self._lock = threading.Lock()

    def add(self, item: GalleryItem) -> None:
        with self._lock:
            self._items.append(item)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def allocate_pending_items(
        self, count: int, model: str, ratio: str, quality: str, prompt: str
    ) -> list[int]:
        """N개의 대기 중 슬롯을 미리 할당하고 각 슬롯의 _items 인덱스를 반환합니다."""
        with self._lock:
            indices: list[int] = []
            for _ in range(count):
                idx = len(self._items)
                item = GalleryItem(
                    image=None,
                    image_path="",
                    thumbnail_path="",
                    model=model,
                    ratio=ratio,
                    quality=quality,
                    prompt=prompt,
                    index=idx,
                    status="pending",
                )
                self._items.append(item)
                indices.append(idx)
            return indices

    def fill_pending_item(
        self,
        gallery_index: int,
        image_path: str,
        thumbnail_path: str,
        status: str = "success",
        error: Optional[str] = None,
    ) -> None:
        """대기 중 슬롯을 완료된 이미지(또는 실패)로 업데이트합니다."""
        with self._lock:
            if 0 <= gallery_index < len(self._items):
                item = self._items[gallery_index]
                item.image_path = image_path
                item.thumbnail_path = thumbnail_path
                item.status = status
                item.error = error

    def fail_remaining_pending(self, gallery_indices: list[int], error: str) -> None:
        """지정된 인덱스 중 아직 pending 상태인 항목을 실패로 표시합니다."""
        with self._lock:
            for idx in gallery_indices:
                if 0 <= idx < len(self._items):
                    item = self._items[idx]
                    if item.status == "pending":
                        item.status = "failed"
                        item.error = error

    def has_pending(self) -> bool:
        """현재 pending 상태인 항목이 하나라도 있으면 True를 반환합니다."""
        with self._lock:
            return any(i.status == "pending" for i in self._items)

    def remove_by_visual_indices(self, visual_indices: list[int]) -> list[str]:
        """
        시각적 인덱스(역순 표시 기준, 대기 중·성공 항목 포함) 목록에 해당하는 항목을 제거합니다.
        제거된 성공 항목의 image_path 목록을 반환합니다.
        """
        with self._lock:
            visible = [
                i for i in reversed(self._items) if i.status in _VISIBLE_STATUSES
            ]
            to_remove = set()
            removed_paths: list[str] = []
            for vi in visual_indices:
                if 0 <= vi < len(visible):
                    item = visible[vi]
                    to_remove.add(id(item))
                    if item.image_path and item.status == "success":
                        removed_paths.append(item.image_path)
            self._items = [i for i in self._items if id(i) not in to_remove]
            return removed_paths

    @property
    def items(self) -> list[GalleryItem]:
        with self._lock:
            return list(self._items)

    @property
    def images(self) -> list[Image.Image]:
        with self._lock:
            return [item.image for item in self._items if item.image is not None]

    @property
    def image_paths(self) -> list[str]:
        with self._lock:
            return [
                item.image_path
                for item in self._items
                if item.image_path and item.status == "success"
            ]

    def get_success_item_by_visual_index(self, visual_idx: int) -> Optional[GalleryItem]:
        """
        갤러리 표시 순서(최신 먼저, 대기 중·성공 항목 포함)에서 시각적 인덱스에 해당하는
        항목을 반환합니다. 해당 항목이 대기 중이거나 범위를 벗어나면 None을 반환합니다.
        """
        with self._lock:
            visible = [
                i for i in reversed(self._items) if i.status in _VISIBLE_STATUSES
            ]
            if 0 <= visual_idx < len(visible):
                item = visible[visual_idx]
                return item if item.status == "success" else None
            return None

    def to_gradio_gallery(self) -> list[tuple[str, str]]:
        """
        Gradio Gallery 컴포넌트에 전달할 형식으로 변환합니다.
        - 대기 중(pending) 항목: 플레이스홀더 이미지와 '⏳ 생성 중...' 캡션
        - 성공(success) 항목: 썸네일(없으면 원본)과 전체 메타데이터 캡션
        최신 항목이 맨 앞(왼쪽 위)에 표시됩니다.
        """
        with self._lock:
            result = []
            for item in reversed(self._items):
                if item.status == "pending":
                    caption = f"⏳ 생성 중... | {item.model} | {item.ratio}"
                    result.append((PLACEHOLDER_IMAGE_PATH, caption))
                    continue
                if not item.image_path or not os.path.exists(item.image_path):
                    continue
                # 썸네일 사용 (있으면); 없으면 원본
                display_path = (
                    item.thumbnail_path
                    if item.thumbnail_path and os.path.exists(item.thumbnail_path)
                    else item.image_path
                )
                # 프롬프트를 전체 표시 (캡션 잘림 없음)
                caption = (
                    f"#{item.index + 1} | {item.model} | {item.ratio} | {item.quality}\n"
                    f"{item.prompt}"
                )
                result.append((display_path, caption))
            return result

    def get_summary(self) -> str:
        with self._lock:
            success = sum(1 for i in self._items if i.status == "success")
            pending = sum(1 for i in self._items if i.status == "pending")
            failed = sum(1 for i in self._items if i.status == "failed")
            total = success + pending + failed
            if total == 0:
                return "아직 생성된 이미지가 없습니다."
            parts = [f"총 {success}장 완료"]
            if pending:
                parts.append(f"생성 중 {pending}장")
            if failed:
                parts.append(f"실패 {failed}장")
            return " · ".join(parts)
