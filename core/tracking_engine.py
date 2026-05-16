from config import LOST_ID_THRESHOLD
from core.logger import log_violation


class ViolationTracker:
    """
    Theo dõi vi phạm trong video với cơ chế "Deferred Logging":
      - KHÔNG ghi biên bản ngay khi phát hiện lần đầu
      - Tích lũy bằng chứng, chọn ảnh RÕ NHẤT (biển số lớn nhất)
      - Ghi biên bản khi xe rời khung hình hoặc khi video kết thúc
    """

    def __init__(self):
        self.violations = {}     # {track_id: ViolationData}
        self.last_seen = {}      # {track_id: frame_idx lần cuối thấy}
        self.logged_ids = set()  # Các ID đã ghi biên bản xong

    def reset(self):
        """Reset toàn bộ tracker — gọi khi bắt đầu video mới."""
        self.violations.clear()
        self.last_seen.clear()
        self.logged_ids.clear()

    def update_violation(self, track_id, plate_text, evidence_img, plate_area, frame_idx):
        """
        Cập nhật bằng chứng vi phạm cho 1 ID.
        Ưu tiên: biển số ĐẦY ĐỦ NHẤT (nhiều ký tự nhất) → ảnh rõ nhất.
        """
        self.last_seen[track_id] = frame_idx

        if track_id in self.logged_ids:
            return  # Đã ghi biên bản rồi, bỏ qua

        new_text_len = len(plate_text.replace('-', '')) if plate_text else 0

        if track_id not in self.violations:
            self.violations[track_id] = {
                'plate_text': plate_text,
                'evidence_img': evidence_img.copy(),
                'plate_area': plate_area,
            }
        else:
            old = self.violations[track_id]
            old_text_len = len(old['plate_text'].replace('-', '')) if old['plate_text'] else 0

            # ★ Ưu tiên 1: text NHIỀU KÝ TỰ HƠN (đọc đầy đủ hơn)
            # ★ Ưu tiên 2: nếu cùng số ký tự → chọn biển số diện tích lớn hơn
            is_better = (
                new_text_len > old_text_len or
                (new_text_len == old_text_len and plate_area > old['plate_area'])
            )

            if is_better:
                old['plate_text'] = plate_text
                old['evidence_img'] = evidence_img.copy()
                old['plate_area'] = plate_area

    def mark_seen(self, track_id, frame_idx):
        """Đánh dấu ID vẫn còn trong khung hình (dù chưa vi phạm)."""
        self.last_seen[track_id] = frame_idx

    def is_logged(self, track_id):
        return track_id in self.logged_ids

    def flush_lost_ids(self, current_frame, output_dir):
        """
        Kiểm tra các ID đã biến mất khỏi khung hình.
        Nếu biến mất > LOST_ID_THRESHOLD frame → ghi biên bản.
        Trả về số vi phạm mới được ghi.
        """
        new_violations = 0
        lost_ids = []

        for track_id, last_frame in self.last_seen.items():
            if (current_frame - last_frame) > LOST_ID_THRESHOLD:
                if track_id in self.violations and track_id not in self.logged_ids:
                    v = self.violations[track_id]
                    # ★ Chỉ ghi nếu CÓ biển số — không ghi KHONG_RO
                    if v['plate_text']:
                        log_violation(v['plate_text'], v['evidence_img'], output_dir)
                        self.logged_ids.add(track_id)
                        print(f"🚨 GHI BIÊN BẢN (ID {track_id}): {v['plate_text']}")
                        new_violations += 1
                    else:
                        self.logged_ids.add(track_id)  # Đánh dấu xong để không check lại
                lost_ids.append(track_id)

        # Dọn dẹp các ID đã mất
        for tid in lost_ids:
            self.last_seen.pop(tid, None)

        return new_violations

    def finalize(self, output_dir):
        """
        Ghi biên bản cho TẤT CẢ vi phạm chưa được log.
        Gọi khi video kết thúc.
        Trả về số vi phạm mới được ghi.
        """
        new_violations = 0
        for track_id, v in self.violations.items():
            if track_id not in self.logged_ids:
                # ★ Chỉ ghi nếu CÓ biển số
                if v['plate_text']:
                    log_violation(v['plate_text'], v['evidence_img'], output_dir)
                    print(f"🚨 GHI BIÊN BẢN (ID {track_id}): {v['plate_text']}")
                    new_violations += 1
                self.logged_ids.add(track_id)
        return new_violations