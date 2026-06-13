from src.config import LOST_ID_THRESHOLD, MIN_VIOLATION_FRAMES, MIN_VIOLATION_RATIO
from src.engine.core.logger import log_violation


class ViolationTracker:
    """
    Theo dõi vi phạm trong video với cơ chế "Deferred Logging":
      - KHÔNG ghi biên bản ngay khi phát hiện lần đầu
      - Tích lũy bằng chứng, chọn ảnh RÕ NHẤT (biển số lớn nhất)
      - Lọc nhiễu bằng Tỷ lệ (Ratio) khi xe rời khung hình
    """

    def __init__(self):
        self.violations = {}     # {track_id: ViolationData}
        self.last_seen = {}      # {track_id: frame_idx lần cuối thấy}
        self.logged_ids = set()  # Các ID đã ghi biên bản xong
        self.violation_votes = {} # {track_id: số frame nhận diện KHÔNG MŨ}
        self.total_valid_votes = {} # {track_id: tổng số frame nhìn thấy ĐẦU NGƯỜI (có mũ + không mũ)}

    def reset(self):
        """Reset toàn bộ tracker — gọi khi bắt đầu video mới."""
        self.violations.clear()
        self.last_seen.clear()
        self.logged_ids.clear()
        self.violation_votes.clear()
        self.total_valid_votes.clear()

    def record_vote(self, track_id, is_violation):
        """Ghi nhận 1 vote. Chỉ đếm khi AI thực sự nhận diện được đầu người (tránh frame mờ rác)."""
        self.total_valid_votes[track_id] = self.total_valid_votes.get(track_id, 0) + 1
        if is_violation:
            self.violation_votes[track_id] = self.violation_votes.get(track_id, 0) + 1

    def is_confirmed_violator_ui(self, track_id):
        """Khóa đỏ trên màn hình nếu đạt đủ số frame tối thiểu."""
        return self.violation_votes.get(track_id, 0) >= MIN_VIOLATION_FRAMES

    def is_final_violator(self, track_id):
        """Quyết định cuối cùng lúc ghi biên bản (Post-processing) bằng tỷ lệ."""
        viol_votes = self.violation_votes.get(track_id, 0)
        total_votes = self.total_valid_votes.get(track_id, 0)
        
        if total_votes == 0:
            return False
            
        ratio = viol_votes / total_votes
        
        # Điều kiện: Phải đủ số frame tối thiểu VÀ đạt tỷ lệ % vi phạm
        return viol_votes >= MIN_VIOLATION_FRAMES and ratio >= MIN_VIOLATION_RATIO

    def update_violation(self, track_id, plate_text, evidence_img, plate_area, frame_idx):
        """Cập nhật bằng chứng vi phạm cho 1 ID."""
        self.last_seen[track_id] = frame_idx

        if track_id in self.logged_ids:
            return

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

            is_better = (
                new_text_len > old_text_len or
                (new_text_len == old_text_len and plate_area > old['plate_area'])
            )

            if is_better:
                old['plate_text'] = plate_text
                old['evidence_img'] = evidence_img.copy()
                old['plate_area'] = plate_area

    def mark_seen(self, track_id, frame_idx):
        self.last_seen[track_id] = frame_idx

    def is_logged(self, track_id):
        return track_id in self.logged_ids

    def flush_lost_ids(self, current_frame, output_dir):
        """Ghi biên bản dựa trên xét duyệt tỷ lệ."""
        new_violations = 0
        lost_ids = []

        for track_id, last_frame in self.last_seen.items():
            if (current_frame - last_frame) > LOST_ID_THRESHOLD:
                if track_id in self.violations and track_id not in self.logged_ids:
                    # XÉT DUYỆT TỶ LỆ CUỐI CÙNG TRƯỚC KHI LƯU
                    if self.is_final_violator(track_id):
                        v = self.violations[track_id]
                        if v['plate_text']:
                            log_violation(v['plate_text'], v['evidence_img'], output_dir)
                            self.logged_ids.add(track_id)
                            print(f"🚨 GHI BIÊN BẢN (ID {track_id}): {v['plate_text']} (Votes: {self.violation_votes.get(track_id, 0)}/{self.total_valid_votes.get(track_id, 0)})")
                            new_violations += 1
                        else:
                            self.logged_ids.add(track_id)
                    else:
                        # Tỷ lệ an toàn cao -> Lọc nhiễu thành công -> Hủy hồ sơ
                        self.logged_ids.add(track_id)
                        print(f"🛡️ LOẠI BỎ FALSE POSITIVE (ID {track_id}): Votes {self.violation_votes.get(track_id, 0)}/{self.total_valid_votes.get(track_id, 0)}")
                lost_ids.append(track_id)

        for tid in lost_ids:
            self.last_seen.pop(tid, None)

        return new_violations

    def finalize(self, output_dir):
        new_violations = 0
        for track_id, v in self.violations.items():
            if track_id not in self.logged_ids:
                if self.is_final_violator(track_id):
                    if v['plate_text']:
                        log_violation(v['plate_text'], v['evidence_img'], output_dir)
                        print(f"🚨 GHI BIÊN BẢN (ID {track_id}): {v['plate_text']} (Votes: {self.violation_votes.get(track_id, 0)}/{self.total_valid_votes.get(track_id, 0)})")
                        new_violations += 1
                self.logged_ids.add(track_id)
        return new_violations