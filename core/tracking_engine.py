class ViolationTracker:
    def __init__(self):
        # Bộ nhớ lưu trữ các ID xe đã bị ghi biên bản
        self.logged_ids = set()

    def is_logged(self, track_id):
        return track_id in self.logged_ids

    def mark_as_logged(self, track_id):
        self.logged_ids.add(track_id)