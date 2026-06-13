import sys
import os

# Đảm bảo Python có thể tìm thấy thư mục src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.server import app

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  🚀 AI PHẠT NGUỘI GIAO THÔNG — Web Server")
    print("  📁 Mở trình duyệt: http://localhost:5000")
    print("=" * 56 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
