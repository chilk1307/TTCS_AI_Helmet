import sys
import os

# Đảm bảo Python có thể tìm thấy thư mục src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.engine.pipeline import main

if __name__ == '__main__':
    main()
