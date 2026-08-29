"""اجرای وب‌اپلیکیشن Flask"""
import sys
from pathlib import Path

# اطمینان از اینکه ریشه پروژه در path است
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("وب‌اپ روی http://127.0.0.1:5000 در حال اجراست")
    print("برای توقف: Ctrl+C")
    app.run(debug=True, host="127.0.0.1", port=5000)
