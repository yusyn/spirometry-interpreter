"""برچسب‌های فارسی برای نمایش در UI و گزارش‌ها"""

PATTERN_FA = {
    "Normal": "نرمال",
    "Obstructive": "انسدادی",
    "Restrictive": "محدودیتی",
    "Mixed": "مختلط",
    "Error": "خطا",
}


def pattern_fa(name: str) -> str:
    if not name:
        return "—"
    return PATTERN_FA.get(str(name), str(name))
