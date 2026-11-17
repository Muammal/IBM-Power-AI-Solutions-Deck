"""gazetteer.py: Iraq's governorates with Arabic, Kurdish (Sorani) and English names plus main cities.

Rule-based matching runs first (deterministic, fast, explainable); the LLM's governorate list is accepted only
for names in this gazetteer. Halabja was recognised as a governorate by the Iraqi parliament in 2025.
"""
import re

GOVERNORATES = {
    "Baghdad": ["بغداد", "Baghdad", "بەغدا", "الكاظمية", "الأعظمية", "الكرادة", "مدينة الصدر", "Kadhimiya", "Karrada"],
    "Basra": ["البصرة", "Basra", "Basrah", "بەسرە", "الزبير", "أم قصر", "الفاو", "Umm Qasr", "Zubair"],
    "Nineveh": ["نينوى", "Nineveh", "Ninawa", "الموصل", "Mosul", "تلعفر", "Tal Afar", "سنجار", "Sinjar", "مووسڵ"],
    "Erbil": ["أربيل", "اربيل", "Erbil", "Arbil", "Hawler", "هەولێر"],
    "Sulaymaniyah": ["السليمانية", "Sulaymaniyah", "Sulaimani", "Slemani", "سلێمانی"],
    "Duhok": ["دهوك", "Duhok", "Dohuk", "دهۆک", "زاخو", "Zakho"],
    "Halabja": ["حلبجة", "Halabja", "هەڵەبجە"],
    "Kirkuk": ["كركوك", "Kirkuk", "کەرکووک"],
    "Anbar": ["الأنبار", "الانبار", "Anbar", "الرمادي", "Ramadi", "الفلوجة", "Fallujah", "القائم", "Al-Qaim"],
    "Diyala": ["ديالى", "Diyala", "بعقوبة", "Baqubah", "خانقين", "Khanaqin"],
    "Salah al-Din": ["صلاح الدين", "Salah al-Din", "Salahuddin", "تكريت", "Tikrit", "سامراء", "Samarra", "بيجي", "Baiji"],
    "Babil": ["بابل", "Babil", "Babylon", "الحلة", "Hillah"],
    "Karbala": ["كربلاء", "Karbala", "Kerbala"],
    "Najaf": ["النجف", "Najaf", "الكوفة", "Kufa"],
    "Wasit": ["واسط", "Wasit", "الكوت", "Kut"],
    "Qadisiyyah": ["القادسية", "Qadisiyyah", "Diwaniyah", "الديوانية"],
    "Muthanna": ["المثنى", "Muthanna", "السماوة", "Samawah"],
    "Dhi Qar": ["ذي قار", "Dhi Qar", "الناصرية", "Nasiriyah"],
    "Maysan": ["ميسان", "Maysan", "العمارة", "Amarah"],
}
NAMES = sorted(GOVERNORATES)

_PATTERNS = []
for gov, names in GOVERNORATES.items():
    for n in names:
        if re.search(r"[A-Za-z]", n):
            _PATTERNS.append((gov, re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE)))
        else:  # Arabic script: allow common prefixes (و، ب، ل، في) glued to the name
            _PATTERNS.append((gov, re.compile(rf"(?:^|[\s\W])(?:[وبلف]|في\s)?{re.escape(n)}(?:$|[\s\W])")))


def find_governorates(text):
    """All governorates mentioned in the text (by governorate or main-city name)."""
    found = []
    for gov, pat in _PATTERNS:
        if gov not in found and pat.search(text or ""):
            found.append(gov)
    return found
