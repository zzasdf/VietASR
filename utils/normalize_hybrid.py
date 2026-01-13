import unicodedata
import re

def normalize_hybrid_tone(text):
    """
    Chuẩn hóa dấu tiếng Việt theo quy tắc hỗn hợp (Hybrid) để khớp với vocab hiện tại:
    1. Âm tiết mở (kết thúc bằng nguyên âm): Dùng kiểu Cũ (hòa, hóa, thủy).
    2. Âm tiết đóng (kết thúc bằng phụ âm/bán nguyên âm): Dùng kiểu Mới (hoàn, toàn, huỳnh).
    3. Ngoại lệ: 'qu' + 'y' -> Kiểu Mới (quý, quỹ).
    """
    text = unicodedata.normalize('NFC', text)
    words = text.split()
    normalized_words = []
    
    # Mapping cho kiểu Cũ (Open syllables)
    # oa -> òa, óa...
    # oe -> òe, óe...
    # uy -> ùy, úy...
    
    # Mapping cho kiểu Mới (Closed syllables)
    # oa -> oà, oá...
    # oe -> oè, oé...
    # uy -> uỳ, uý...
    
    # Hàm chuyển đổi đơn giản
    def to_old_style(word):
        mapping = {
            'oà': 'òa', 'oá': 'óa', 'oả': 'ỏa', 'oã': 'õa', 'oạ': 'ọa',
            'oè': 'òe', 'oé': 'óe', 'oẻ': 'ỏe', 'oẽ': 'õe', 'oẹ': 'ọe',
            'uỳ': 'ùy', 'uý': 'úy', 'uỷ': 'ủy', 'uỹ': 'ũy', 'uỵ': 'ụy'
        }
        res = word
        for k, v in mapping.items():
            res = res.replace(k, v)
        return res

    def to_new_style(word):
        mapping = {
            'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
            'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
            'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ'
        }
        res = word
        for k, v in mapping.items():
            res = res.replace(k, v)
        return res

    for word in words:
        # Kiểm tra xem từ có chứa oa, oe, uy không
        lower_word = word.lower()
        
        # Check 'uy'
        if 'uy' in lower_word or 'uý' in lower_word or 'ùy' in lower_word or 'uỷ' in lower_word or 'uỹ' in lower_word or 'uỵ' in lower_word:
            # Ngoại lệ: Quý, Quỹ -> New Style
            if lower_word.startswith('qu'):
                normalized_words.append(to_new_style(word))
                continue
                
            # Check closed syllable (có phụ âm cuối)
            # Các phụ âm cuối tiếng Việt: c, ch, m, n, ng, nh, p, t
            # Hoặc bán nguyên âm cuối (nhưng uy là bán nguyên âm rồi nên ít khi có bán nguyên âm nữa, trừ khi uya?)
            # Regex đơn giản: nếu sau 'uy' còn ký tự chữ cái -> Closed
            # Cần cẩn thận với dấu thanh.
            
            # Đơn giản hóa: Nếu kết thúc bằng 'y' (có dấu) -> Open -> Old Style
            # Nếu sau 'y' còn ký tự khác -> Closed -> New Style
            
            # Remove tone marks to check ending
            base_word = ''.join(c for c in unicodedata.normalize('NFD', lower_word) if unicodedata.category(c) != 'Mn')
            
            if base_word.endswith('uy'):
                normalized_words.append(to_old_style(word))
            else:
                normalized_words.append(to_new_style(word))
            continue

        # Check 'oa', 'oe'
        # Tương tự: Open -> Old, Closed -> New
        base_word = ''.join(c for c in unicodedata.normalize('NFD', lower_word) if unicodedata.category(c) != 'Mn')
        
        if 'oa' in base_word or 'oe' in base_word:
            if base_word.endswith('oa') or base_word.endswith('oe'):
                normalized_words.append(to_old_style(word))
            else:
                normalized_words.append(to_new_style(word))
            continue
            
        normalized_words.append(word)

    return ' '.join(normalized_words)

if __name__ == "__main__":
    test_cases = [
        "hòa bình", "hoàn thành", "văn hóa", "kế hoạch",
        "thủy lợi", "huỳnh quang", "quý khách", "tùy ý",
        "tòa nhà", "toàn bộ", "khỏe mạnh", "khoản thu"
    ]
    print("Test Hybrid Normalization:")
    for t in test_cases:
        print(f"{t} -> {normalize_hybrid_tone(t)}")
