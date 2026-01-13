import unicodedata
import re

def normalize_vietnamese_tone(text):
    """
    Chuẩn hóa vị trí dấu tiếng Việt về kiểu mới (dấu đặt trên nguyên âm chính).
    Ví dụ: hòa -> hoà, thủy -> thuỷ
    """
    # Bảng chuyển đổi các nguyên âm có dấu
    # Kiểu cũ (dấu trên nguyên âm đầu) -> Kiểu mới (dấu trên nguyên âm chính)
    mapping = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
        'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
        'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ',
        'ÒA': 'OÀ', 'ÓA': 'OÁ', 'ỎA': 'OẢ', 'ÕA': 'OÃ', 'ỌA': 'OẠ',
        'ÒE': 'OÈ', 'ÓE': 'OÉ', 'ỎE': 'OẺ', 'ÕE': 'OẼ', 'ỌE': 'OẸ',
        'ÙY': 'UỲ', 'ÚY': 'UÝ', 'ỦY': 'UỶ', 'ŨY': 'UỸ', 'ỤY': 'UỴ'
    }
    
    # Chuẩn hóa Unicode về NFC trước
    text = unicodedata.normalize('NFC', text)
    
    # Thay thế các trường hợp đặc biệt
    for old, new in mapping.items():
        text = text.replace(old, new)
        
    return text

if __name__ == "__main__":
    # Test cases
    test_words = ["hòa", "hóa", "thủy", "tòa", "họa"]
    print("Test chuẩn hóa dấu:")
    for w in test_words:
        normalized = normalize_vietnamese_tone(w)
        print(f"{w} -> {normalized}")
