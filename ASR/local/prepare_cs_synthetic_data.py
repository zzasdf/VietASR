#!/usr/bin/env python3
"""
Generate Code-Switching (CS) vocabulary and sentences for Call Center domain.
Creates Vietnamese sentences containing common English terms used in customer service.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


# ============================================================================
# CS VOCABULARY FOR CALL CENTER DOMAIN
# ============================================================================

CS_VOCABULARY = {
    # Telecom / Internet
    "telecom": [
        "sim", "data", "wifi", "router", "modem", "internet", "network", "signal",
        "roaming", "package", "combo", "gói", "top up", "hotline", "call center",
        "bandwidth", "speed", "download", "upload", "streaming", "buffer",
        "connection", "disconnect", "reconnect", "billing", "invoice", "account",
        "prepaid", "postpaid", "subscriber", "upgrade", "downgrade", "cancel",
        "activate", "deactivate", "block", "unblock", "port", "number", "imei",
        "phone", "mobile", "smartphone", "device", "handset", "gadget",
        "4G", "5G", "LTE", "fiber", "ADSL", "FTTH", "broadband",
    ],
    
    # Banking / Finance
    "banking": [
        "bank", "account", "balance", "transfer", "transaction", "deposit", "withdraw",
        "ATM", "card", "credit", "debit", "visa", "mastercard", "PIN", "OTP",
        "online", "mobile banking", "app", "payment", "fee", "charge", "limit",
        "loan", "interest", "rate", "installment", "mortgage", "insurance",
        "saving", "investment", "stock", "bond", "fund", "portfolio",
        "exchange", "currency", "USD", "VND", "EUR", "convert",
        "statement", "history", "reference", "confirm", "verify", "authenticate",
        "block", "freeze", "unlock", "reset", "password", "security",
    ],
    
    # E-commerce / Delivery
    "ecommerce": [
        "order", "cart", "checkout", "payment", "COD", "delivery", "shipping",
        "tracking", "status", "pending", "processing", "shipped", "delivered",
        "return", "refund", "exchange", "warranty", "voucher", "coupon", "discount",
        "sale", "flash sale", "promo", "promotion", "deal", "offer", "free ship",
        "seller", "buyer", "shop", "store", "mall", "marketplace",
        "review", "rating", "feedback", "comment", "like", "share",
        "size", "color", "stock", "sold out", "available", "quantity",
        "address", "contact", "phone", "email", "name", "update",
    ],
    
    # Tech Support
    "tech_support": [
        "support", "help", "assist", "guide", "tutorial", "manual", "FAQ",
        "issue", "problem", "error", "bug", "fix", "solve", "resolve",
        "install", "uninstall", "update", "upgrade", "version", "software",
        "app", "application", "system", "setting", "config", "option",
        "login", "logout", "sign in", "sign up", "register", "account",
        "password", "username", "email", "verify", "confirm", "reset",
        "backup", "restore", "recover", "delete", "clear", "cache",
        "screen", "display", "touch", "button", "menu", "icon",
        "restart", "reboot", "shutdown", "power", "battery", "charge",
        "connect", "sync", "bluetooth", "wifi", "NFC", "GPS",
    ],
    
    # Customer Service General
    "customer_service": [
        "customer", "service", "support", "care", "center", "agent", "staff",
        "request", "complaint", "feedback", "suggest", "report", "ticket",
        "wait", "hold", "queue", "transfer", "connect", "line", "extension",
        "information", "detail", "update", "check", "verify", "confirm",
        "policy", "term", "condition", "rule", "regulation", "guideline",
        "process", "procedure", "step", "instruction", "form", "document",
        "ID", "CMND", "CCCD", "passport", "license", "contract",
        "deadline", "expire", "extend", "renew", "maintain", "cancel",
        "free", "charge", "cost", "price", "tax", "total",
        "OK", "yes", "no", "please", "sorry", "thank", "welcome",
    ],
    
    # Common Actions
    "actions": [
        "check", "confirm", "verify", "update", "change", "modify", "edit",
        "add", "remove", "delete", "create", "open", "close", "cancel",
        "start", "stop", "pause", "resume", "continue", "finish", "complete",
        "send", "receive", "forward", "reply", "copy", "print", "save",
        "download", "upload", "import", "export", "sync", "backup",
        "search", "find", "filter", "sort", "select", "choose", "pick",
        "enter", "input", "type", "click", "tap", "press", "hold",
        "call", "ring", "answer", "hang up", "dial", "redial", "miss",
    ],
}


# ============================================================================
# SENTENCE TEMPLATES FOR CALL CENTER
# ============================================================================

TEMPLATES = {
    # Opening / Greeting
    "greeting": [
        "Xin chào quý khách đã gọi đến {domain} {company}",
        "Dạ em chào anh chị ạ em là {name} từ bộ phận {department}",
        "Cảm ơn anh chị đã liên hệ {hotline} của {company}",
    ],
    
    # Request Information
    "request_info": [
        "Anh chị cho em xin {info_type} để em {action} giúp ạ",
        "Em cần {verify} thông tin {account} của anh chị",
        "Anh chị vui lòng cung cấp số {id_type} để em {check}",
        "Để em {process} thì anh chị cho em biết {detail} ạ",
    ],
    
    # Check / Verify
    "check_status": [
        "Em {check} thấy {item} của anh chị đang ở trạng thái {status}",
        "Theo hệ thống thì {account} anh chị hiện tại {state}",
        "Em đã {verify} và thấy {issue} do {reason}",
        "Dạ em {confirm} lại là anh chị muốn {action} đúng không ạ",
    ],
    
    # Problem / Issue
    "problem": [
        "Anh chị đang gặp {issue} với {service} phải không ạ",
        "Dạ để em {check} xem {error} là do đâu ạ",
        "Trường hợp này em sẽ {escalate} lên bộ phận {department} để {resolve}",
        "Em rất xin lỗi vì anh chị phải {wait} lâu như vậy",
    ],
    
    # Solution / Action
    "solution": [
        "Em sẽ {action} {item} cho anh chị ngay ạ",
        "Anh chị vui lòng {step} để em {complete} thủ tục",
        "Để {fix} vấn đề này anh chị cần {do}",
        "Em đã {done} xong rồi anh chị {verify} lại giúp em",
    ],
    
    # Telecom Specific
    "telecom": [
        "Gói {package} của anh chị còn {amount} {unit} {data}",
        "Em sẽ {activate} gói {combo} mới cho anh chị",
        "Để {upgrade} lên gói {plan} anh chị chỉ cần {action}",
        "Anh chị đang dùng {sim} loại {type} với số {number}",
        "Dịch vụ {internet} của anh chị sẽ được {renew} vào ngày {date}",
    ],
    
    # Banking Specific
    "banking": [
        "Tài khoản {account} của anh chị hiện có số dư {balance}",
        "Giao dịch {transfer} của anh chị đã được {confirm} thành công",
        "Em sẽ {block} thẻ {card} ngay để đảm bảo {security}",
        "Mã {OTP} sẽ được gửi về {phone} của anh chị trong vòng {time}",
        "Để {reset} mật khẩu {password} anh chị vui lòng {verify}",
    ],
    
    # E-commerce Specific
    "ecommerce": [
        "Đơn hàng {order} của anh chị đang ở trạng thái {status}",
        "Em sẽ {refund} lại {amount} về {account} của anh chị",
        "Mã {voucher} đã được {apply} giảm {discount} cho đơn hàng",
        "Sản phẩm sẽ được {delivery} đến {address} trong {time}",
        "Anh chị muốn {return} hay {exchange} sản phẩm này ạ",
    ],
    
    # Closing
    "closing": [
        "Dạ anh chị còn cần {support} thêm gì không ạ",
        "Cảm ơn anh chị đã sử dụng dịch vụ của {company}",
        "Nếu cần {help} thêm anh chị gọi lại {hotline} nhé",
        "Chúc anh chị một ngày {good} em xin phép kết thúc cuộc gọi",
    ],
}


def get_all_cs_words() -> List[str]:
    """Get all unique CS words from vocabulary."""
    all_words = set()
    for category_words in CS_VOCABULARY.values():
        for word in category_words:
            all_words.add(word.lower())
    return sorted(list(all_words))


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

# Vietnamese number to words
NUMBER_WORDS = {
    '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn',
    '5': 'năm', '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín',
    '10': 'mười', '11': 'mười một', '12': 'mười hai', '15': 'mười lăm',
    '20': 'hai mươi', '24': 'hai bốn', '30': 'ba mươi', '50': 'năm mươi',
    '100': 'một trăm', '500': 'năm trăm', '1000': 'một nghìn',
}

UNIT_WORDS = {
    'GB': 'ghi bai', 'MB': 'mê bai', 'KB': 'ki bai',
    'K': 'nghìn', 'k': 'nghìn',
    '%': 'phần trăm',
}


def normalize_number(text: str) -> str:
    """Convert numbers to Vietnamese words."""
    import re
    
    # Handle common patterns like 2GB, 500MB, 100K, 10%
    def replace_with_unit(match):
        num = match.group(1)
        unit = match.group(2)
        num_word = NUMBER_WORDS.get(num, num)
        unit_word = UNIT_WORDS.get(unit, unit)
        return f"{num_word} {unit_word}"
    
    text = re.sub(r'(\d+)(GB|MB|KB|K|k|%)', replace_with_unit, text)
    
    # Handle standalone numbers
    def replace_number(match):
        num = match.group(0)
        return NUMBER_WORDS.get(num, num)
    
    text = re.sub(r'\b(\d+)\b', replace_number, text)
    
    return text


def normalize_text_for_asr(text: str) -> str:
    """
    Full text normalization for ASR training:
    1. Lowercase
    2. Number to words  
    3. Hybrid tone normalization (Vietnamese diacritics)
    """
    import sys
    from pathlib import Path
    
    # Add utils to path
    utils_path = Path(__file__).parent.parent.parent / "utils"
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    
    from normalize_hybrid import normalize_hybrid_tone
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Number to words
    text = normalize_number(text)
    
    # Step 3: Hybrid tone normalization
    text = normalize_hybrid_tone(text)
    
    # Step 4: Clean up whitespace
    text = ' '.join(text.split())
    
    return text


def generate_sentence(template: str, cs_words: List[str]) -> Tuple[str, List[str]]:
    """
    Generate a sentence by filling template placeholders with CS words.
    Returns the sentence and list of CS words used.
    """
    used_cs_words = []
    result = template
    
    # Common Vietnamese fillers
    fillers = {
        "domain": ["call center", "hotline", "support", "service"],
        "company": ["Viettel", "VNPT", "MobiFone", "FPT", "Techcombank", "VPBank", "Shopee", "Tiki", "Lazada"],
        "name": ["Hương", "Trang", "Linh", "Nam", "Minh"],
        "department": ["chăm sóc khách hàng", "kỹ thuật", "bán hàng", "support"],
        "hotline": ["hotline", "tổng đài", "call center"],
        "info_type": ["email", "số phone", "account", "mã ID", "số CMND"],
        "action": ["check", "verify", "update", "process"],
        "verify": ["verify", "check", "confirm", "xác nhận"],
        "account": ["account", "tài khoản", "số điện thoại"],
        "id_type": ["CMND", "CCCD", "passport", "ID"],
        "check": ["check", "verify", "xác minh"],
        "process": ["process", "xử lý", "giải quyết"],
        "detail": ["chi tiết", "thông tin", "info"],
        "item": ["gói cước", "tài khoản", "đơn hàng", "dịch vụ", "order", "package", "account"],
        "status": ["pending", "processing", "active", "inactive", "block", "cancel"],
        "state": ["đang active", "bị block", "hết hạn", "còn hiệu lực"],
        "issue": ["lỗi", "vấn đề", "error", "issue", "problem"],
        "reason": ["lỗi hệ thống", "chưa verify", "thiếu thông tin", "quá hạn"],
        "confirm": ["confirm", "xác nhận", "check"],
        "service": ["internet", "mobile", "banking", "app"],
        "error": ["error", "lỗi", "bug"],
        "escalate": ["transfer", "chuyển", "báo cáo"],
        "resolve": ["fix", "xử lý", "giải quyết"],
        "wait": ["chờ", "wait", "hold"],
        "step": ["nhấn confirm", "click vào link", "nhập OTP", "verify email"],
        "complete": ["hoàn tất", "complete", "finish"],
        "fix": ["fix", "sửa", "resolve"],
        "do": ["reset password", "verify lại", "update info", "restart app"],
        "done": ["update", "fix", "process", "reset"],
        "package": ["MAX", "BIG", "ST", "D", "V", "COMBO"],
        "amount": ["500MB", "2GB", "5GB", "unlimited"],
        "unit": ["data", "phút gọi", "tin nhắn"],
        "data": ["data", "dữ liệu"],
        "activate": ["activate", "kích hoạt", "đăng ký"],
        "combo": ["combo 4G", "combo data", "gói MAX"],
        "upgrade": ["upgrade", "nâng cấp"],
        "plan": ["gói VIP", "gói MAX", "gói COMBO", "plan"],
        "sim": ["sim", "số"],
        "type": ["prepaid", "postpaid", "trả trước", "trả sau"],
        "number": ["điện thoại", "tài khoản"],
        "internet": ["internet", "wifi", "fiber", "4G"],
        "renew": ["renew", "gia hạn", "tự động gia hạn"],
        "date": ["mai", "tuần sau", "cuối tháng"],
        "balance": ["2 triệu", "500 nghìn", "100 nghìn", "balance"],
        "transfer": ["chuyển tiền", "transfer", "giao dịch"],
        "block": ["block", "khóa", "tạm dừng"],
        "card": ["thẻ ATM", "thẻ credit", "visa", "mastercard"],
        "security": ["an toàn", "security", "bảo mật"],
        "OTP": ["OTP", "mã xác thực"],
        "phone": ["số điện thoại đăng ký", "phone"],
        "time": ["2 phút", "5 phút", "24 giờ"],
        "reset": ["reset", "đặt lại"],
        "password": ["password", "mật khẩu", "PIN"],
        "order": ["đơn hàng", "order", "mã đơn"],
        "refund": ["refund", "hoàn tiền"],
        "voucher": ["voucher", "mã giảm giá", "coupon"],
        "apply": ["apply", "áp dụng"],
        "discount": ["10%", "20%", "50K", "100K"],
        "delivery": ["giao", "ship", "delivery"],
        "address": ["địa chỉ đăng ký", "address"],
        "return": ["trả hàng", "return"],
        "exchange": ["đổi hàng", "exchange"],
        "support": ["hỗ trợ", "support", "giúp đỡ"],
        "help": ["help", "hỗ trợ", "tư vấn"],
        "good": ["tốt lành", "vui vẻ", "may mắn"],
    }
    
    # Find and replace placeholders
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    
    for ph in placeholders:
        if ph in fillers:
            replacement = random.choice(fillers[ph])
            # Track if replacement is a CS word
            if replacement.lower() in [w.lower() for w in cs_words]:
                used_cs_words.append(replacement.lower())
            result = result.replace('{' + ph + '}', replacement, 1)
    
    return result, used_cs_words


def generate_cs_sentences(num_sentences: int = 1000, seed: int = 42) -> List[Dict]:
    """Generate CS sentences for call center domain."""
    random.seed(seed)
    
    cs_words = get_all_cs_words()
    all_templates = []
    for category, templates in TEMPLATES.items():
        for t in templates:
            all_templates.append((category, t))
    
    sentences = []
    for i in range(num_sentences):
        # Pick random template
        category, template = random.choice(all_templates)
        
        # Generate sentence
        sentence, cs_used = generate_sentence(template, cs_words)
        
        sentences.append({
            "id": f"cs_callcenter_{i:05d}",
            "text": sentence,
            "category": category,
            "cs_words": cs_used,
        })
    
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Generate CS sentences for call center")
    parser.add_argument("--num-sentences", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("data4/cs_augmentation/sentences"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get vocabulary
    cs_words = get_all_cs_words()
    print(f"Total unique CS words: {len(cs_words)}")
    
    # Save vocabulary
    vocab_file = args.output_dir / "cs_vocabulary.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(cs_words),
            "by_category": {cat: words for cat, words in CS_VOCABULARY.items()},
            "all_words": cs_words,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {vocab_file}")
    
    # Generate sentences
    print(f"Generating {args.num_sentences} sentences...")
    sentences = generate_cs_sentences(args.num_sentences, args.seed)
    
    # Apply text normalization
    print("Applying text normalization (lowercase, numbers, tones)...")
    for s in sentences:
        s["text"] = normalize_text_for_asr(s["text"])
    
    # Save sentences
    sentences_file = args.output_dir / "cs_sentences.jsonl"
    with open(sentences_file, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(sentences)} sentences to {sentences_file}")
    
    # Stats
    cs_used_count = sum(1 for s in sentences if s["cs_words"])
    print(f"\nStats:")
    print(f"  - Sentences with CS words: {cs_used_count}/{len(sentences)}")
    print(f"  - Categories: {len(TEMPLATES)}")
    
    # Sample
    print(f"\nSample sentences:")
    for s in random.sample(sentences, min(5, len(sentences))):
        print(f"  [{s['category']}] {s['text']}")


if __name__ == "__main__":
    main()
