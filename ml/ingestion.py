"""Nigerian Bank CSV Statement Importer (GTBank, Access, OPay, Kuda) & Receipt Text Parser."""

from __future__ import annotations

import csv
from datetime import datetime
import io
import re
from typing import Any

from ml.data import generate_receipt_benchmark


def parse_ngn_amount(raw_val: str | float | int | None) -> float:
    """Parse Nigerian currency string ('₦19,887.50', 'NGN 42,750.00', '(1,500.00)') to float."""
    if raw_val is None:
        return 0.0
    if isinstance(raw_val, (int, float)):
        return round(float(raw_val), 2)
    text = str(raw_val).strip()
    if not text or text in ("-", "--", "N/A", "nil"):
        return 0.0

    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1]
    if "-" in text:
        is_negative = True

    cleaned = re.sub(r"(?i)(ngn|₦|n(?=\d)|\s|,|\+|-)", "", text)
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if not match:
        return 0.0
    val = float(match.group(0))
    return round(-val if is_negative else val, 2)


def normalize_date_str(raw_date: str) -> str:
    """Normalize diverse Nigerian bank/receipt date strings into ISO 'YYYY-MM-DD'."""
    text = str(raw_date).strip()
    if not text:
        return datetime.now().strftime("%Y-%m-%d")
    # Strip trailing time if present
    date_part = text.split()[0].strip()

    formats = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d-%b-%Y",
        "%d %b %Y",
        "%d/%m/%y",
        "%Y/%m/%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(date_part, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Try full text with time or 2-token date
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            tokens = " ".join(text.split()[:3])
            return datetime.strptime(tokens, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return date_part


def infer_channel_and_merchant(narration: str) -> tuple[str, str]:
    """Extract payment channel and clean merchant name from a Nigerian bank narration string."""
    upper = narration.upper().strip()
    channel = "NIP"
    if any(k in upper for k in ("POS PUR", "POS/", "MONIEPOINT POS", "TERMINAL")):
        channel = "POS"
    elif any(k in upper for k in ("USSD", "*737*", "*901*", "*894*")):
        channel = "USSD"
    elif any(k in upper for k in ("WEB PUR", "CARD PUR", "PAYSTACK", "FLUTTERWAVE")):
        channel = "WEB"
    elif any(k in upper for k in ("STAMP DUTY", "SMS ALERT", "MAINTENANCE FEE", "EMTL", "TRANSFER FEE")):
        channel = "BANK_FEE"

    # Extract clean merchant segment
    cleaned = re.sub(r"^(NIP/[A-Z]+/|POS PUR\s+|WEB PUR\s+|CARD PUR\s+|TRSF TO\s+|USSD/)", "", upper)
    cleaned = re.sub(r"/\d+$", "", cleaned)
    cleaned = re.sub(r"\bREF\s*\d+\b", "", cleaned).strip()
    merchant = " ".join(cleaned.split()[:4]) if cleaned else "UNKNOWN MERCHANT"
    return channel, merchant


def parse_bank_csv(csv_text: str, bank_hint: str = "auto") -> list[dict[str, Any]]:
    """Parse CSV exports from GTBank, Access Bank, OPay, or Kuda into normalized records."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        return []

    headers_map = {h.strip().lower(): h for h in reader.fieldnames if h}

    # Detect bank if auto
    detected_bank = bank_hint
    if bank_hint.lower() == "auto":
        header_set = set(headers_map.keys())
        if "trans. date" in header_set or "value date" in header_set and "remarks" in header_set:
            detected_bank = "GTBank"
        elif "transaction date" in header_set and "narration" in header_set and "debit" in header_set:
            detected_bank = "Access Bank"
        elif "trans. time" in header_set or "channel" in header_set and "trans. type" in header_set:
            detected_bank = "OPay"
        elif "money out" in header_set or "money in" in header_set:
            detected_bank = "Kuda"
        else:
            detected_bank = "GTBank"

    date_keys = ("trans. date", "transaction date", "trans. time", "date", "posting date")
    narr_keys = ("remarks", "narration", "description", "details", "reference")
    debit_keys = ("debit", "money out", "withdrawal", "dr")
    credit_keys = ("credit", "money in", "deposit", "cr")
    amount_keys = ("amount", "transaction amount", "value")

    records: list[dict[str, Any]] = []
    for row in reader:
        raw_date = ""
        for k in date_keys:
            if k in headers_map and row.get(headers_map[k]):
                raw_date = str(row[headers_map[k]])
                break

        narration = ""
        for k in narr_keys:
            if k in headers_map and row.get(headers_map[k]):
                narration = str(row[headers_map[k]]).strip()
                break

        debit_val = 0.0
        for k in debit_keys:
            if k in headers_map and row.get(headers_map[k]):
                debit_val = abs(parse_ngn_amount(row[headers_map[k]]))
                if debit_val > 0:
                    break

        credit_val = 0.0
        for k in credit_keys:
            if k in headers_map and row.get(headers_map[k]):
                credit_val = abs(parse_ngn_amount(row[headers_map[k]]))
                if credit_val > 0:
                    break

        tx_type = "expense"
        amount_ngn = debit_val
        if debit_val == 0.0 and credit_val > 0.0:
            tx_type = "income"
            amount_ngn = credit_val
        elif debit_val == 0.0 and credit_val == 0.0:
            for k in amount_keys:
                if k in headers_map and row.get(headers_map[k]):
                    signed = parse_ngn_amount(row[headers_map[k]])
                    tx_type_col = str(row.get(headers_map.get("trans. type", ""), "")).lower()
                    if signed < 0 or "debit" in tx_type_col or "out" in tx_type_col or "dr" in tx_type_col:
                        tx_type = "expense"
                    elif signed > 0 and ("credit" in tx_type_col or "in" in tx_type_col or "cr" in tx_type_col):
                        tx_type = "income"
                    amount_ngn = abs(signed)
                    break

        if amount_ngn <= 0 or not narration:
            continue

        channel, merchant = infer_channel_and_merchant(narration)
        if "channel" in headers_map and row.get(headers_map["channel"]):
            channel = str(row[headers_map["channel"]]).strip().upper()
        if "merchant" in headers_map and row.get(headers_map["merchant"]):
            merchant = str(row[headers_map["merchant"]]).strip().upper()

        records.append(
            {
                "date": normalize_date_str(raw_date),
                "bank": detected_bank,
                "channel": channel,
                "merchant": merchant,
                "narration": narration,
                "amount_ngn": round(amount_ngn, 2),
                "tx_type": tx_type,
            }
        )

    return records


def extract_receipt_fields(raw_text: str) -> dict[str, Any]:
    """Parse noisy receipt text (pasted or from an external OCR tool; this code does no OCR) from a Nigerian receipt/POS slip into structured fields.

    Returns `{merchant, date, total_ngn, vat_ngn, payment_channel, confidence}`.
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return {
            "merchant": "UNKNOWN",
            "date": "",
            "total_ngn": 0.0,
            "vat_ngn": 0.0,
            "payment_channel": "POS",
            "confidence": 0.0,
        }

    # 1. Extract Merchant: check explicit 'Merchant:' or 'Beneficiary:' tag first, else top non-header line
    merchant = ""
    for ln in lines:
        m_tag = re.search(r"(?i)^(?:merchant|beneficiary|store|biller)\s*:\s*(.+)$", ln)
        if m_tag:
            merchant = m_tag.group(1).strip().upper()
            break
    if not merchant:
        skip_headers = ("OPAY TRANSACTION", "GTBANK NIP", "MONIEPOINT POS RECEIPT", "*** THERMAL")
        for ln in lines:
            up = ln.upper().strip("* -")
            if not up or any(up.startswith(sh) for sh in skip_headers):
                continue
            merchant = up
            break

    # 2. Extract Date
    extracted_date = ""
    date_regex = re.compile(
        r"(?i)(?:date|dt)?\s*:?\s*(\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4})"
    )
    for ln in lines:
        m_dt = date_regex.search(ln)
        if m_dt:
            extracted_date = normalize_date_str(m_dt.group(1))
            break

    # 3. Extract VAT (NGN)
    vat_ngn = 0.0
    vat_found = False
    vat_regex = re.compile(r"(?i)\b(?:vat|tax)\b[^:\n]*:\s*(?:ngn|₦|n)?\s*([\d,]+(?:\.\d{1,2})?)")
    for ln in lines:
        m_vat = vat_regex.search(ln)
        if m_vat:
            vat_ngn = parse_ngn_amount(m_vat.group(1))
            vat_found = True
            break

    # 4. Extract Total (NGN)
    total_ngn = 0.0
    total_found = False
    total_regex = re.compile(
        r"(?i)(?:grand\s+total|total\s+amount|total\s+paid|total\s+charged|amount\s+paid|net\s+payable|^total)\s*:?\s*(?:ngn|₦|n)?\s*([\d,]+(?:\.\d{1,2})?)"
    )
    for ln in lines:
        if re.search(r"(?i)sub[\s-]*total", ln):
            continue
        m_tot = total_regex.search(ln)
        if m_tot:
            total_ngn = parse_ngn_amount(m_tot.group(1))
            total_found = True
            break

    # 5. Extract Payment Channel
    payment_channel = "POS"
    channel_found = False
    joined_upper = raw_text.upper()
    if any(k in joined_upper for k in ("NIP TRANSFER", "CHANNEL: NIP", "PAYMENT CHANNEL: NIP", "MODE: NIP")):
        payment_channel = "NIP"
        channel_found = True
    elif any(k in joined_upper for k in ("USSD", "CHANNEL: USSD", "MODE: USSD")):
        payment_channel = "USSD"
        channel_found = True
    elif any(k in joined_upper for k in ("WEB CARD", "CHANNEL: WEB", "PAYMENT CHANNEL: WEB", "MODE: WEB")):
        payment_channel = "WEB"
        channel_found = True
    elif any(k in joined_upper for k in ("POS", "TERMINAL", "DEBIT CARD")):
        payment_channel = "POS"
        channel_found = True

    # Confidence calculation across the 5 extracted fields
    signals = [
        bool(merchant and merchant != "UNKNOWN"),
        bool(extracted_date),
        total_found and total_ngn > 0,
        vat_found,
        channel_found,
    ]
    confidence = round(sum(signals) / len(signals), 2)

    return {
        "merchant": merchant,
        "date": extracted_date,
        "total_ngn": round(total_ngn, 2),
        "vat_ngn": round(vat_ngn, 2),
        "payment_channel": payment_channel,
        "confidence": confidence,
    }


def evaluate_receipt_parser() -> dict[str, Any]:
    """Evaluate `extract_receipt_fields` on the 40-receipt Nigerian benchmark set."""
    benchmark = generate_receipt_benchmark()
    fields = ("merchant", "date", "total_ngn", "vat_ngn", "payment_channel")
    field_hits = {f: 0 for f in fields}
    full_receipt_hits = 0
    total_fields = len(benchmark) * len(fields)

    for item in benchmark:
        parsed = extract_receipt_fields(item["raw_text"])
        exp = item["expected"]
        all_match = True
        for f in fields:
            if f in ("total_ngn", "vat_ngn"):
                match = abs(float(parsed[f]) - float(exp[f])) < 1e-2
            else:
                match = str(parsed[f]).strip().upper() == str(exp[f]).strip().upper()
            if match:
                field_hits[f] += 1
            else:
                all_match = False
        if all_match:
            full_receipt_hits += 1

    total_matched_fields = sum(field_hits.values())
    per_field_acc = {f: round(field_hits[f] / len(benchmark), 4) for f in fields}

    return {
        "benchmark_receipts": len(benchmark),
        "total_evaluated_fields": total_fields,
        "receipt_field_exact_match_rate": round(total_matched_fields / total_fields, 4),
        "receipt_full_exact_match_rate": round(full_receipt_hits / len(benchmark), 4),
        "per_field_accuracy": per_field_acc,
    }
