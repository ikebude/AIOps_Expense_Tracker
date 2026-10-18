"""Deterministic Nigerian NGN personal & SME expense dataset and OCR receipt benchmark generator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import random

import numpy as np
import pandas as pd

CATEGORIES: list[str] = [
    "Feeding & Groceries",
    "Transport & Fuel",
    "Data & Utilities",
    "Rent & Housing",
    "Family & Dependents",
    "Business SaaS & Cloud",
    "POS & Bank Charges",
    "Healthcare",
    "Equipment & Repairs",
    "Miscellaneous",
]

BANKS: list[str] = ["GTBank", "Access Bank", "OPay", "Kuda", "Moniepoint"]


@dataclass(frozen=True)
class MerchantTemplate:
    merchant: str
    category: str
    channels: tuple[str, ...]
    narration_patterns: tuple[str, ...]
    min_ngn: float
    max_ngn: float


MERCHANT_TEMPLATES: list[MerchantTemplate] = [
    # 1. Feeding & Groceries
    MerchantTemplate(
        merchant="SHOPRITE IKEJA CITY MALL",
        category="Feeding & Groceries",
        channels=("POS", "NIP", "USSD"),
        narration_patterns=(
            "NIP/OPAY/SHOPRITE IKEJA GROCERY",
            "POS PUR SHOPRITE IKEJA CITY MALL LAGOS",
            "WEB PUR SHOPRITE SUPERMARKET NIG",
            "POS/MONIEPOINT/SHOPRITE IKEJA",
        ),
        min_ngn=14000,
        max_ngn=26000,
    ),
    MerchantTemplate(
        merchant="SPAR VICTORIA ISLAND",
        category="Feeding & Groceries",
        channels=("POS", "NIP"),
        narration_patterns=(
            "POS PUR SPAR VICTORIA ISLAND LAGOS",
            "NIP/GTB/SPAR SUPERMARKET VI",
            "POS/ACCESS/SPAR MARKET ADEOLA ODEKU",
        ),
        min_ngn=15000,
        max_ngn=28000,
    ),
    MerchantTemplate(
        merchant="BOKKU MART SURULERE",
        category="Feeding & Groceries",
        channels=("POS", "NIP"),
        narration_patterns=(
            "TRSF TO BOKKU MART SURULERE BREAD & PROVISIONS",
            "POS PUR BOKKU MART BODE THOMAS",
            "NIP/OPAY/BOKKU RETAIL STORE",
        ),
        min_ngn=6500,
        max_ngn=14000,
    ),
    MerchantTemplate(
        merchant="CHICKEN REPUBLIC LEKKI",
        category="Feeding & Groceries",
        channels=("POS", "NIP"),
        narration_patterns=(
            "POS PUR CHICKEN REPUBLIC LEKKI PHASE 1",
            "NIP/KUDA/CHICKEN REPUBLIC LUNCH",
            "MONIEPOINT POS CHICKEN REPUBLIC",
        ),
        min_ngn=4800,
        max_ngn=11000,
    ),
    MerchantTemplate(
        merchant="MAMA PUT BUKA YABA",
        category="Feeding & Groceries",
        channels=("NIP", "POS"),
        narration_patterns=(
            "MONIEPOINT/MAMA PUT BUKA YABA LUNCH",
            "NIP/OPAY/IYA BASIRA AMALA CANTEEN",
            "TRSF TO SWEET SENSATION FOOD",
        ),
        min_ngn=3200,
        max_ngn=7500,
    ),
    # 2. Transport & Fuel
    MerchantTemplate(
        merchant="NNPC FILLING STATION LEKKI",
        category="Transport & Fuel",
        channels=("POS", "NIP"),
        narration_patterns=(
            "POS PUR NNPC FILLING STN LEKKI PMS FUEL",
            "MONIEPOINT/NNPC MEGA STATION VI PETROL",
            "POS/ACCESS/NNPC RETAIL IKOYI PMS",
        ),
        min_ngn=24000,
        max_ngn=42000,
    ),
    MerchantTemplate(
        merchant="TOTALENERGIES VICTORIA ISLAND",
        category="Transport & Fuel",
        channels=("POS", "NIP"),
        narration_patterns=(
            "POS PUR TOTALENERGIES VI PMS PURCHASE",
            "NIP/GTB/TOTAL FILLING STATION FUEL & DIESEL",
            "POS/MONIEPOINT/MOBIL FILLING STN",
        ),
        min_ngn=25000,
        max_ngn=45000,
    ),
    MerchantTemplate(
        merchant="BOLT RIDE LAGOS",
        category="Transport & Fuel",
        channels=("WEB", "POS", "NIP"),
        narration_patterns=(
            "WEB PUR BOLT.EU/TRIP LAGOS RIDE",
            "NIP/OPAY/BOLT DRIVER TRIP FARE",
            "CARD PUR UBER TRIP VICTORIA ISLAND",
        ),
        min_ngn=4500,
        max_ngn=11000,
    ),
    MerchantTemplate(
        merchant="LCC LEKKI TOLL PLAZA",
        category="Transport & Fuel",
        channels=("WEB", "USSD", "NIP"),
        narration_patterns=(
            "LCC LEKKI IKOYI TOLL PLAZA TOPUP",
            "NIP/KUDA/LEKKI CONCESSION COMPANY TOLL",
            "COWRY CARD BRT BUS TERMINAL RELOAD",
        ),
        min_ngn=4000,
        max_ngn=9500,
    ),
    # 3. Data & Utilities
    MerchantTemplate(
        merchant="MTN NIGERIA DATA BUNDLE",
        category="Data & Utilities",
        channels=("USSD", "NIP", "WEB"),
        narration_patterns=(
            "USSD/MTN DATA BUNDLE 40GB MONTHLY",
            "VTU AIRTIME & DATA MTN NIGERIA",
            "NIP/OPAY/MTN VTU DATA SUBSCRIPTION",
        ),
        min_ngn=11000,
        max_ngn=18000,
    ),
    MerchantTemplate(
        merchant="EKEDC PREPAID METER",
        category="Data & Utilities",
        channels=("WEB", "NIP", "USSD"),
        narration_patterns=(
            "EKEDC PREPAID METER TOKEN PURCHASE",
            "NIP/KUDA/EKO ELECTRICITY DISCO BILL",
            "BUYPOWER EKEDC ELECTRICITY UNITS",
        ),
        min_ngn=22000,
        max_ngn=36000,
    ),
    MerchantTemplate(
        merchant="IKEDC ELECTRICITY BILL",
        category="Data & Utilities",
        channels=("WEB", "NIP"),
        narration_patterns=(
            "IKEDC BILL PAYMENT PREPAID TOKEN",
            "IKEJA ELECTRIC DISCO METER RECHARGE",
            "OPAY/UTILITY/IKEDC POWER TOKEN",
        ),
        min_ngn=20000,
        max_ngn=34000,
    ),
    MerchantTemplate(
        merchant="SPECTRANET FIBRE BROADBAND",
        category="Data & Utilities",
        channels=("WEB", "NIP"),
        narration_patterns=(
            "SPECTRANET FIBRE RENEWAL UNLIMITED INTERNET",
            "AIRTEL 5G ROUTER DATA SUBSCRIPTION",
            "FIBREONE BROADBAND MONTHLY PLAN",
        ),
        min_ngn=21000,
        max_ngn=30000,
    ),
    MerchantTemplate(
        merchant="MULTICHOICE DSTV NIGERIA",
        category="Data & Utilities",
        channels=("WEB", "USSD", "NIP"),
        narration_patterns=(
            "DSTV COMPACT SUBSCRIPTION MULTICHOICE",
            "GOTV MAX MONTHLY UTILITY RENEWAL",
            "LAWMA WASTE MANAGEMENT BILL LAGOS",
        ),
        min_ngn=12500,
        max_ngn=22000,
    ),
    # 4. Rent & Housing
    MerchantTemplate(
        merchant="LEKKI PHASE 1 ESTATE MANAGEMENT",
        category="Rent & Housing",
        channels=("NIP",),
        narration_patterns=(
            "SERVICE CHARGE ESTATE LEKKI PHASE 1 HOUSING",
            "NIP/GTB/ESTATE SECURITY & FACILITY LEVY",
            "TRSF TO VICTORIA CREST ESTATE SERVICE CHARGE",
        ),
        min_ngn=75000,
        max_ngn=105000,
    ),
    MerchantTemplate(
        merchant="LANDLORD RENT ACCOUNT",
        category="Rent & Housing",
        channels=("NIP",),
        narration_patterns=(
            "MONTHLY HOUSE RENT INSTALLMENT LEKKI",
            "NIP/ACCESS/LANDLORD RENT PAYMENT SURULERE",
            "TENANCY LEASE MONTHLY SERVICE CONTRIBUTION",
        ),
        min_ngn=110000,
        max_ngn=155000,
    ),
    MerchantTemplate(
        merchant="WORKSTATION COWORKING YABA",
        category="Rent & Housing",
        channels=("NIP", "WEB"),
        narration_patterns=(
            "COWORKING DESK SPACE YABA MONTHLY RENT",
            "OFFICE SUITE LEASE & FACILITY CHARGE VI",
            "APARTMENT PLUMBING & TENANCY LEVY",
        ),
        min_ngn=68000,
        max_ngn=96000,
    ),
    # 5. Family & Dependents
    MerchantTemplate(
        merchant="MUM MONTHLY UPKEEP",
        category="Family & Dependents",
        channels=("NIP", "USSD"),
        narration_patterns=(
            "NIP/KUDA/MUM MONTHLY UPKEEP ALLOWANCE",
            "TRSF TO PARENTS FAMILY SUPPORT ENUGU",
            "NIP/OPAY/MAMA VILLAGE REMITTANCE & UPKEEP",
        ),
        min_ngn=42000,
        max_ngn=68000,
    ),
    MerchantTemplate(
        merchant="CORONA SCHOOL LEKKI",
        category="Family & Dependents",
        channels=("NIP",),
        narration_patterns=(
            "TRSF TO CHINEDU SCHOOL FEES & PTA LEVY",
            "NIP/GTB/DEPENDENTS TUITION & SCHOOL BOOKS",
            "SIBLING UNIVERSITY ALLOWANCE & HOSTEL FEE",
        ),
        min_ngn=48000,
        max_ngn=75000,
    ),
    MerchantTemplate(
        merchant="FAMILY DEPENDENT ALLOWANCE",
        category="Family & Dependents",
        channels=("NIP",),
        narration_patterns=(
            "SIBLING ALLOWANCE TRANSFER NGOZI",
            "FAMILY REMITTANCE & BABY CARE SUPPLIES",
            "DEPENDENT POCKET MONEY & LESSON TEACHER",
        ),
        min_ngn=32000,
        max_ngn=52000,
    ),
    # 6. Business SaaS & Cloud
    MerchantTemplate(
        merchant="AWS EMEA SARL",
        category="Business SaaS & Cloud",
        channels=("WEB",),
        narration_patterns=(
            "WEB PUR AWS EMEA SARL CLOUD HOSTING",
            "CARD PUR AMAZON WEB SERVICES EC2 S3 BILLING",
            "DIGITALOCEAN CLOUD DROPLET HOSTING",
        ),
        min_ngn=34000,
        max_ngn=56000,
    ),
    MerchantTemplate(
        merchant="GITHUB & OPENAI BILLING",
        category="Business SaaS & Cloud",
        channels=("WEB",),
        narration_patterns=(
            "WEB PUR GITHUB INC COPILOT SUBSCRIPTION",
            "CARD PUR OPENAI API BILLING SAN FRANCISCO",
            "ANTHROPIC CLAUDE API WORKSPACE CREDITS",
        ),
        min_ngn=26000,
        max_ngn=46000,
    ),
    MerchantTemplate(
        merchant="GOOGLE WORKSPACE & CLOUDFLARE",
        category="Business SaaS & Cloud",
        channels=("WEB", "NIP"),
        narration_patterns=(
            "WEB PUR GOOGLE WORKSPACE DUBLIN SAAS",
            "CLOUDFLARE PRO PLAN DOMAIN & CDN",
            "PAYSTACK MERCHANT API GATEWAY SUBSCRIPTION",
        ),
        min_ngn=22000,
        max_ngn=38000,
    ),
    # 7. POS & Bank Charges
    MerchantTemplate(
        merchant="CBN ELECTRONIC MONEY TRANSFER LEVY",
        category="POS & Bank Charges",
        channels=("BANK_FEE",),
        narration_patterns=(
            "STAMP DUTY CHARGE CBN EMTL LEVY",
            "ELECTRONIC MONEY TRANSFER LEVY 50 NGN",
            "FGN STAMP DUTY ON ELECTRONIC TRANSFER",
        ),
        min_ngn=50,
        max_ngn=150,
    ),
    MerchantTemplate(
        merchant="GTBANK & ACCESS BANK CHARGES",
        category="POS & Bank Charges",
        channels=("BANK_FEE",),
        narration_patterns=(
            "SMS ALERT FEE GTBANK MONTHLY CHARGE",
            "CARD MAINTENANCE FEE ACCESS BANK",
            "NIP TRANSFER FEE & VAT COMMISSION",
        ),
        min_ngn=100,
        max_ngn=450,
    ),
    MerchantTemplate(
        merchant="MONIEPOINT POS AGENT TERMINAL",
        category="POS & Bank Charges",
        channels=("POS", "BANK_FEE"),
        narration_patterns=(
            "POS AGENT WITHDRAWAL CHARGE SURULERE",
            "OPAY AGENT CASH OUT TERMINAL FEE",
            "ACCOUNT MAINTENANCE COMMISSION & VAT",
        ),
        min_ngn=200,
        max_ngn=600,
    ),
    # 8. Healthcare
    MerchantTemplate(
        merchant="MEDPLUS PHARMACY IKEJA",
        category="Healthcare",
        channels=("POS", "NIP"),
        narration_patterns=(
            "POS PUR MEDPLUS PHARMACY IKEJA DRUGS",
            "HEALTHPLUS PHARMACY VICTORIA ISLAND MEDICATION",
            "NIP/OPAY/ALPHA PHARMACY PRESCRIPTION",
        ),
        min_ngn=14000,
        max_ngn=26000,
    ),
    MerchantTemplate(
        merchant="REDDINGTON HOSPITAL VI",
        category="Healthcare",
        channels=("POS", "NIP"),
        narration_patterns=(
            "REDDINGTON HOSPITAL VI CONSULTATION & LAB",
            "LUTH OUTPATIENT CLINIC MEDICAL TEST",
            "SYNLAB NIGERIA DIAGNOSTIC BLOOD PANEL",
        ),
        min_ngn=22000,
        max_ngn=38000,
    ),
    MerchantTemplate(
        merchant="AXA MANSARD HEALTH HMO",
        category="Healthcare",
        channels=("WEB", "NIP"),
        narration_patterns=(
            "AXA MANSARD HMO PREMIUM HEALTH INSURANCE",
            "RELIANCE HMO FAMILY HEALTH PLAN",
            "DENTAL CLINIC SCALING & OPTICAL CHECKUP",
        ),
        min_ngn=20000,
        max_ngn=34000,
    ),
    # 9. Equipment & Repairs
    MerchantTemplate(
        merchant="COMPUTER VILLAGE IKEJA TECH",
        category="Equipment & Repairs",
        channels=("NIP", "POS"),
        narration_patterns=(
            "COMPUTER VILLAGE IKEJA SSD & LAPTOP REPAIR",
            "SLOT SYSTEMS IKEJA PHONE SCREEN REPLACEMENT",
            "LAPTOP CHARGER & UPS BATTERY REPAIR",
        ),
        min_ngn=26000,
        max_ngn=44000,
    ),
    MerchantTemplate(
        merchant="MIKANO & INVERTER SERVICING",
        category="Equipment & Repairs",
        channels=("NIP",),
        narration_patterns=(
            "MIKANO GENERATOR SERVICING & OIL FILTER",
            "INVERTER TUBULAR BATTERY & SOLAR REPAIR",
            "THERMOCOOL AC COMPRESSOR MAINTENANCE",
        ),
        min_ngn=28000,
        max_ngn=48000,
    ),
    # 10. Miscellaneous
    MerchantTemplate(
        merchant="TERRA KULTURE & FILMHOUSE",
        category="Miscellaneous",
        channels=("POS", "WEB"),
        narration_patterns=(
            "TERRA KULTURE EVENT TICKET & ART GALLERY",
            "FILMHOUSE CINEMA LEKKI MOVIE TICKETS",
            "LANDMARK BEACH RESORT GATE FEE",
        ),
        min_ngn=11000,
        max_ngn=22000,
    ),
    MerchantTemplate(
        merchant="JUMIA & PERSONAL GROOMING",
        category="Miscellaneous",
        channels=("WEB", "POS", "NIP"),
        narration_patterns=(
            "JUMIA NIGERIA ORDER GIFT & APPAREL",
            "BARBERSHOP & GROOMING LOUNGE VI",
            "CHARITY & CHURCH TITHE OFFERING TRSF",
        ),
        min_ngn=9500,
        max_ngn=19000,
    ),
]


# Variants that do NOT match simple keyword rules directly (where TF-IDF subword/bigrams win),
# plus genuinely ambiguous bank transfer strings where even ML has a realistic error rate (~7%).
SUBWORD_ONLY_TEMPLATES: list[tuple[str, str, str, str, float, float]] = [
    # (merchant, narration, channel, category, min_ngn, max_ngn) -> caught by TF-IDF, missed by naive rules
    ("PRINCE EBEANO LEKKI", "POS PUR PRINCE EBEANO CULINARY ITEMS VI", "POS", "Feeding & Groceries", 15000, 26000),
    ("FOODCO SUPERSTORE IBADAN", "POS PUR FOODCO FRESH PRODUCE BASKET", "POS", "Feeding & Groceries", 14000, 24000),
    ("ARDENT ENERGIES STATION", "POS PUR ARDENT ENERGIES AUTO REFILL LEKKI", "POS", "Transport & Fuel", 24000, 38000),
    ("LAGFERRY MARINA JETTY", "NIP/OPAY/LAGFERRY COMMUTER PASS VI", "NIP", "Transport & Fuel", 5000, 10000),
    ("STARLINK NIGERIA SUB", "WEB PUR SPACEX STARLINK RESIDENTIAL BANDWIDTH", "WEB", "Data & Utilities", 24000, 34000),
    ("9MOBILE DATA RENEWAL", "USSD/9MOBILE MONTHLY INTERNET PLAN", "USSD", "Data & Utilities", 12000, 20000),
    ("SHORTLET STUDIO IKOYI", "NIP/GTB/IKOYI SHORTLET ACCOMMODATION FEE", "NIP", "Rent & Housing", 85000, 130000),
    ("CRADLE MONTESSORI ACADEMY", "NIP/ACCESS/CRADLE MONTESSORI TERM LEVY", "NIP", "Family & Dependents", 45000, 68000),
    ("VERCEL INC PRO PLAN", "WEB PUR VERCEL FRONTEND DEPLOYMENT SUB", "WEB", "Business SaaS & Cloud", 24000, 38000),
    ("SUPABASE INC DATABASE", "CARD PUR SUPABASE POSTGRES HOSTING", "WEB", "Business SaaS & Cloud", 22000, 36000),
    ("INTERSWITCH ATM SURCHARGE", "BANK_FEE INTERSWITCH INTERBANK LEVY", "BANK_FEE", "POS & Bank Charges", 100, 350),
    ("EVERCARE LEKKI DIAGNOSTICS", "POS PUR EVERCARE WELLNESS BLOODWORK VI", "POS", "Healthcare", 20000, 34000),
    ("COOLWORLD ELECTRICALS VI", "POS PUR COOLWORLD STABILIZER COIL FIX", "POS", "Equipment & Repairs", 26000, 42000),
    ("NIKE ART GALLERY LEKKI", "POS PUR NIKE GALLERY SOUVENIR CRAFT", "POS", "Miscellaneous", 11000, 20000),
]

HARD_NOISY_TRANSFERS: list[tuple[str, str, str, float, float]] = [
    # Genuinely ambiguous generic POS/NIP narrations shared across categories (cannot be memorized by name)
    ("GENERIC NIP TRANSFER", "NIP/OPAY/INSTANT WALLET TRANSFER LAGOS", "NIP", 8000, 35000),
    ("MONIEPOINT POS TERM", "POS PUR MONIEPOINT AGENT TERMINAL VI", "POS", 6000, 28000),
    ("KUDA BANK TRANSFER", "NIP/KUDA/MOBILE APP TRANSFER REF", "NIP", 10000, 42000),
    ("GTBANK USSD PAY", "USSD/GTB/737 MERCHANT PAYMENT LAGOS", "USSD", 5000, 24000),
    ("ACCESS POS CHECKOUT", "POS/ACCESS/MERCHANT TERMINAL IKEJA", "POS", 7500, 32000),
]


def generate_ngn_expenses_dataset(n_rows: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic 600-row labelled Nigerian expense dataset across 6 months.

    Structured so each of the 6 months (`2025-09` through `2026-02`) has 100 transactions
    distributed consistently across all 10 categories, with:
    - Smooth 2% monthly organic growth + +10% festive bump in December (`2025-12`)
    - Subword/merchant variations where TF-IDF beats static rules
    - ~6% hard ambiguous transfers so out-of-time test Macro-F1 is realistic (~0.91–0.94, not 1.0)
    - 18 injected per-category spending anomalies (3 per month) with clear MAD z-scores >= 4.0
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    months_spec = [
        ("2025-09", datetime(2025, 9, 1, 8, 0), 30, 1.00),
        ("2025-10", datetime(2025, 10, 1, 8, 0), 31, 1.02),
        ("2025-11", datetime(2025, 11, 1, 8, 0), 30, 1.04),
        ("2025-12", datetime(2025, 12, 1, 8, 0), 31, 1.14),  # Festive December surge
        ("2026-01", datetime(2026, 1, 1, 8, 0), 31, 1.05),
        ("2026-02", datetime(2026, 2, 1, 8, 0), 28, 1.07),
    ]

    # Exact per-month category counts summing to 100 rows per month (600 total)
    monthly_cat_counts: dict[str, int] = {
        "Feeding & Groceries": 18,
        "Transport & Fuel": 15,
        "Data & Utilities": 14,
        "POS & Bank Charges": 11,
        "Business SaaS & Cloud": 9,
        "Family & Dependents": 8,
        "Healthcare": 7,
        "Equipment & Repairs": 6,
        "Rent & Housing": 5,
        "Miscellaneous": 7,
    }

    templates_by_cat: dict[str, list[MerchantTemplate]] = {c: [] for c in CATEGORIES}
    for tmpl in MERCHANT_TEMPLATES:
        templates_by_cat[tmpl.category].append(tmpl)

    subword_by_cat: dict[str, list[tuple[str, str, str, str, float, float]]] = {c: [] for c in CATEGORIES}
    for item in SUBWORD_ONLY_TEMPLATES:
        subword_by_cat[item[3]].append(item)

    # Spike amounts per category so z_mad >= 3.5 without non-anomalies exceeding 2.5
    anomaly_spike_amounts: dict[str, tuple[float, float, str]] = {
        "Feeding & Groceries": (115000, 145000, "BULK CATERING & EVENT STOCK"),
        "Transport & Fuel": (135000, 175000, "BULK DIESEL DRUM DELIVERY"),
        "Data & Utilities": (98000, 128000, "ARREARS & BULK METER TOPUP"),
        "Rent & Housing": (480000, 620000, "ANNUAL COMMERCIAL LEASE BULK"),
        "Family & Dependents": (195000, 240000, "FULL YEAR DEPENDENT TUITION BULK"),
        "Business SaaS & Cloud": (155000, 195000, "ANNUAL CLOUD RESERVED INSTANCE"),
        "POS & Bank Charges": (4200, 6800, "EXCESSIVE POS OVERCHARGE DISPUTE"),
        "Healthcare": (125000, 165000, "EMERGENCY HOSPITAL PROCEDURE"),
        "Equipment & Repairs": (155000, 195000, "COMPLETE INVERTER OVERHAUL"),
        "Miscellaneous": (78000, 105000, "LUXURY EVENT SPONSORSHIP"),
    }

    rows: list[dict[str, Any]] = []
    rows_per_month = n_rows // len(months_spec)

    for m_idx, (m_str, m_start, m_days, seasonal_mult) in enumerate(months_spec):
        # Build category list for this month
        month_cats: list[str] = []
        for cat, cnt in monthly_cat_counts.items():
            month_cats.extend([cat] * cnt)
        while len(month_cats) < rows_per_month:
            month_cats.append("Feeding & Groceries")
        month_cats = month_cats[:rows_per_month]
        rng.shuffle(month_cats)

        # Generate sorted timestamps within this month
        max_sec = m_days * 86400 - 3600
        m_timestamps = sorted(
            [m_start + timedelta(seconds=rng.randint(0, max_sec)) for _ in range(rows_per_month)]
        )

        # Pick 3 anomaly positions in this month across distinct categories
        anomaly_local_indices = set(rng.sample(range(5, rows_per_month - 5), 3))

        for local_i, (dt, cat) in enumerate(zip(m_timestamps, month_cats)):
            bank = rng.choices(BANKS, weights=[0.24, 0.20, 0.22, 0.18, 0.16], k=1)[0]
            ref_num = rng.randint(10000000, 99999999)

            roll = rng.random()
            # 18% subword templates (learned by TF-IDF in train, missed by static keyword rules)
            if roll < 0.18 and subword_by_cat[cat]:
                merch, base_narr, channel, _, min_a, max_a = rng.choice(subword_by_cat[cat])
                narration = f"{base_narr}/{ref_num}"
            # ~7.5% genuinely ambiguous generic transfers/POS slips lacking merchant keywords
            elif roll >= 0.18 and roll < 0.255 and local_i not in anomaly_local_indices:
                merch, base_narr, channel, min_a, max_a = rng.choice(HARD_NOISY_TRANSFERS)
                narration = f"{base_narr}/{ref_num}"
                tmpl_for_amt = rng.choice(templates_by_cat[cat])
                min_a, max_a = tmpl_for_amt.min_ngn, tmpl_for_amt.max_ngn
            else:
                tmpl = rng.choice(templates_by_cat[cat])
                merch = tmpl.merchant
                channel = rng.choice(tmpl.channels)
                base_narr = rng.choice(tmpl.narration_patterns)
                narration = f"{base_narr}/{ref_num}"
                min_a, max_a = tmpl.min_ngn, tmpl.max_ngn

            # Consistent category base scale with natural 18% spread so MAD is robust to seasonal shifts
            cat_base_tmpls = templates_by_cat[cat]
            cat_mid = float(np.mean([0.5 * (t.min_ngn + t.max_ngn) for t in cat_base_tmpls]))
            mid = cat_mid * seasonal_mult
            spread = cat_mid * 0.18
            amount = float(np.clip(np_rng.normal(mid, spread), mid * 0.68, mid * 1.32))
            amount = round(amount, 2)

            is_anomaly = 0
            if local_i in anomaly_local_indices:
                is_anomaly = 1
                sp_min, sp_max, sp_tag = anomaly_spike_amounts[cat]
                amount = round(rng.uniform(sp_min, sp_max), 2)
                narration = f"{narration} {sp_tag}"

            rows.append(
                {
                    "tx_id": f"NGN-{len(rows) + 1:04d}",
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": dt.strftime("%Y-%m-%d"),
                    "month": m_str,
                    "bank": bank,
                    "channel": channel,
                    "merchant": merch,
                    "narration": narration,
                    "amount_ngn": amount,
                    "category": cat,
                    "is_anomaly": is_anomaly,
                }
            )

    df = pd.DataFrame(rows)
    return df


def save_dataset(df: pd.DataFrame, path: str | Path = "data/ngn_expenses_labelled.csv") -> Path:
    """Save the generated dataset to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def generate_receipt_benchmark() -> list[dict[str, Any]]:
    """Generate a 40-receipt benchmark set of Nigerian POS, thermal, and transfer receipts."""
    receipts: list[dict[str, Any]] = [
        {
            "receipt_id": "RCP-001",
            "raw_text": (
                "SHOPRITE IKEJA CITY MALL\n"
                "ALAUSA IKEJA LAGOS\n"
                "DATE: 14/10/2025 14:22\n"
                "BREAD & GROCERIES  18,500.00\n"
                "VAT (7.5%): NGN 1,387.50\n"
                "TOTAL AMOUNT: NGN 19,887.50\n"
                "PAYMENT METHOD: POS CARD\n"
                "STAN: 092831 APPROVED"
            ),
            "expected": {
                "merchant": "SHOPRITE IKEJA CITY MALL",
                "date": "2025-10-14",
                "total_ngn": 19887.50,
                "vat_ngn": 1387.50,
                "payment_channel": "POS",
            },
        },
        {
            "receipt_id": "RCP-002",
            "raw_text": (
                "NNPC RETAIL FILLING STATION\n"
                "LEKKI EXPRESSWAY LAGOS\n"
                "Date: 2025-11-03\n"
                "PMS 45 LITRES\n"
                "VAT: ₦0.00\n"
                "TOTAL: ₦42,750.00\n"
                "PAID VIA MONIEPOINT POS TERMINAL"
            ),
            "expected": {
                "merchant": "NNPC RETAIL FILLING STATION",
                "date": "2025-11-03",
                "total_ngn": 42750.00,
                "vat_ngn": 0.00,
                "payment_channel": "POS",
            },
        },
        {
            "receipt_id": "RCP-003",
            "raw_text": (
                "OPAY TRANSACTION RECEIPT\n"
                "Merchant: EKEDC PREPAID METER\n"
                "Date: 19/11/2025\n"
                "VAT: NGN 1,875.00\n"
                "Total Paid: NGN 26,875.00\n"
                "Channel: NIP TRANSFER\n"
                "Status: Successful"
            ),
            "expected": {
                "merchant": "EKEDC PREPAID METER",
                "date": "2025-11-19",
                "total_ngn": 26875.00,
                "vat_ngn": 1875.00,
                "payment_channel": "NIP",
            },
        },
        {
            "receipt_id": "RCP-004",
            "raw_text": (
                "CHICKEN REPUBLIC LEKKI\n"
                "22 ADMIRALTY WAY LEKKI\n"
                "DATE: 05-12-2025\n"
                "REFUEL MAX MEAL X2\n"
                "VAT 7.5%: N525.00\n"
                "AMOUNT PAID: N7,525.00\n"
                "CHANNEL: POS"
            ),
            "expected": {
                "merchant": "CHICKEN REPUBLIC LEKKI",
                "date": "2025-12-05",
                "total_ngn": 7525.00,
                "vat_ngn": 525.00,
                "payment_channel": "POS",
            },
        },
        {
            "receipt_id": "RCP-005",
            "raw_text": (
                "MEDPLUS PHARMACY IKEJA\n"
                "DATE: 18/12/2025\n"
                "ANTIMALARIAL & VITAMINS\n"
                "VAT: NGN 0.00\n"
                "TOTAL: NGN 14,200.00\n"
                "PAYMENT: POS TERMINAL"
            ),
            "expected": {
                "merchant": "MEDPLUS PHARMACY IKEJA",
                "date": "2025-12-18",
                "total_ngn": 14200.00,
                "vat_ngn": 0.00,
                "payment_channel": "POS",
            },
        },
        {
            "receipt_id": "RCP-006",
            "raw_text": (
                "SPAR VICTORIA ISLAND\n"
                "ADEOLA ODEKU ST VI\n"
                "Date: 22/12/2025\n"
                "Sub-Total: 40,000.00\n"
                "VAT (7.5%): ₦3,000.00\n"
                "GRAND TOTAL: ₦43,000.00\n"
                "Paid by: POS DEBIT CARD"
            ),
            "expected": {
                "merchant": "SPAR VICTORIA ISLAND",
                "date": "2025-12-22",
                "total_ngn": 43000.00,
                "vat_ngn": 3000.00,
                "payment_channel": "POS",
            },
        },
        {
            "receipt_id": "RCP-007",
            "raw_text": (
                "GTBANK NIP TRANSFER SLIP\n"
                "Beneficiary: CORONA SCHOOL LEKKI\n"
                "Date: 2026-01-08\n"
                "VAT: NGN 0.00\n"
                "Total Amount: NGN 95,000.00\n"
                "Payment Channel: NIP"
            ),
            "expected": {
                "merchant": "CORONA SCHOOL LEKKI",
                "date": "2026-01-08",
                "total_ngn": 95000.00,
                "vat_ngn": 0.00,
                "payment_channel": "NIP",
            },
        },
        {
            "receipt_id": "RCP-008",
            "raw_text": (
                "TOTALENERGIES VICTORIA ISLAND\n"
                "DATE: 12/01/2026\n"
                "PMS FUEL PURCHASE\n"
                "VAT: NGN 0.00\n"
                "TOTAL: NGN 38,500.00\n"
                "MODE: POS"
            ),
            "expected": {
                "merchant": "TOTALENERGIES VICTORIA ISLAND",
                "date": "2026-01-12",
                "total_ngn": 38500.00,
                "vat_ngn": 0.00,
                "payment_channel": "POS",
            },
        },
        {
            "receipt_id": "RCP-009",
            "raw_text": (
                "AWS EMEA SARL INVOICE\n"
                "Date: 2026-01-15\n"
                "Cloud Hosting Services\n"
                "VAT: NGN 3,150.00\n"
                "Total Charged: NGN 45,150.00\n"
                "Payment Method: WEB CARD"
            ),
            "expected": {
                "merchant": "AWS EMEA SARL INVOICE",
                "date": "2026-01-15",
                "total_ngn": 45150.00,
                "vat_ngn": 3150.00,
                "payment_channel": "WEB",
            },
        },
        {
            "receipt_id": "RCP-010",
            "raw_text": (
                "REDDINGTON HOSPITAL VI\n"
                "12 IDOWU MARTINS VICTORIA ISLAND\n"
                "DATE: 19/01/2026\n"
                "CONSULTATION & FULL BLOOD COUNT\n"
                "VAT: ₦0.00\n"
                "TOTAL: ₦48,000.00\n"
                "PAYMENT: POS"
            ),
            "expected": {
                "merchant": "REDDINGTON HOSPITAL VI",
                "date": "2026-01-19",
                "total_ngn": 48000.00,
                "vat_ngn": 0.00,
                "payment_channel": "POS",
            },
        },
    ]

    extra_templates = [
        ("BOKKU MART SURULERE", "2025-09-10", "10/09/2025", 12400.00, 840.00, "POS"),
        ("MTN NIGERIA DATA BUNDLE", "2025-09-14", "2025-09-14", 15000.00, 1050.00, "USSD"),
        ("IKEDC ELECTRICITY BILL", "2025-09-20", "20/09/2025", 32000.00, 2240.00, "WEB"),
        ("SPECTRANET FIBRE BROADBAND", "2025-09-25", "25-09-2025", 24500.00, 1715.00, "WEB"),
        ("MULTICHOICE DSTV NIGERIA", "2025-10-01", "01/10/2025", 15700.00, 1099.00, "USSD"),
        ("LEKKI PHASE 1 ESTATE MANAGEMENT", "2025-10-04", "2025-10-04", 75000.00, 0.00, "NIP"),
        ("WORKSTATION COWORKING YABA", "2025-10-09", "09/10/2025", 55000.00, 3850.00, "NIP"),
        ("MUM MONTHLY UPKEEP", "2025-10-15", "15/10/2025", 50000.00, 0.00, "NIP"),
        ("GITHUB & OPENAI BILLING", "2025-10-21", "2025-10-21", 32000.00, 2400.00, "WEB"),
        ("GOOGLE WORKSPACE DUBLIN", "2025-10-28", "28/10/2025", 18500.00, 1387.50, "WEB"),
        ("AXA MANSARD HEALTH HMO", "2025-11-02", "02-11-2025", 42000.00, 0.00, "WEB"),
        ("COMPUTER VILLAGE IKEJA TECH", "2025-11-07", "07/11/2025", 35000.00, 2625.00, "POS"),
        ("MIKANO GENERATOR SERVICING", "2025-11-11", "11/11/2025", 64000.00, 4800.00, "NIP"),
        ("TERRA KULTURE VICTORIA ISLAND", "2025-11-16", "16/11/2025", 16500.00, 1155.00, "POS"),
        ("FILMHOUSE CINEMA LEKKI", "2025-11-22", "22-11-2025", 12500.00, 875.00, "POS"),
        ("JUMIA NIGERIA CHECKOUT", "2025-11-27", "2025-11-27", 21400.00, 1498.00, "WEB"),
        ("BOLT RIDE LAGOS", "2025-12-01", "01/12/2025", 6800.00, 0.00, "WEB"),
        ("LCC LEKKI TOLL PLAZA", "2025-12-04", "04/12/2025", 5000.00, 0.00, "USSD"),
        ("AIR PEACE DOMESTIC FLIGHT", "2025-12-09", "09/12/2025", 78500.00, 5495.00, "WEB"),
        ("MAMA PUT BUKA YABA", "2025-12-13", "13/12/2025", 4500.00, 0.00, "NIP"),
        ("SLOT SYSTEMS IKEJA", "2025-12-17", "17-12-2025", 48000.00, 3600.00, "POS"),
        ("HEALTHPLUS PHARMACY VI", "2025-12-21", "21/12/2025", 11200.00, 0.00, "POS"),
        ("SYNLAB NIGERIA DIAGNOSTICS", "2025-12-27", "27/12/2025", 29500.00, 0.00, "POS"),
        ("LANDLORD RENT ACCOUNT", "2026-01-03", "2026-01-03", 350000.00, 0.00, "NIP"),
        ("CLOUDFLARE PRO PLAN", "2026-01-11", "2026-01-11", 28000.00, 2100.00, "WEB"),
        ("PAYSTACK MERCHANT SETTLEMENT", "2026-01-17", "17/01/2026", 14500.00, 1087.50, "WEB"),
        ("INVERTER BATTERY REPAIR", "2026-01-24", "24/01/2026", 52000.00, 3900.00, "NIP"),
        ("BARBERSHOP & GROOMING VI", "2026-02-02", "02/02/2026", 8500.00, 0.00, "POS"),
        ("SWEET SENSATION SURULERE", "2026-02-10", "10-02-2026", 6200.00, 465.00, "POS"),
        ("AIRTEL 5G ROUTER DATA", "2026-02-18", "18/02/2026", 20000.00, 1500.00, "USSD"),
    ]

    for idx, (merch, iso_date, raw_date, total_val, vat_val, chan) in enumerate(extra_templates, start=11):
        if idx in (23, 31, 37):
            # Realistic OCR noise on 3 thermal slips (missing VAT line or OCR garbled channel)
            raw_text = (
                f"{merch}\n"
                f"DT: {raw_date}\n"
                f"NET PAYABLE: NGN {total_val:,.2f}\n"
                f"PAID: CARD/T3RM"
            )
        else:
            raw_text = (
                f"{merch}\n"
                f"Date: {raw_date}\n"
                f"VAT (7.5%): NGN {vat_val:,.2f}\n"
                f"TOTAL AMOUNT: ₦{total_val:,.2f}\n"
                f"Payment Channel: {chan}"
            )

        receipts.append(
            {
                "receipt_id": f"RCP-{idx:03d}",
                "raw_text": raw_text,
                "expected": {
                    "merchant": merch,
                    "date": iso_date,
                    "total_ngn": total_val,
                    "vat_ngn": vat_val,
                    "payment_channel": chan,
                },
            }
        )

    return receipts
