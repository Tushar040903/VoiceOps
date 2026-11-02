"""Dataset loader and relational mapping engine for ZentroMart."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from src.config import DATA_DIR, PRACTICE_DIR, KB_DIR
from src.data.date_utils import parse_date

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads, cleans, and joins all tabular CSVs and practice files."""

    def __init__(self, data_dir: Optional[Path] = None, practice_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir or DATA_DIR)
        self.practice_dir = Path(practice_dir or PRACTICE_DIR)
        self.kb_dir = Path(KB_DIR)

        self.customers: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.order_items: Dict[str, List[Dict[str, Any]]] = {}
        self.products: Dict[str, Dict[str, Any]] = {}
        self.products_by_name: Dict[str, Dict[str, Any]] = {}
        self.pincodes: Dict[str, Dict[str, Any]] = {}
        self.holidays: List[Dict[str, Any]] = []
        self.support_logs: List[Dict[str, Any]] = []

        self._load_all()

    def _read_csv(self, file_path: Path) -> List[Dict[str, str]]:
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    def _load_all(self):
        # 1. Customers
        customers_raw = self._read_csv(self.data_dir / "customers.csv")
        for row in customers_raw:
            cid = str(row["customer_id"]).strip()
            row["is_plus"] = str(row.get("is_plus", "")).lower() == "true"
            self.customers[cid] = row

        # 2. Products
        products_raw = self._read_csv(self.data_dir / "products.csv")
        for row in products_raw:
            sku = str(row["sku"]).strip()
            row["assured"] = str(row.get("assured", "")).lower() == "true"
            try:
                row["price_inr"] = int(row.get("price_inr", 0))
            except (ValueError, TypeError):
                row["price_inr"] = 0
            try:
                row["warranty_months"] = int(row.get("warranty_months", 0))
            except (ValueError, TypeError):
                row["warranty_months"] = 0
            self.products[sku] = row
            # Index by normalized name for fuzzy / lookup matching
            name_key = row.get("product_name", "").strip().lower()
            if name_key:
                self.products_by_name[name_key] = row

        # 3. Order Items
        items_raw = self._read_csv(self.data_dir / "order_items.csv")
        for row in items_raw:
            oid = str(row["order_id"]).strip()
            sku = str(row.get("sku", "")).strip()
            if sku in self.products:
                row["product"] = self.products[sku]
            try:
                row["qty"] = int(row.get("qty", 1))
            except (ValueError, TypeError):
                row["qty"] = 1
            try:
                row["unit_price_inr"] = int(row.get("unit_price_inr", 0))
            except (ValueError, TypeError):
                row["unit_price_inr"] = 0
            self.order_items.setdefault(oid, []).append(row)

        # 4. Orders
        orders_raw = self._read_csv(self.data_dir / "orders.csv")
        for row in orders_raw:
            oid = str(row["order_id"]).strip()
            cid = str(row.get("customer_id", "")).strip()
            row["customer"] = self.customers.get(cid)
            row["items"] = self.order_items.get(oid, [])
            try:
                row["contacts_last_7_days"] = int(row.get("contacts_last_7_days", 0))
            except (ValueError, TypeError):
                row["contacts_last_7_days"] = 0
            try:
                row["total_inr"] = int(row.get("total_inr", 0))
            except (ValueError, TypeError):
                row["total_inr"] = 0
            self.orders[oid] = row

        # 5. Pincodes
        pincodes_raw = self._read_csv(self.data_dir / "pincodes.csv")
        for row in pincodes_raw:
            pin = str(row["pincode"]).strip()
            row["serviceable"] = str(row.get("serviceable", "")).lower() == "true"
            row["cod_available"] = str(row.get("cod_available", "")).lower() == "true"
            row["express_available"] = str(row.get("express_available", "")).lower() == "true"
            self.pincodes[pin] = row

        # 6. Holidays
        self.holidays = self._read_csv(self.data_dir / "holidays.csv")

        # 7. Support Logs
        self.support_logs = self._read_csv(self.data_dir / "support_logs.csv")

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        clean_id = "".join(ch for ch in str(order_id) if ch.isdigit())
        return self.orders.get(clean_id)

    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        clean_id = str(customer_id).strip()
        return self.customers.get(clean_id)

    def get_product(self, sku: str) -> Optional[Dict[str, Any]]:
        clean_sku = str(sku).strip().upper()
        return self.products.get(clean_sku)

    def find_product_by_name(self, query: str) -> Optional[Dict[str, Any]]:
        q = query.strip().lower()
        for name, prod in self.products_by_name.items():
            if q in name or name in q:
                return prod
        return None

    def get_pincode_info(self, pincode: str) -> Optional[Dict[str, Any]]:
        clean_pin = "".join(ch for ch in str(pincode) if ch.isdigit())
        return self.pincodes.get(clean_pin)

    def load_practice_questions(self) -> List[Dict[str, Any]]:
        return self._read_csv(self.practice_dir / "questions.csv")

    def load_practice_conversations(self) -> List[Dict[str, Any]]:
        return self._read_csv(self.practice_dir / "conversations.csv")

