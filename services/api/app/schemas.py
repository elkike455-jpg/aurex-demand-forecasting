from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    superadmin = "superadmin"
    admin = "admin"
    staff = "staff"
    seller = "seller"
    customer = "customer"


class OrderStatus(str, Enum):
    pending = "pending"
    paid = "paid"
    processing = "processing"
    shipped = "shipped"
    delivered = "delivered"
    cancelled = "cancelled"
    refunded = "refunded"


class PaymentStatus(str, Enum):
    pending = "pending"
    succeeded = "succeeded"
    failed = "failed"
    refunded = "refunded"


class CheckoutLineItem(BaseModel):
    product_id: str
    name: str
    sku: str
    quantity: int = Field(gt=0)
    price: float = Field(ge=0)


class CheckoutSessionRequest(BaseModel):
    order_id: str
    customer_email: EmailStr
    currency: str = "usd"
    items: List[CheckoutLineItem]


class CheckoutSessionResponse(BaseModel):
    session_id: str
    url: str


class PaymentRecord(BaseModel):
    order_id: str
    provider: str = "stripe"
    provider_transaction_id: Optional[str] = None
    amount: float
    currency: str = "usd"
    status: PaymentStatus


class ReceiptAddress(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[str] = None
    city: Optional[str] = None
    zip: Optional[str] = None


class ReceiptLineItem(BaseModel):
    product_id: str
    name: str
    sku: str
    quantity: int = Field(gt=0)
    price: float = Field(ge=0)
    seller_name: str = "AUREX Global"


class ReceiptEmailRequest(BaseModel):
    order_id: str
    order_number: str
    customer_email: EmailStr
    customer_name: Optional[str] = None
    status: OrderStatus
    payment_status: PaymentStatus
    provider_transaction_id: Optional[str] = None
    currency: str = "usd"
    subtotal: float = Field(ge=0)
    shipping: float = Field(ge=0)
    total: float = Field(ge=0)
    created_at: str
    shipping_address: Optional[ReceiptAddress] = None
    items: List[ReceiptLineItem]


class ReceiptEmailResponse(BaseModel):
    sent: bool
    message: str
    outbox_path: Optional[str] = None
