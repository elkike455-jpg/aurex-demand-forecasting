import os

import stripe
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .receipts import send_receipt_email
from .schemas import CheckoutSessionRequest, CheckoutSessionResponse, ReceiptEmailRequest, ReceiptEmailResponse

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

app = FastAPI(title="AUREX API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("AUREX_WEB_ORIGIN", "http://127.0.0.1:5174")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "aurex-api"}


@app.post("/payments/stripe/checkout-session", response_model=CheckoutSessionResponse)
def create_checkout_session(payload: CheckoutSessionRequest):
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe is not configured")

    session = stripe.checkout.Session.create(
        mode="payment",
        customer_email=payload.customer_email,
        line_items=[
            {
                "quantity": item.quantity,
                "price_data": {
                    "currency": payload.currency,
                    "unit_amount": int(round(item.price * 100)),
                    "product_data": {"name": item.name, "metadata": {"sku": item.sku}},
                },
            }
            for item in payload.items
        ],
        success_url=f"{os.getenv('AUREX_WEB_ORIGIN', 'http://127.0.0.1:5174')}/checkout?payment=success",
        cancel_url=f"{os.getenv('AUREX_WEB_ORIGIN', 'http://127.0.0.1:5174')}/checkout?payment=cancelled",
        metadata={"order_id": payload.order_id},
    )
    return CheckoutSessionResponse(session_id=session.id, url=session.url)


@app.post("/payments/stripe/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(default="")):
    secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="Stripe webhook secret is not configured")

    body = await request.body()
    try:
        event = stripe.Webhook.construct_event(body, stripe_signature, secret)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook") from exc

    # TODO: Persist payment/order status when database is connected.
    # Handled event types:
    # - checkout.session.completed
    # - payment_intent.succeeded
    # - payment_intent.payment_failed
    # - charge.refunded
    return {"received": True, "type": event["type"]}


@app.post("/receipts/email", response_model=ReceiptEmailResponse)
def email_receipt(payload: ReceiptEmailRequest):
    try:
        sent, message, outbox_path = send_receipt_email(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Receipt email failed") from exc
    return ReceiptEmailResponse(sent=sent, message=message, outbox_path=outbox_path)
