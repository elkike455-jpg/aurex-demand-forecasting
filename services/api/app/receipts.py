import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from .schemas import ReceiptEmailRequest


def money(value: float, currency: str = "usd") -> str:
    return f"{currency.upper()} ${value:,.2f}"


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text_line(text: str, x: int, y: int, size: int = 11) -> str:
    return f"BT /F1 {size} Tf {x} {y} Td ({_pdf_escape(text)}) Tj ET\n"


def build_receipt_pdf(payload: ReceiptEmailRequest) -> bytes:
    lines = [
        ("AUREX GLOBAL", 50, 785, 18),
        ("Ticket de compra", 50, 762, 14),
        (f"Pedido: {payload.order_number}", 50, 735, 12),
        (f"Fecha: {payload.created_at}", 50, 718, 10),
        (f"Cliente: {payload.customer_name or payload.customer_email}", 50, 701, 10),
        (f"Email: {payload.customer_email}", 50, 684, 10),
        (f"Estado: {payload.status} | Pago: {payload.payment_status}", 50, 667, 10),
        (f"Transaccion: {payload.provider_transaction_id or 'simulada'}", 50, 650, 10),
    ]

    address = payload.shipping_address
    if address:
      lines.extend([
          ("Envio", 50, 625, 12),
          (f"{address.address or 'Direccion no capturada'}", 50, 608, 10),
          (f"{address.city or ''} {address.zip or ''}".strip(), 50, 591, 10),
      ])

    y = 558
    lines.append(("Producto / Vendedor / Cantidad / Total", 50, y, 11))
    y -= 18
    for item in payload.items:
        total = item.price * item.quantity
        product_line = f"{item.name[:48]} | {item.seller_name[:24]} | x{item.quantity} | {money(total, payload.currency)}"
        lines.append((product_line, 50, y, 9))
        y -= 16
        if y < 120:
            lines.append(("Mas productos en el pedido. Consulta AUREX para el detalle completo.", 50, y, 9))
            break

    lines.extend([
        ("Resumen", 360, 160, 12),
        (f"Subtotal: {money(payload.subtotal, payload.currency)}", 360, 140, 10),
        (f"Envio: {money(payload.shipping, payload.currency) if payload.shipping else 'Gratis'}", 360, 124, 10),
        (f"Total: {money(payload.total, payload.currency)}", 360, 104, 12),
        ("Gracias por comprar en AUREX.", 50, 64, 10),
    ])

    stream = "".join(_pdf_text_line(*line) for line in lines)
    stream_bytes = stream.encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream_bytes)).encode() + b" >>\nstream\n" + stream_bytes + b"endstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode())
    return bytes(pdf)


def send_receipt_email(payload: ReceiptEmailRequest) -> tuple[bool, str, str | None]:
    pdf = build_receipt_pdf(payload)
    filename = f"{payload.order_number}.pdf"

    smtp_host = os.getenv("SMTP_HOST")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM", "tickets@aurex.local")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_tls = os.getenv("SMTP_TLS", "true").lower() != "false"

    if not smtp_host or not smtp_user or not smtp_password:
        outbox = Path(os.getenv("AUREX_RECEIPT_OUTBOX", "outbox/receipts"))
        outbox.mkdir(parents=True, exist_ok=True)
        path = outbox / filename
        path.write_bytes(pdf)
        return False, "SMTP no configurado. PDF guardado en outbox local.", str(path)

    message = EmailMessage()
    message["Subject"] = f"Tu ticket AUREX {payload.order_number}"
    message["From"] = smtp_from
    message["To"] = payload.customer_email
    message.set_content(
        "\n".join([
            f"Hola {payload.customer_name or 'cliente'},",
            "",
            f"Adjuntamos tu ticket de compra {payload.order_number}.",
            f"Total: {money(payload.total, payload.currency)}",
            "",
            "Gracias por comprar en AUREX.",
        ])
    )
    message.add_attachment(pdf, maintype="application", subtype="pdf", filename=filename)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
        if smtp_tls:
            smtp.starttls()
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(message)

    return True, f"Ticket enviado a {payload.customer_email} el {datetime.utcnow().isoformat()}Z.", None
