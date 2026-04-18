# aurex-demand-forecasting

## Exploratory Demand Regimes (Dr. Yow Step)

Final EDA outputs are available in:
- `reports/regime_eda_final/regime_summary.csv`
- `reports/regime_eda_final/regime_counts.csv`
- `reports/regime_eda_final/plots/`
- `reports/regime_eda_final/section_final.md`

Paper-ready figure pack (6 selected examples):
- `reports/paper_figures/regime_eda_final/M5_intermittent.png`
- `reports/paper_figures/regime_eda_final/M5_transition.png`
- `reports/paper_figures/regime_eda_final/Amazon_intermittent.png`
- `reports/paper_figures/regime_eda_final/Amazon_transition.png`
- `reports/paper_figures/regime_eda_final/Favorita_stable.png`
- `reports/paper_figures/regime_eda_final/Favorita_transition.png`

Run command:
```powershell
.\.venv\Scripts\python.exe -m src.experiments.run_regime_eda --datasets m5,favorita,amazon --n-products 15 --amazon-file Health_and_Household.jsonl.gz --amazon-max-rows 100000 --out-dir reports/regime_eda_final
```

## AUREX Web Admin + Payments

Frontend app:
```powershell
cd apps/web
npm install
npm run dev -- --host 127.0.0.1
```

Admin demo credentials:
- `elkike455@gmail.com` / `aurexroot123`
- `admin@aurex.local` / `admin123`
- `staff@aurex.local` / `staff123`
- `seller@aurex.local` / `seller123`
- `customer@aurex.local` / `customer123`

Admin routes:
- `/admin`
- `/admin/products`
- `/admin/categories`
- `/admin/orders`
- `/admin/customers`
- `/admin/payments`
- `/admin/inventory`
- `/admin/analytics`
- `/admin/settings`

The current web implementation uses a local persistent commerce store in `localStorage` so the admin foundation can be tested without a database. The backend contract and PostgreSQL-oriented schema are prepared in `services/api`.

Payment flow:
- Default local mode records a Stripe-shaped successful payment for development.
- Set `VITE_STRIPE_ENABLED=true` and `VITE_API_BASE_URL=http://127.0.0.1:8000` to use the API checkout-session endpoint.
- Stripe secrets must live only in the backend environment. See `.env.example`.
- After successful checkout, the web app calls `POST /receipts/email` so the API can generate a PDF receipt and email it to the buyer.
- If SMTP is not configured, the API writes the generated PDF to `outbox/receipts` for local testing.

Receipt email setup:
- Configure `SMTP_HOST`, `SMTP_PORT`, `SMTP_TLS`, `SMTP_USER`, `SMTP_PASSWORD`, and `SMTP_FROM` in the API environment.
- SMTP works with providers such as SendGrid SMTP, Mailgun SMTP, Brevo, or Gmail app passwords.
- Never expose SMTP credentials in the frontend.

API skeleton:
```powershell
cd services/api
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\uvicorn app.main:app --reload
```

Database baseline:
- See `services/api/schema.sql` for users, products, categories, orders, order items, payments, inventory movements, and forecast snapshots.
- The schema is designed so future forecasting outputs can be linked by product through `forecast_snapshots`.
