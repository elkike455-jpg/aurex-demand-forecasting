train:
	python -m src.train_lstm

evaluate:
	python -m src.evaluate

api:
	uvicorn services.api.app.main:app --reload --port 8000
