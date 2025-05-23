FROM python:3.10

WORKDIR /app

COPY . /app
RUN pip install "numpy<2.0"

RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & python bot.py"]
