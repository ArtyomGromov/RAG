from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import pickle

PICKLE_PATH = "qdrant_points.pkl"    
COLLECTION = "pdfminer_all-MiniLM-L6-v2"    
HOST = "qdrant"   # для контейнера внутри docker-compose — имя сервиса, не 127.0.0.1
PORT = 6333
BATCH_SIZE = 512

# 1. Загрузка points
with open(PICKLE_PATH, "rb") as f:
    all_points = pickle.load(f)
print(f"Загружено {len(all_points)} points")

# 2. Qdrant
qdrant = QdrantClient(host=HOST, port=PORT, timeout=30)

# 3. Создание коллекции, если её нет
VECTOR_SIZE = len(all_points[0].vector)
existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION not in existing:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("Коллекция создана!")
else:
    print("Коллекция уже существует")

# 4. Батчево заливаем points
points_for_upload = []
for p in all_points:
    if isinstance(p, dict):
        points_for_upload.append(PointStruct(**p))
    elif isinstance(p, PointStruct):
        points_for_upload.append(p)
    elif hasattr(p, "id") and hasattr(p, "vector") and hasattr(p, "payload"):
        points_for_upload.append(
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
        )
    else:
        print("Неизвестный формат:", p)
        continue

print("Финальное число points:", len(points_for_upload))

for i in range(0, len(points_for_upload), BATCH_SIZE):
    batch = points_for_upload[i:i+BATCH_SIZE]
    qdrant.upsert(collection_name=COLLECTION, points=batch)
    print(f"Загружено: {i+len(batch)}/{len(points_for_upload)}")
