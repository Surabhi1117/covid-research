import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "covid_research"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

def get_mongo_db():
    return db

async def init_mongo():
    """Initialize MongoDB indexes."""
    papers = db.papers
    # Create text index for keyword search
    await papers.create_index([("title", "text"), ("abstract", "text")])
    # Unique index for cord_uid
    await papers.create_index("cord_uid", unique=True)
    print("MongoDB initialized with indexes.")
