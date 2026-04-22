import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import json
from datetime import datetime

async def export_to_json():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client.covid_research
    papers = await db.papers.find({}).to_list(length=200)
    
    for p in papers:
        p['_id'] = str(p['_id'])
        if isinstance(p.get('publish_time'), datetime):
            p['publish_time'] = p['publish_time'].isoformat()
        if isinstance(p.get('ingested_at'), datetime):
            p['ingested_at'] = p['ingested_at'].isoformat()
            
    with open('papers.json', 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=4)
    print(f"Successfully exported {len(papers)} papers to papers.json")

if __name__ == "__main__":
    asyncio.run(export_to_json())
