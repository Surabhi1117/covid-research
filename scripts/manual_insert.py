import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

async def insert_papers():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client.covid_research
    
    papers = [
        {
            'cord_uid': 'PLOS_0257096',
            'title': 'Knowledge, beliefs, attitudes and perceived risk about COVID-19 vaccine and determinants of COVID-19 vaccine acceptance in Bangladesh',
            'authors': 'Ali, M., Hossain, M. J., et al.',
            'abstract': "The objectives of this study were to evaluate the acceptance of the COVID-19 vaccines and examine the factors associated with the acceptance in Bangladesh. A web-based anonymous cross-sectional survey was conducted among 605 respondents. Results showed that 61.16% of respondents were willing to accept the COVID-19 vaccine. Factors such as age, gender, location, and perceived risk were significantly associated with vaccine acceptance.",
            'doi': '10.1371/journal.pone.0257096',
            'url': 'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0257096',
            'publish_time': datetime(2021, 9, 8),
            'ingested_at': datetime.utcnow()
        },
        {
            'cord_uid': 'PLOS_0243838',
            'title': 'A multi-country survey of public knowledge, attitudes, and beliefs about COVID-19',
            'authors': 'Bates, B. R., et al.',
            'abstract': "Understanding public knowledge and attitudes is crucial for effective public health messaging. This study surveys knowledge and beliefs about COVID-19 across multiple countries, identifying key gaps and psychological determinants of protective behaviors.",
            'doi': '10.1371/journal.pone.0243838',
            'url': 'https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243838',
            'publish_time': datetime(2020, 12, 17),
            'ingested_at': datetime.utcnow()
        }
    ]
    
    for p in papers:
        await db.papers.update_one({'cord_uid': p['cord_uid']}, {'$set': p}, upsert=True)
    print(f"Successfully inserted {len(papers)} high-quality papers.")

if __name__ == "__main__":
    asyncio.run(insert_papers())
