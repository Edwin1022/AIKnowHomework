import os
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from backend.db.models import Base

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")
DATABASE_URL = f"sqlite+aiosqlite:///{_DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=True)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
