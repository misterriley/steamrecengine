from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from databases import Database
from sqlalchemy import Column, Integer, String, Table, MetaData, Float, create_engine, inspect
import uvicorn
from typing import List
import math
import time
import pandas as pd
import asyncpg

from gamesCache import GamesCache

# Database configuration
DATABASE_URL = "postgresql://bleem:bleem112358@localhost:5432/game_data"
database = Database(DATABASE_URL)

# FastAPI instance
app = FastAPI()

# Database metadata and table definitions
metadata = MetaData()

# Table definitions
users = Table(
    "users",
    metadata,
    Column("user_id", Integer, primary_key=True),
    Column("name", String(255)),
)

user_preferences = Table(
    "user_preferences",
    metadata,
    Column("preference_id", Integer, primary_key=True),
    Column("user_id", Integer),
    Column("game_id", Integer),
    Column("rating", Float),
    Column("status", Integer),
)

constants = Table(
    "constants",
    metadata,
    Column("constant_id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("value", String(255), nullable=False),
)

computation_states = Table(
    "computation_states",
    metadata,
    Column("computation_id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("status", String(255), nullable=False),
    Column("message", String(255), nullable=True),
)

computation_results = Table(
    "computation_results",
    metadata,
    Column("computation_id", Integer, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("game_id", Integer, nullable=False),
    Column("predicted_rating", Float, nullable=False),
    Column("predicted_probability", Float, nullable=False),
)

# Caching games data
games_cache = None


# Startup and shutdown hooks
@app.on_event("startup")
async def startup():
    await database.connect()
    global games_cache
    all_games = await load_all_games()
    games_cache = GamesCache(all_games)


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# Helper functions
async def load_all_games(chunk_size: int = 50000):
    """Fetch games data in chunks using asyncpg."""
    conn = await asyncpg.connect(DATABASE_URL)
    offset = 0
    all_data = []

    while True:
        query = f"SELECT * FROM games ORDER BY game_id LIMIT {chunk_size} OFFSET {offset}"
        rows = await conn.fetch(query)

        if not rows:
            break

        # Convert rows to dictionaries and append
        all_data.extend([dict(row) for row in rows])
        offset += chunk_size

    await conn.close()
    return pd.DataFrame(all_data)


# Routes
@app.get("/")
def read_root():
    return {"message": "Backend API is running"}


@app.get("/users/")
async def get_all_users():
    rows = await database.fetch_all(users.select())
    return rows


@app.get("/user_preferences/{user_id}")
async def get_user_preferences(user_id: int):
    rows = await database.fetch_all(user_preferences.select().where(user_preferences.c.user_id == user_id))
    if not rows:
        raise HTTPException(status_code=404, detail=f"No preferences found for user_id {user_id}")
    sanitized_rows = [
        {**row, "rating": None if row["rating"] is None or math.isnan(row["rating"]) else row["rating"]}
        for row in rows
    ]
    return sanitized_rows


@app.get("/get_game_names_by_ids/")
async def get_game_names_by_ids(game_ids: List[int] = Query(...)):
    if game_ids != sorted(game_ids):
        raise HTTPException(status_code=400, detail="The list of game_ids must be sorted")
    rows = await database.fetch_all(games.select().where(games.c.game_id.in_(game_ids)))
    found_game_ids = {row["game_id"] for row in rows}
    missing_game_ids = [game_id for game_id in game_ids if game_id not in found_game_ids]
    if missing_game_ids:
        raise HTTPException(status_code=404, detail=f"The following game_ids are not found: {missing_game_ids}")
    return {"game_names": [row["name"] for row in rows]}


@app.post("/add_preference/")
async def add_preference(user_id: int, game_id: int, status: int):
    query = user_preferences.insert().values(user_id=user_id, game_id=game_id, rating=None, status=status)
    last_record_id = await database.execute(query)
    return {"preference_id": last_record_id}


@app.delete("/delete_preference/")
async def delete_preference(preference_id: int):
    preference = await database.fetch_one(
        user_preferences.select().where(user_preferences.c.preference_id == preference_id))
    if not preference:
        raise HTTPException(status_code=404, detail=f"Preference with ID {preference_id} not found")
    await database.execute(user_preferences.delete().where(user_preferences.c.preference_id == preference_id))
    return {"message": f"Preference with ID {preference_id} has been deleted"}


@app.put("/set_rating_by_preference_id/")
async def set_rating_by_preference_id(preference_id: int, rating: float = None):
    preference = await database.fetch_one(
        user_preferences.select().where(user_preferences.c.preference_id == preference_id))
    if not preference:
        raise HTTPException(status_code=404, detail=f"Preference with ID {preference_id} not found")
    await database.execute(
        user_preferences.update().where(user_preferences.c.preference_id == preference_id).values(rating=rating))
    return {"message": f"Rating for preference ID {preference_id} has been updated to {rating}"}


@app.put("/set_status_by_preference_id/")
async def set_status_by_preference_id(preference_id: int, status: int):
    if status is None:
        raise HTTPException(status_code=400, detail="Status cannot be null")
    preference = await database.fetch_one(
        user_preferences.select().where(user_preferences.c.preference_id == preference_id))
    if not preference:
        raise HTTPException(status_code=404, detail=f"Preference with ID {preference_id} not found")
    await database.execute(
        user_preferences.update().where(user_preferences.c.preference_id == preference_id).values(status=status))
    return {"message": f"Status for preference ID {preference_id} has been updated to {status}"}


@app.get("/get_constants/")
async def get_constants():
    rows = await database.fetch_all(constants.select())
    if not rows:
        raise HTTPException(status_code=404, detail="No constants found")
    return rows


@app.post("/run_prediction_computation/")
async def run_prediction_computation(user_id: int, background_tasks: BackgroundTasks):
    query = computation_states.insert().values(user_id=user_id, status="started", message="Computation has started")
    computation_id = await database.execute(query)
    background_tasks.add_task(perform_computation, computation_id, user_id)
    return {"computation_id": computation_id}


async def perform_computation(computation_id: int, user_id: int):
    try:
        await database.execute(
            computation_states.update().where(computation_states.c.computation_id == computation_id).values(
                status="in progress", message=f"Computation for user {user_id} is running"))
        time.sleep(5)
        await database.execute(
            computation_states.update().where(computation_states.c.computation_id == computation_id).values(
                status="completed", message=f"Computation for user {user_id} completed successfully"))
    except Exception as e:
        await database.execute(
            computation_states.update().where(computation_states.c.computation_id == computation_id).values(
                status="failed", message=f"Computation for user {user_id} failed: {str(e)}"))


@app.get("/retrieve_computation_results/")
async def retrieve_computation_results(computation_id: int):
    rows = await database.fetch_all(
        computation_results.select().where(computation_results.c.computation_id == computation_id))
    if not rows:
        raise HTTPException(status_code=404, detail=f"No results found for computation_id {computation_id}")
    return [{"ordinal": row["ordinal"], "game_id": row["game_id"], "predicted_rating": row["predicted_rating"],
             "predicted_probability": row["predicted_probability"]} for row in rows]


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
