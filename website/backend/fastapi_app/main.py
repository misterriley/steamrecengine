import asyncio
import logging
import math
from typing import List

import asyncpg
import pandas as pd
import uvicorn
from databases import Database
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import Column, Integer, String, Table, MetaData, Float
from fastapi.middleware.cors import CORSMiddleware

from constantsCache import ConstantsCache
from gamesCache import GamesCache
from predictionEngine import PredictionEngine

# FastAPI instance
app = FastAPI()

origins = [
    "http://localhost:8080",  # or whatever port your frontend runs on
    "http://127.0.0.1:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = "postgresql://bleem:bleem112358@localhost:5432/game_data"
database = Database(DATABASE_URL)

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
    Column("computation_state", Integer, nullable=False),
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
games_cache: GamesCache = None
constants_cache: ConstantsCache = None


# Startup and shutdown hooks
@app.on_event("startup")
async def startup():
    await database.connect()

    global constants_cache
    all_constants = await get_constants()
    constants_cache = ConstantsCache(all_constants)

    global games_cache
    games_cache = await load_games_cache()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# Helper functions
async def load_games_cache(chunk_size: int = 50000, max_rows=None):
    """Fetch games data in chunks using asyncpg."""
    conn = await asyncpg.connect(DATABASE_URL)
    offset = 0
    all_data = []
    if max_rows is not None:
        chunk_size = min(chunk_size, max_rows)

    while True:
        query = f"SELECT * FROM games ORDER BY game_id LIMIT {chunk_size} OFFSET {offset}"
        rows = await conn.fetch(query)

        if not rows:
            break

        # Convert rows to dictionaries and append
        all_data.extend([dict(row) for row in rows])

        # Stop if we've reached the maximum rows for testing
        if max_rows is not None and len(all_data) >= max_rows:
            all_data = all_data[:max_rows]  # Truncate to the exact number
            break

        offset += chunk_size

    await conn.close()
    return GamesCache(pd.DataFrame(all_data))


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
    rows = await database.fetch_all(
        user_preferences.select()
        .where(user_preferences.c.user_id == user_id)
        .order_by(user_preferences.c.game_id)
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No preferences found for user_id {user_id}")
    sanitized_rows = [
        {**row, "rating": None if row["rating"] is None or math.isnan(row["rating"]) else row["rating"]}
        for row in rows
    ]

    ret = []
    for row in sanitized_rows:
        if row["rating"] is not None or row["status"] == constants_cache.STATUS_MORE or row["status"] == constants_cache.STATUS_LESS or row["status"] == constants_cache.STATUS_IGNORE:
            game_id = row["game_id"]
            row["name"] = games_cache.get_game_name(game_id)
            ret.append(row)

    return ret


@app.get("/get_game_names_by_ids/")
async def get_game_names_by_ids(game_ids: List[int] = Query(...)):
    if game_ids != sorted(game_ids):
        raise HTTPException(status_code=400, detail="The list of game_ids must be sorted")
    t = games_cache.game_data
    filtered_games = t.loc[t['game_id'].isin(game_ids), :]

    # Check if any game_ids are missing
    if len(filtered_games) != len(game_ids):
        found_game_ids = set(filtered_games['game_id'])  # Extract found game_ids
        missing_game_ids = set(game_ids) - found_game_ids  # Find missing game_ids
        raise HTTPException(status_code=404, detail=f"The following game_ids are not found: {missing_game_ids}")
    return {"game_names": filtered_games['name'].tolist()}


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
    c = [{"name": row["name"], "value": row["value"]} for row in rows]
    for d in c:
        if d["value"] == round(d["value"]):
            d["value"] = int(d["value"])
    return c


@app.get("/run_prediction_computation/{user_id}")
async def run_prediction_computation(user_id: int):
    user_prefs = await get_user_preferences(user_id)
    query = computation_states.insert().values(user_id=user_id, computation_state=constants_cache.COMP_STATUS_STARTED)
    computation_id = await database.execute(query)
    asyncio.create_task(gather_predictions(computation_id, user_id, user_prefs))
    return {"computation_id": computation_id}


async def gather_predictions(computation_id: int, user_id: int, user_prefs):
    try:
        await update_comp_status(computation_id, constants_cache.COMP_STATUS_FINDING_BEST_REGRESSIONS)
        regr_best_rd = PredictionEngine.tune_regression(games_cache, constants_cache, user_prefs)

        await update_comp_status(computation_id, constants_cache.COMP_STATUS_FINDING_BEST_CLASSIFIER)
        lgst_best_rd = PredictionEngine.tune_classifier(games_cache, constants_cache, user_prefs)

        await update_comp_status(computation_id, constants_cache.COMP_STATUS_SORTING_PREDICTIONS)
        predictions = PredictionEngine.get_predictions(regr_best_rd, lgst_best_rd, games_cache, constants_cache, None)

        await update_comp_status(computation_id, constants_cache.COMP_STATUS_FINISHED)
        return predictions
    except Exception as e:
        print(e)
        await update_comp_status(computation_id, constants_cache.COMP_STATUS_ERROR)
        return None


async def update_comp_status(computation_id, comp_status):
    await database.execute(
        computation_states.update().where(computation_states.c.computation_id == computation_id).values(
            computation_state=comp_status))

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
