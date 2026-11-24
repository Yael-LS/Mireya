"""Lo que me acompaña: las canciones y las series de cada etapa de Mireya.

Canciones (con Spotify), sus elegidas del día, la semana y el mes, y las series
que ve (con TMDB: póster, temporadas y capítulos). El biógrafo conoce todo esto
por `chat_context()`; lo que ella escribe solo cuando dejó «que el biógrafo lo
sepa» encendido.
"""

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Literal

import requests
from fastapi import APIRouter, Header, HTTPException, Query, Request
from google.genai import types
from pydantic import BaseModel, Field

from app.database import get_supabase
from app.refugio import _generate
from app.security import limiter, verify_api_key
from app.spotify_client import SEARCH_URL, _get_access_token

logger = logging.getLogger("acompana")

router = APIRouter(prefix="/api/acompana")

SONGS = "companion_songs"
PICKS = "companion_picks"
SERIES = "companion_series"
EPISODES = "companion_episodes"
MONTHS = "companion_months"

TMDB_URL = "https://api.themoviedb.org/3"
TMDB_IMAGES = "https://image.tmdb.org/t/p/"
CDMX = timezone(timedelta(hours=-6))

MOODS = ("heartbreak", "strength", "nostalgia", "dance")
MOOD_LABELS = {"heartbreak": "me rompe el corazón", "strength": "me da fuerzas", "nostalgia": "me pone nostálgica", "dance": "me hace bailar"}
STATUSES = ("pending", "watching", "finished", "paused", "dropped")
STATUS_LABELS = {"pending": "pendiente", "watching": "viendo", "finished": "terminada", "paused": "en pausa", "dropped": "abandonada"}
KIND_LABELS = {"anime": "anime", "kdrama": "k-drama", "cdrama": "c-drama", "jdrama": "j-drama", "asian": "drama asiático", "series": "serie"}
Mood = Literal["heartbreak", "strength", "nostalgia", "dance"]
Status = Literal["pending", "watching", "finished", "paused", "dropped"]
PickKind = Literal["day", "week", "month"]


def today() -> date:
    return datetime.now(CDMX).date()


def period_start(kind: str, day: date) -> date:
    if kind == "week":
        return day - timedelta(days=day.weekday())
    if kind == "month":
        return day.replace(day=1)
    return day


# ---------- Lo que sabe el biógrafo ----------

_context_cache: tuple[float, str] = (0.0, "")


def _forget_context() -> None:
    global _context_cache
    _context_cache = (0.0, "")


def _mx_day(iso: str) -> str:
    """La fecha en el centro de México (los timestamps vienen en UTC)."""
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(CDMX).date().isoformat()


def _clip(text: str | None, size: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= size else text[: size - 1] + "…"


def chat_context() -> str:
    """Resumen compacto para el prompt del chat. Nunca rompe el chat: si falla, vacío."""
    global _context_cache
    if time.time() - _context_cache[0] < 60:
        return _context_cache[1]
    try:
        db = get_supabase()
        songs = db.table(SONGS).select("*").order("created_at", desc=True).limit(40).execute().data or []
        picks = db.table(PICKS).select("*").order("period", desc=True).limit(40).execute().data or []
        series = db.table(SERIES).select("*").order("updated_at", desc=True).limit(30).execute().data or []
        episodes = db.table(EPISODES).select("*").order("watched_at", desc=True).limit(60).execute().data or []
    except Exception:
        logger.exception("No se pudo armar el contexto de Lo que me acompaña")
        return ""
    by_id = {song["id"]: song for song in songs}
    lines: list[str] = []
    names = {"day": "canción del día", "week": "canción de la semana", "month": "canción del mes"}
    for pick in picks[:24]:
        song = by_id.get(pick["song_id"])
        if song:
            lines.append(f"- {names[pick['kind']]} ({pick['period']}): «{song['title']}» de {song['artist']}")
    if lines:
        lines.insert(0, "Sus canciones elegidas:")
    song_lines = []
    for song in songs[:20]:
        mood = song.get("mood_label") or MOOD_LABELS.get(song.get("mood") or "", "")
        line = f"- «{song['title']}» de {song['artist']} (agregada el {_mx_day(song['created_at'])}{', ' + mood if mood else ''})"
        if song.get("share") and song.get("note"):
            line += f": ella escribió «{_clip(song['note'], 280)}»"
        song_lines.append(line)
    if song_lines:
        lines += ["Canciones que la acompañan:", *song_lines]
    series_lines = []
    for show in series:
        watched = [e for e in episodes if e["series_id"] == show["id"]]
        line = f"- {show['title']} ({KIND_LABELS.get(show.get('kind') or '', 'serie')}, {STATUS_LABELS.get(show['status'], show['status'])}"
        if watched:
            last = watched[0]
            line += f", va en T{last['season']}·E{last['episode']} visto el {_mx_day(last['watched_at'])}"
        if show.get("rating"):
            line += f", le dio {show['rating']} de 5 corazones"
        line += ")"
        if show.get("share"):
            if show.get("characters"):
                line += f"; personajes favoritos: {', '.join(show['characters'][:6])}"
            if show.get("note"):
                line += f"; sobre la serie escribió «{_clip(show['note'], 240)}»"
            if show.get("final_opinion"):
                line += f"; opinión final: «{_clip(show['final_opinion'], 240)}»"
            notes = [f"E{e['episode']}: «{_clip(e['note'], 140)}»" for e in watched if e.get("note")][:2]
            if notes:
                line += f"; notas de capítulos: {' '.join(notes)}"
        series_lines.append(line)
    if series_lines:
        lines += ["Series, k-dramas y animes que registró:", *series_lines]
    text = "\n".join(lines)[:5000]
    _context_cache = (time.time(), text)
    return text


# ---------- Todo de una vez ----------

def _series_out(show: dict, episodes: list[dict]) -> dict:
    mine = sorted((e for e in episodes if e["series_id"] == show["id"]), key=lambda e: (e["season"], e["episode"]))
    return {**show, "episodes": mine}


@router.get("")
def load(authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    db = get_supabase()
    episodes = db.table(EPISODES).select("*").execute().data or []
    return {
        "today": today().isoformat(),
        "songs": db.table(SONGS).select("*").order("created_at", desc=True).execute().data or [],
        "picks": db.table(PICKS).select("*").order("period", desc=True).execute().data or [],
        "series": [_series_out(s, episodes) for s in db.table(SERIES).select("*").order("updated_at", desc=True).execute().data or []],
        "months": db.table(MONTHS).select("*").execute().data or [],
    }


# ---------- Canciones ----------

@router.get("/spotify")
@limiter.limit("60/minute")
def search_spotify(request: Request, q: str = Query(min_length=1, max_length=120), authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    try:
        response = requests.get(
            SEARCH_URL,
            params={"q": q, "type": "track", "limit": 8, "market": "MX"},
            headers={"Authorization": f"Bearer {_get_access_token()}"},
            timeout=10,
        )
        response.raise_for_status()
    except Exception:
        raise HTTPException(status_code=502, detail="Spotify no respondió. Intenta otra vez en un momento.")
    songs = []
    for track in response.json().get("tracks", {}).get("items", []):
        album = track.get("album", {})
        images = album.get("images") or []
        songs.append({
            "spotify_id": track["id"],
            "title": track["name"],
            "artist": ", ".join(artist["name"] for artist in track["artists"]),
            "album": album.get("name"),
            "cover_url": images[0]["url"] if images else None,
            "spotify_url": (track.get("external_urls") or {}).get("spotify"),
        })
    return songs


class SongIn(BaseModel):
    spotify_id: str | None = Field(default=None, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    artist: str = Field(min_length=1, max_length=200)
    album: str | None = Field(default=None, max_length=200)
    cover_url: str | None = Field(default=None, max_length=500)
    spotify_url: str | None = Field(default=None, max_length=500)


class SongPatch(BaseModel):
    lyric: str | None = Field(default=None, max_length=4000)
    note: str | None = Field(default=None, max_length=20000)
    mood: Mood | None = None
    mood_label: str | None = Field(default=None, max_length=60)
    share: bool | None = None


@router.post("/songs", status_code=201)
def add_song(body: SongIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    _forget_context()
    return get_supabase().table(SONGS).insert(body.model_dump()).execute().data[0]


@router.patch("/songs/{song_id}")
def update_song(song_id: str, body: SongPatch, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    changes = body.model_dump(exclude_unset=True)
    if "mood_label" in changes:
        changes["mood_label"] = (changes["mood_label"] or "").strip() or None
    changes["updated_at"] = datetime.now(timezone.utc).isoformat()
    result = get_supabase().table(SONGS).update(changes).eq("id", song_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Esa canción ya no está.")
    _forget_context()
    return result.data[0]


@router.delete("/songs/{song_id}", status_code=204)
def delete_song(song_id: str, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    get_supabase().table(SONGS).delete().eq("id", song_id).execute()
    _forget_context()


class PickIn(BaseModel):
    kind: PickKind
    song_id: str
    # Cualquier día dentro del periodo; sin él, el periodo actual. Sirve para llenar
    # semanas o meses que se le olvidaron.
    period: date | None = None


@router.post("/picks")
def toggle_pick(body: PickIn, authorization: str | None = Header(default=None)):
    """Elige la canción del día, de la semana o del mes; si ya era esa, la quita."""
    verify_api_key(authorization)
    current_period = period_start(body.kind, today())
    chosen = period_start(body.kind, body.period) if body.period else current_period
    if chosen > current_period:
        raise HTTPException(status_code=422, detail="Ese periodo todavía no llega.")
    period = chosen.isoformat()
    db = get_supabase()
    current = db.table(PICKS).select("*").eq("kind", body.kind).eq("period", period).execute().data
    _forget_context()
    if current and current[0]["song_id"] == body.song_id:
        db.table(PICKS).delete().eq("kind", body.kind).eq("period", period).execute()
        return {"kind": body.kind, "period": period, "song_id": None}
    row = {"kind": body.kind, "period": period, "song_id": body.song_id}
    db.table(PICKS).upsert(row).execute()
    return row


class MoodIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    title: str | None = Field(default=None, max_length=200)
    artist: str | None = Field(default=None, max_length=200)


class MoodGuess(BaseModel):
    mood: Literal["heartbreak", "strength", "nostalgia", "dance"]
    label: str = Field(max_length=60)


MOOD_INSTRUCTION = """
Lees lo que Mireya escribió sobre una canción y eliges el sentimiento que mejor
la describe: heartbreak (le rompe el corazón, duele, desamor, tristeza),
strength (le da fuerzas, ánimo, valentía, soltar), nostalgia (le recuerda algo,
añoranza, calma de otros tiempos) o dance (alegría, la hace bailar, euforia).
`label` es una frase corta en primera persona y en minúsculas, como la diría
ella (máximo 6 palabras), por ejemplo «me rompe el corazón… pero bonito» o
«me da fuerzas para soltar». Respeta su intención; no inventes nada.
"""


@router.post("/mood")
@limiter.limit("20/minute")
def guess_mood(request: Request, body: MoodIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    song = f"Canción: «{body.title}» de {body.artist}\n" if body.title else ""
    try:
        response = _generate(
            f"{song}Lo que ella escribió:\n{body.text[:6000]}",
            types.GenerateContentConfig(system_instruction=MOOD_INSTRUCTION, response_mime_type="application/json", response_schema=MoodGuess, temperature=0.2),
        )
        parsed = response.parsed if isinstance(response.parsed, MoodGuess) else MoodGuess.model_validate_json(response.text or "{}")
    except Exception:
        logger.exception("No se pudo adivinar el sentimiento")
        raise HTTPException(status_code=503, detail="No pude leer el sentimiento ahora; elígelo tú.")
    return {"mood": parsed.mood, "label": parsed.label.strip()}


# ---------- Series (TMDB) ----------

def _tmdb(path: str, **params) -> dict:
    token = os.getenv("TMDB_READ_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="Falta configurar TMDB en el servidor.")
    try:
        response = requests.get(f"{TMDB_URL}{path}", params=params, headers={"Authorization": f"Bearer {token}", "accept": "application/json"}, timeout=12)
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="TMDB no respondió. Intenta otra vez en un momento.")
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="No encontré esa serie en TMDB.")
    if not response.ok:
        raise HTTPException(status_code=502, detail="TMDB no respondió. Intenta otra vez en un momento.")
    return response.json()


def _image(path: str | None, size: str = "w500") -> str | None:
    return f"{TMDB_IMAGES}{size}{path}" if path else None


def _kind(countries: list[str], genres: list[int]) -> str:
    countries = countries or []
    if 16 in (genres or []) and "JP" in countries:
        return "anime"
    if "KR" in countries:
        return "kdrama"
    if any(c in countries for c in ("CN", "TW", "HK")):
        return "cdrama"
    if "JP" in countries:
        return "jdrama"
    if any(c in countries for c in ("TH", "PH", "VN", "ID")):
        return "asian"
    return "series"


@router.get("/tmdb/search")
@limiter.limit("60/minute")
def search_series(request: Request, q: str = Query(min_length=1, max_length=120), authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    data = _tmdb("/search/tv", query=q, language="es-MX", include_adult="false")
    results = []
    for show in data.get("results", [])[:10]:
        results.append({
            "tmdb_id": show["id"],
            "title": show.get("name") or show.get("original_name"),
            "original_title": show.get("original_name"),
            "year": (show.get("first_air_date") or "")[:4] or None,
            "poster_url": _image(show.get("poster_path"), "w342"),
            "kind": _kind(show.get("origin_country") or [], show.get("genre_ids") or []),
            "overview": _clip(show.get("overview"), 220),
        })
    return results


_season_cache: dict[tuple[int, int], tuple[float, list[dict]]] = {}


def _generic(name: str | None, number: int) -> bool:
    return not name or name.strip().lower() in (f"episodio {number}", f"episode {number}", f"capítulo {number}")


@router.get("/tmdb/{tmdb_id}/season/{season}")
def season_episodes(tmdb_id: int, season: int, authorization: str | None = Header(default=None)):
    """Capítulos de una temporada: título en español y, si TMDB no lo tiene, en inglés."""
    verify_api_key(authorization)
    cached = _season_cache.get((tmdb_id, season))
    if cached and time.time() - cached[0] < 6 * 3600:
        return cached[1]
    spanish = _tmdb(f"/tv/{tmdb_id}/season/{season}", language="es-MX").get("episodes", [])
    english = {}
    if any(_generic(e.get("name"), e["episode_number"]) for e in spanish):
        try:
            english = {e["episode_number"]: e.get("name") for e in _tmdb(f"/tv/{tmdb_id}/season/{season}", language="en-US").get("episodes", [])}
        except HTTPException:
            english = {}
    episodes = []
    for e in spanish:
        number = e["episode_number"]
        name = e.get("name")
        if _generic(name, number) and not _generic(english.get(number), number):
            name = english[number]
        episodes.append({
            "episode": number,
            "title": None if _generic(name, number) else name,
            "air_date": e.get("air_date"),
            "still_url": _image(e.get("still_path"), "w300"),
            "runtime": e.get("runtime"),
        })
    _season_cache[(tmdb_id, season)] = (time.time(), episodes)
    return episodes


class SeriesIn(BaseModel):
    tmdb_id: int
    status: Status = "watching"


@router.post("/series", status_code=201)
def add_series(body: SeriesIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    db = get_supabase()
    if db.table(SERIES).select("id").eq("tmdb_id", body.tmdb_id).execute().data:
        raise HTTPException(status_code=409, detail="Esa serie ya está en tu lista.")
    show = _tmdb(f"/tv/{body.tmdb_id}", language="es-MX")
    english = show.get("name")
    try:
        english = _tmdb(f"/tv/{body.tmdb_id}", language="en-US").get("name") or english
    except HTTPException:
        pass
    seasons = [
        {"n": s["season_number"], "count": s.get("episode_count") or 0, "name": s.get("name")}
        for s in show.get("seasons") or []
        if s.get("season_number", 0) > 0 and (s.get("episode_count") or 0) > 0
    ]
    row = {
        "tmdb_id": body.tmdb_id,
        "title": show.get("name") or show.get("original_name"),
        "alt_title": english if english and english != show.get("name") else None,
        "original_title": show.get("original_name"),
        "poster_url": _image(show.get("poster_path")),
        "backdrop_url": _image(show.get("backdrop_path"), "w780"),
        "kind": _kind(show.get("origin_country") or [], [g["id"] for g in show.get("genres") or []]),
        "year": (show.get("first_air_date") or "")[:4] or None,
        "seasons": seasons,
        "total_episodes": sum(s["count"] for s in seasons),
        "status": body.status,
        "started_at": today().isoformat() if body.status == "watching" else None,
    }
    created = db.table(SERIES).insert(row).execute().data[0]
    _forget_context()
    return {**created, "episodes": []}


class SeriesPatch(BaseModel):
    status: Status | None = None
    note: str | None = Field(default=None, max_length=20000)
    characters: list[str] | None = Field(default=None, max_length=30)
    final_opinion: str | None = Field(default=None, max_length=20000)
    rating: int | None = Field(default=None, ge=0, le=5)
    share: bool | None = None


@router.patch("/series/{series_id}")
def update_series(series_id: str, body: SeriesPatch, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    changes = body.model_dump(exclude_unset=True)
    if "characters" in changes:
        changes["characters"] = [c.strip()[:60] for c in changes["characters"] or [] if c.strip()]
    if changes.get("rating") == 0:
        changes["rating"] = None
    db = get_supabase()
    current = db.table(SERIES).select("status,started_at,finished_at").eq("id", series_id).execute().data
    if not current:
        raise HTTPException(status_code=404, detail="Esa serie ya no está.")
    status = changes.get("status")
    if status == "finished" and not current[0].get("finished_at"):
        changes["finished_at"] = today().isoformat()
    if status and status != "finished":
        changes["finished_at"] = None
    if status == "watching" and not current[0].get("started_at"):
        changes["started_at"] = today().isoformat()
    changes["updated_at"] = datetime.now(timezone.utc).isoformat()
    updated = db.table(SERIES).update(changes).eq("id", series_id).execute().data[0]
    episodes = db.table(EPISODES).select("*").eq("series_id", series_id).execute().data or []
    _forget_context()
    return _series_out(updated, episodes)


@router.delete("/series/{series_id}", status_code=204)
def delete_series(series_id: str, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    get_supabase().table(SERIES).delete().eq("id", series_id).execute()
    _forget_context()


class EpisodeIn(BaseModel):
    season: int = Field(ge=0, le=500)
    episode: int = Field(ge=1, le=5000)
    title: str | None = Field(default=None, max_length=300)
    note: str | None = Field(default=None, max_length=10000)


@router.post("/series/{series_id}/episodes")
def watch_episode(series_id: str, body: EpisodeIn, authorization: str | None = Header(default=None)):
    """Marca un capítulo como visto (hoy). Si ya estaba, solo actualiza su nota."""
    verify_api_key(authorization)
    db = get_supabase()
    show = db.table(SERIES).select("id,status,started_at").eq("id", series_id).execute().data
    if not show:
        raise HTTPException(status_code=404, detail="Esa serie ya no está.")
    existing = db.table(EPISODES).select("id").eq("series_id", series_id).eq("season", body.season).eq("episode", body.episode).execute().data
    row = {"series_id": series_id, **body.model_dump(exclude_unset=True)}
    if existing:
        saved = db.table(EPISODES).update(row).eq("id", existing[0]["id"]).execute().data[0]
    else:
        saved = db.table(EPISODES).insert(row).execute().data[0]
    # Ver un capítulo la pone «viendo» si estaba pendiente o en pausa.
    touch = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if show[0]["status"] in ("pending", "paused"):
        touch["status"] = "watching"
    if not show[0].get("started_at"):
        touch["started_at"] = today().isoformat()
    db.table(SERIES).update(touch).eq("id", series_id).execute()
    _forget_context()
    return saved


@router.delete("/series/{series_id}/episodes/{episode_id}", status_code=204)
def unwatch_episode(series_id: str, episode_id: str, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    get_supabase().table(EPISODES).delete().eq("id", episode_id).eq("series_id", series_id).execute()
    _forget_context()


# ---------- Por etapas ----------

MONTH_INSTRUCTION = """
Eres el biógrafo de Mireya. Con la lista de canciones que eligió y las series
que vio en un mes, escribe dos frases cálidas y con personalidad sobre cómo fue
ese mes para ella, en tercera persona («Noviembre fue de soltar para Mireya…»).
Usa solo lo que dicen los datos; no inventes hechos ni nombres.
"""


class MonthIn(BaseModel):
    summary_of: str = Field(min_length=1, max_length=8000)


@router.post("/months/{month}")
@limiter.limit("10/minute")
def summarize_month(request: Request, month: date, body: MonthIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    first = month.replace(day=1)
    try:
        response = _generate(f"Mes: {first.isoformat()[:7]}\n{body.summary_of}", types.GenerateContentConfig(system_instruction=MONTH_INSTRUCTION, temperature=0.7))
        summary = (response.text or "").strip()
    except Exception:
        logger.exception("No se pudo resumir el mes")
        summary = ""
    if not summary:
        raise HTTPException(status_code=503, detail="El biógrafo no pudo escribir el resumen ahora. Intenta en un rato.")
    row = {"month": first.isoformat(), "summary": summary[:1200], "updated_at": datetime.now(timezone.utc).isoformat()}
    get_supabase().table(MONTHS).upsert(row).execute()
    return row
