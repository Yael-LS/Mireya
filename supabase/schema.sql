-- Ejecutar una vez en el SQL Editor de Supabase.
create extension if not exists "pgcrypto";

create table if not exists public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  title text not null default 'Nueva conversación',
  user_id text not null default 'default',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.chat_sessions(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null check (char_length(content) > 0),
  created_at timestamptz not null default now()
);

create index if not exists chat_sessions_user_updated_at_idx
  on public.chat_sessions (user_id, updated_at desc);
create index if not exists messages_session_created_at_idx
  on public.messages (session_id, created_at asc);

create or replace function public.touch_chat_session()
returns trigger language plpgsql as $$
begin
  update public.chat_sessions set updated_at = now() where id = new.session_id;
  return new;
end;
$$;

drop trigger if exists messages_touch_session on public.messages;
create trigger messages_touch_session
after insert on public.messages
for each row execute function public.touch_chat_session();

-- Modo de la conversación: "nostalgia" cuando nace de un momento del Refugio.
-- ("chat_mode" y no "mode": PostgREST confunde "mode" con la función de agregado).
alter table public.chat_sessions add column if not exists chat_mode text not null default 'normal';

-- El Refugio: todo lo que le hace bien a Mireya (canciones, series, momentos…).
create table if not exists public.refugio_items (
  id uuid primary key default gen_random_uuid(),
  kind text not null check (kind in ('song', 'series', 'kdrama', 'anime', 'movie', 'book', 'quote', 'moment', 'other')),
  title text not null check (char_length(title) > 0),
  creator text,
  note text,
  url text,
  photo_path text,
  moment_date date,
  source text not null default 'manual' check (source in ('profile', 'memory', 'manual')),
  source_ref text,
  confidence real,
  hidden boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists refugio_items_kind_idx on public.refugio_items (kind, created_at desc);

-- Clave del Refugio (una sola fila; solo el hash PBKDF2, nunca la clave).
create table if not exists public.refugio_lock (
  id int primary key default 1 check (id = 1),
  pin_hash text not null,
  hint text,
  updated_at timestamptz not null default now()
);

-- Reacción con corazón a un mensaje.
alter table public.messages add column if not exists hearted boolean not null default false;

-- Caja de ideas: lo que a Mireya le gustaría que su scrapbook tuviera después.
create table if not exists public.scrapbook_ideas (
  id uuid primary key default gen_random_uuid(),
  text text not null check (char_length(text) between 1 and 300),
  done boolean not null default false,
  done_at timestamptz,
  created_at timestamptz not null default now()
);

-- El backend usa SUPABASE_SERVICE_ROLE_KEY. Si se habilita acceso directo desde
-- el navegador, activa RLS y crea políticas ligadas a auth.uid() antes de ello.
