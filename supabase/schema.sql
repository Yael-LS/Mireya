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

-- El diario de Mireya: vive aparte del biógrafo, nada de esto llega a la IA.
create table if not exists public.diary_folders (
  id uuid primary key default gen_random_uuid(),
  name text not null check (char_length(name) between 1 and 40),
  color text not null,
  used_at timestamptz not null default now(),
  created_at timestamptz not null default now()
);

create table if not exists public.diary_entries (
  day date primary key,
  body text not null default '',
  feel text not null default '',
  moods text[] not null default '{}',
  items jsonb not null default '[]',
  folder_id uuid references public.diary_folders(id) on delete set null,
  ink text not null default '#27405f',
  font text not null default 'caveat',
  updated_at timestamptz not null default now()
);

create table if not exists public.diary_letters (
  id uuid primary key default gen_random_uuid(),
  open_on date not null,
  sealed boolean not null,
  opened boolean not null default false,
  body text not null,
  ink text,
  font text,
  created_at timestamptz not null default now()
);

-- Una sola fila con el calendario y las preferencias.
create table if not exists public.diary_settings (
  id int primary key default 1 check (id = 1),
  era_by int[] not null default '{0,1,2,3,4,5,6,7,8,9,10,11}',
  cal_see int not null default 30,
  bg_all text,
  bg_month jsonb not null default '{}',
  my_bg text,
  ink text not null default '#27405f',
  font text not null default 'caveat'
);

-- Carpetitas de ejemplo: clasifican días, para que se entienda para qué sirven.
insert into public.diary_folders (name, color)
select name, color from (values
  ('Días felices', '#cfe4f6'), ('Días grises', '#b5d4ef'), ('Para no olvidar', '#ffffff'), ('Para volver a leer', '#d6dcf6')
) as ejemplos(name, color)
where not exists (select 1 from public.diary_folders);

-- El backend usa SUPABASE_SERVICE_ROLE_KEY. Si se habilita acceso directo desde
-- el navegador, activa RLS y crea políticas ligadas a auth.uid() antes de ello.
