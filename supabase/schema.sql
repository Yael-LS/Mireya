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

-- El backend usa SUPABASE_SERVICE_ROLE_KEY. Si se habilita acceso directo desde
-- el navegador, activa RLS y crea políticas ligadas a auth.uid() antes de ello.
