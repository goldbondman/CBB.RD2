-- CBB RD2 initial Supabase schema
-- Safe to run in Supabase Postgres.

create extension if not exists pgcrypto;

-- enums/check domains kept as text + checks for easier evolution

create table if not exists public.teams (
  id uuid primary key default gen_random_uuid(),
  season int not null,
  source_team_id text not null,
  team_name text not null,
  conference text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (season, source_team_id),
  unique (season, team_name)
);

create table if not exists public.raw_games (
  id uuid primary key default gen_random_uuid(),
  season int not null,
  source text not null,
  external_game_id text not null,
  payload jsonb not null,
  pulled_at timestamptz not null default now(),
  verification_status text not null default 'partial' check (verification_status in ('verified','partial','conflict','rejected')),
  verification_notes text,
  unique (season, source, external_game_id)
);

create table if not exists public.games (
  id uuid primary key default gen_random_uuid(),
  season int not null,
  game_datetime_utc timestamptz not null,
  home_team_id uuid not null references public.teams(id),
  away_team_id uuid not null references public.teams(id),
  home_score int,
  away_score int,
  status text not null check (status in ('scheduled','live','final')),
  venue text,
  source text not null,
  external_game_id text not null,
  verification_status text not null default 'partial' check (verification_status in ('verified','partial','conflict','rejected')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (season, source, external_game_id),
  unique (season, game_datetime_utc, home_team_id, away_team_id)
);

create table if not exists public.market_lines (
  id uuid primary key default gen_random_uuid(),
  game_id uuid not null references public.games(id),
  book text not null,
  pulled_at timestamptz not null default now(),
  spread_home numeric,
  total numeric,
  ml_home int,
  ml_away int,
  unique (game_id, book, pulled_at)
);

create table if not exists public.team_game_features (
  id uuid primary key default gen_random_uuid(),
  game_id uuid not null references public.games(id),
  team_id uuid not null references public.teams(id),
  home_away text not null check (home_away in ('home','away')),
  feature_set text not null,
  features jsonb not null default '{}'::jsonb,
  pulled_at timestamptz not null default now(),
  verification_status text not null default 'partial' check (verification_status in ('verified','partial','conflict','rejected')),
  unique (game_id, team_id, feature_set)
);

create table if not exists public.predictions (
  id uuid primary key default gen_random_uuid(),
  model_version text not null,
  created_at timestamptz not null default now(),
  game_id uuid not null references public.games(id),
  market_snapshot_id uuid,
  pred_spread numeric,
  pred_total numeric,
  win_prob_home numeric,
  edges jsonb,
  confidence numeric,
  notes text,
  unique (model_version, game_id)
);

create table if not exists public.bets (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id),
  created_at timestamptz not null default now(),
  game_id uuid not null references public.games(id),
  bet_type text not null,
  side text not null,
  line numeric,
  odds int,
  stake numeric not null check (stake > 0),
  result text,
  unique (user_id, game_id, bet_type, created_at)
);

create table if not exists public.dq_audit (
  id uuid primary key default gen_random_uuid(),
  entity_type text not null,
  entity_id uuid,
  severity text not null,
  reason_codes text[] not null default '{}',
  details jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

-- Required indexes
create index if not exists idx_games_season_datetime on public.games(season, game_datetime_utc);
create index if not exists idx_games_home_team_id on public.games(home_team_id);
create index if not exists idx_games_away_team_id on public.games(away_team_id);
create index if not exists idx_market_lines_game_id on public.market_lines(game_id);
create index if not exists idx_market_lines_pulled_at on public.market_lines(pulled_at);
create index if not exists idx_team_game_features_game_id on public.team_game_features(game_id);
create index if not exists idx_team_game_features_team_id on public.team_game_features(team_id);
create index if not exists idx_predictions_model_version on public.predictions(model_version);
create index if not exists idx_predictions_game_id on public.predictions(game_id);
create index if not exists idx_bets_user_id on public.bets(user_id);
create index if not exists idx_bets_game_id on public.bets(game_id);

-- RLS
alter table public.teams enable row level security;
alter table public.games enable row level security;
alter table public.predictions enable row level security;
alter table public.team_game_features enable row level security;
alter table public.market_lines enable row level security;
alter table public.bets enable row level security;

-- model/public read policies
create policy "teams_read_auth" on public.teams
for select to authenticated using (true);

create policy "games_read_auth" on public.games
for select to authenticated using (true);

create policy "predictions_read_auth" on public.predictions
for select to authenticated using (true);

create policy "team_game_features_read_auth" on public.team_game_features
for select to authenticated using (true);

create policy "market_lines_read_auth" on public.market_lines
for select to authenticated using (true);

-- bets owner policies
create policy "bets_select_own" on public.bets
for select to authenticated
using (user_id = auth.uid());

create policy "bets_insert_own" on public.bets
for insert to authenticated
with check (user_id = auth.uid());

create policy "bets_update_own" on public.bets
for update to authenticated
using (user_id = auth.uid())
with check (user_id = auth.uid());

create policy "bets_delete_own" on public.bets
for delete to authenticated
using (user_id = auth.uid());

-- deterministic upsert examples (for app/server usage)
-- games: on conflict (season, source, external_game_id) do update set ...
-- predictions: on conflict (model_version, game_id) do nothing
