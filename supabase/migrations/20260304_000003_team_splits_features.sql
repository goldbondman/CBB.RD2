-- Team home/away split features (model-ready, append/upsert by team/date)

create table if not exists public.team_splits_features (
  id uuid primary key default gen_random_uuid(),
  team_id uuid not null references public.teams(id),
  season int not null,
  feature_date date not null,
  source text not null default 'team_splits.csv',
  pulled_at timestamptz not null default now(),
  verification_status text not null default 'partial' check (verification_status in ('verified','partial','conflict','rejected')),
  verification_notes text,
  efg_home numeric,
  efg_away numeric,
  orb_home numeric,
  orb_away numeric,
  tov_home numeric,
  tov_away numeric,
  pace_home numeric,
  pace_away numeric,
  netrtg_home numeric,
  netrtg_away numeric,
  ftr_home numeric,
  ftr_away numeric,
  unique (team_id, season, feature_date)
);

create index if not exists idx_team_splits_features_team_id on public.team_splits_features(team_id);
create index if not exists idx_team_splits_features_season_date on public.team_splits_features(season, feature_date);
create index if not exists idx_team_splits_features_pulled_at on public.team_splits_features(pulled_at);

alter table public.team_splits_features enable row level security;

create policy "team_splits_features_read_auth" on public.team_splits_features
for select to authenticated using (true);

-- deterministic upsert shape (server-side):
-- insert into public.team_splits_features (...) values (...)
-- on conflict (team_id, season, feature_date) do update set
--   pulled_at = excluded.pulled_at,
--   verification_status = excluded.verification_status,
--   verification_notes = excluded.verification_notes,
--   efg_home = excluded.efg_home,
--   efg_away = excluded.efg_away,
--   orb_home = excluded.orb_home,
--   orb_away = excluded.orb_away,
--   tov_home = excluded.tov_home,
--   tov_away = excluded.tov_away,
--   pace_home = excluded.pace_home,
--   pace_away = excluded.pace_away,
--   netrtg_home = excluded.netrtg_home,
--   netrtg_away = excluded.netrtg_away,
--   ftr_home = excluded.ftr_home,
--   ftr_away = excluded.ftr_away;
