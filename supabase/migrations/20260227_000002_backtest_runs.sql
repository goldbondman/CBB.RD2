-- Backtest run tracking for model-combo and weight experiments
-- Stores reproducible run configuration + metrics snapshots.

create table if not exists public.backtest_runs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id),
  created_at timestamptz not null default now(),
  start_date date,
  end_date date,
  selected_models text[] not null,
  weight_overrides jsonb,
  optimizer_metric text not null default 'ats' check (optimizer_metric in ('ats','mae','brier')),
  optimized_weights jsonb,
  n_games int not null default 0 check (n_games >= 0),
  verification_status text not null default 'partial' check (verification_status in ('verified','partial','conflict','rejected')),
  verification_notes text,
  run_hash text generated always as (
    md5(
      coalesce(start_date::text,'') || '|' ||
      coalesce(end_date::text,'') || '|' ||
      coalesce(array_to_string(selected_models, ','), '') || '|' ||
      coalesce(weight_overrides::text, '') || '|' ||
      optimizer_metric
    )
  ) stored
);

create table if not exists public.backtest_run_metrics (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null references public.backtest_runs(id) on delete cascade,
  model_name text not null,
  n_games int not null default 0 check (n_games >= 0),
  ats_pct numeric,
  ou_pct numeric,
  spread_mae numeric,
  spread_rmse numeric,
  brier_score numeric,
  edge_roi_sim numeric,
  created_at timestamptz not null default now(),
  unique (run_id, model_name)
);

-- Index rules
create index if not exists idx_backtest_runs_user_id on public.backtest_runs(user_id);
create index if not exists idx_backtest_runs_created_at on public.backtest_runs(created_at desc);
create index if not exists idx_backtest_runs_run_hash on public.backtest_runs(run_hash);
create index if not exists idx_backtest_run_metrics_run_id on public.backtest_run_metrics(run_id);
create index if not exists idx_backtest_run_metrics_model_name on public.backtest_run_metrics(model_name);

-- RLS
alter table public.backtest_runs enable row level security;
alter table public.backtest_run_metrics enable row level security;

-- Owner-only run table policies
create policy "backtest_runs_select_own" on public.backtest_runs
for select to authenticated
using (user_id = auth.uid());

create policy "backtest_runs_insert_own" on public.backtest_runs
for insert to authenticated
with check (user_id = auth.uid());

create policy "backtest_runs_update_own" on public.backtest_runs
for update to authenticated
using (user_id = auth.uid())
with check (user_id = auth.uid());

create policy "backtest_runs_delete_own" on public.backtest_runs
for delete to authenticated
using (user_id = auth.uid());

-- Metrics table policies inherit ownership via run_id
create policy "backtest_run_metrics_select_own" on public.backtest_run_metrics
for select to authenticated
using (
  exists (
    select 1
    from public.backtest_runs r
    where r.id = backtest_run_metrics.run_id
      and r.user_id = auth.uid()
  )
);

create policy "backtest_run_metrics_insert_own" on public.backtest_run_metrics
for insert to authenticated
with check (
  exists (
    select 1
    from public.backtest_runs r
    where r.id = backtest_run_metrics.run_id
      and r.user_id = auth.uid()
  )
);

create policy "backtest_run_metrics_update_own" on public.backtest_run_metrics
for update to authenticated
using (
  exists (
    select 1
    from public.backtest_runs r
    where r.id = backtest_run_metrics.run_id
      and r.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1
    from public.backtest_runs r
    where r.id = backtest_run_metrics.run_id
      and r.user_id = auth.uid()
  )
);

create policy "backtest_run_metrics_delete_own" on public.backtest_run_metrics
for delete to authenticated
using (
  exists (
    select 1
    from public.backtest_runs r
    where r.id = backtest_run_metrics.run_id
      and r.user_id = auth.uid()
  )
);

-- Deterministic upsert strategy (server-side):
-- backtest_runs: insert ... on conflict (run_hash, user_id) do update set ...
-- backtest_run_metrics: insert ... on conflict (run_id, model_name) do update set ...

-- Supporting unique constraint for deterministic run upsert
create unique index if not exists ux_backtest_runs_user_hash on public.backtest_runs(user_id, run_hash);
